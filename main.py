# ============================================================
#  main.py
#  KOSPI Top 50 공매도 영향 예측 파이프라인 진입점
#
#  실행 방법:
#    python main.py
#    python main.py --short data/df_short.csv --price data/df_price.csv
# ============================================================

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (VSCode에서 바로 실행 가능하도록)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import (
    SHORT_PATH, PRICE_PATH, TOP_N, TARGET_COL, CLASS_COL,
    Q_LOW, Q_HIGH,
)
from src.utils.logger import setup_logger
from src.data.loader import load_raw, select_top_n, fill_missing
from src.features.engineer import (
    add_stock_features, aggregate_daily, add_time_features, make_binary_target,
)
from src.models.trainer import cross_validate, train_final, build_ensembles
from src.models.predictor import predict_tomorrow
from src.models.explainer import compute_shap
from src.utils.visualizer import plot_dashboard, plot_shap_waterfall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KOSPI Top 50 공매도 영향 예측")
    parser.add_argument("--short",   type=str, default=str(SHORT_PATH), help="공매도 CSV 경로")
    parser.add_argument("--price",   type=str, default=str(PRICE_PATH), help="주가 CSV 경로")
    parser.add_argument("--top_n",   type=int, default=TOP_N,           help="상위 N개 종목 수")
    parser.add_argument("--no_plot", action="store_true",               help="시각화 생략")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    logger = setup_logger()

    # ── Step 1. 데이터 로드 ───────────────────────────────────
    logger.info("=" * 65)
    logger.info("  [Step 1] 데이터 로드")
    logger.info("=" * 65)
    df_short, df_price = load_raw(args.short, args.price)

    # ── Step 2. Top N 종목 선정 ───────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 2] 시가총액 기준 Top %d 종목 선정", args.top_n)
    logger.info("=" * 65)
    top_codes, top50 = select_top_n(df_short, n=args.top_n)
    df_short = df_short[df_short["ISU_CD"].isin(top_codes)].copy()
    df_price = df_price[df_price["ISU_CD"].isin(top_codes)].copy()

    # ── Step 3. 결측치 처리 ───────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 3] 결측치 처리")
    logger.info("=" * 65)
    df_short, df_price = fill_missing(df_short, df_price)

    # ── Step 4. 파생 피처 & 일별 집계 ────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 4] 파생 피처 & 일별 집계")
    logger.info("=" * 65)
    df_m  = add_stock_features(df_short, df_price)
    daily = aggregate_daily(df_m, top50)

    # ── Step 5. 시계열 피처 엔지니어링 ───────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 5] 시계열 피처 생성")
    logger.info("=" * 65)
    daily = add_time_features(daily)

    # ── Step 6. Binary 타겟 생성 ─────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 6] Binary 타겟 생성 (Extreme-State)")
    logger.info("=" * 65)
    daily_model, q30, q70 = make_binary_target(daily, q_low=Q_LOW, q_high=Q_HIGH)

    feature_cols = [
        c for c in daily_model.columns
        if c not in ["date", TARGET_COL, CLASS_COL]
        and daily_model[c].dtype != object
    ]
    X     = daily_model[feature_cols].values
    y     = daily_model[CLASS_COL].values
    dates = daily_model["date"].values
    logger.info("▶ 피처 수: %d  |  학습 데이터: %d 거래일", len(feature_cols), len(X))

    # ── Step 7. 시계열 교차검증 ──────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 7] 시계열 교차검증")
    logger.info("=" * 65)
    cv_results, best_base = cross_validate(X, y, dates)

    # ── Step 8. 전체 데이터 최종 학습 ────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 8] 전체 데이터로 최종 학습")
    logger.info("=" * 65)
    trained, scaler_final = train_final(X, y)

    # ── Step 9. 앙상블 ────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 9] 앙상블 구성 (Soft Voting + Stacking)")
    logger.info("=" * 65)
    voting_clf, stacking_clf, ens_cv, best_ensemble = build_ensembles(X, y, trained)
    final_ensemble = voting_clf if best_ensemble == "SoftVoting" else stacking_clf

    # ── Step 10. 내일 예측 ───────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 10] 내일 공매도 영향 방향 예측")
    logger.info("=" * 65)
    prob_dict, last_date, next_date = predict_tomorrow(
        daily_model, feature_cols, trained, scaler_final,
        voting_clf, stacking_clf, final_ensemble, best_ensemble,
    )

    # ── Step 11. SHAP 분석 ───────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  [Step 11] SHAP 분석")
    logger.info("=" * 65)
    shap_results = compute_shap(trained, X, scaler_final)

    # ── Step 12. 시각화 ──────────────────────────────────────
    if not args.no_plot:
        logger.info("\n" + "=" * 65)
        logger.info("  [Step 12] 시각화")
        logger.info("=" * 65)
        plot_dashboard(
            daily_model=daily_model,
            cv_results=cv_results,
            ens_cv=ens_cv,
            prob_dict=prob_dict,
            shap_results=shap_results,
            feature_cols=feature_cols,
            X=X,
            y=y,
            next_date=next_date,
            best_ensemble=best_ensemble,
            final_ensemble=final_ensemble,
        )
        plot_shap_waterfall(shap_results, X, feature_cols)

    # ── Step 13. 최종 요약 ───────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  ★ 최종 결과 요약")
    logger.info("=" * 65)

    final_probs = prob_dict["Final_Ens"]
    pred_label  = (
        "📈 고압력 (공매도 집중 가능)"
        if final_probs[1] > 0.5
        else "📉 저압력 (공매도 완화 가능)"
    )

    logger.info("  분석 대상  : KOSPI 시가총액 Top %d개 종목", args.top_n)
    logger.info("  예측 기준일: %s", last_date.strftime("%Y-%m-%d"))
    logger.info("  예측 방식  : Extreme-State Binary (고압력/저압력)")
    logger.info("  중립 제외  : q30=%.6f ~ q70=%.6f", q30, q70)
    logger.info("\n  ★ 내일(%s) 예측: %s", next_date.strftime("%Y-%m-%d"), pred_label)
    logger.info(
        "     P(고압력)=%.4f  |  P(저압력)=%.4f  |  신뢰도=%.4f",
        final_probs[1], final_probs[0], max(final_probs),
    )

    logger.info("\n  [모델별 CV F1-weighted]")
    import numpy as np
    all_f1 = (
        [(n, np.mean(cv_results[n]["f1_weighted"])) for n in cv_results]
        + [("SoftVoting", np.mean(ens_cv["SoftVoting"]["f1_weighted"])),
           ("Stacking",   np.mean(ens_cv["Stacking"]["f1_weighted"]))]
    )
    from src.models.definitions import LINEAR_MODELS
    for name, f1w in sorted(all_f1, key=lambda x: -x[1]):
        tag = " [선형]" if name in LINEAR_MODELS else ""
        ok  = " ✅"    if f1w >= 0.7            else ""
        logger.info("    %-22s F1-weighted=%.4f%s%s", name, f1w, tag, ok)

    logger.info("\n" + "=" * 65)
    logger.info("  ✅ 최종 앙상블(%s) 예측: %s", best_ensemble, pred_label)
    logger.info("  출력: outputs/short_selling_v3.png  /  outputs/shap_waterfall_v3.png")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
