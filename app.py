# ============================================================
#  app.py
#  KOSPI Top 50 공매도 영향 예측 - Streamlit 대시보드
#
#  실행: streamlit run app.py
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

# ── 페이지 설정 (반드시 최상단) ──────────────────────────────
st.set_page_config(
    page_title="KOSPI 공매도 영향 예측",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.config import TARGET_COL, CLASS_COL, Q_LOW, Q_HIGH, TOP_N
from src.data.loader import load_raw, select_top_n, fill_missing
from src.features.engineer import (
    add_stock_features, aggregate_daily,
    add_time_features, make_binary_target,
)
from src.models.trainer import cross_validate, train_final, build_ensembles
from src.models.predictor import predict_tomorrow
from src.models.explainer import compute_shap, get_mean_abs_shap


# ════════════════════════════════════════════════════════════
#  캐시: 파이프라인 전체를 한 번만 실행
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_pipeline(short_path: str, price_path: str, top_n: int, q_low: float, q_high: float):
    """데이터 로드 → 피처 → 학습 → 예측까지 전 파이프라인."""

    # 1. 로드
    df_short, df_price = load_raw(short_path, price_path)

    # 2. Top N
    top_codes, top50 = select_top_n(df_short, n=top_n)
    df_short = df_short[df_short["ISU_CD"].isin(top_codes)].copy()
    df_price = df_price[df_price["ISU_CD"].isin(top_codes)].copy()

    # 3. 결측치
    df_short, df_price = fill_missing(df_short, df_price)

    # 4. 피처
    df_m  = add_stock_features(df_short, df_price)
    daily = aggregate_daily(df_m, top50)
    daily = add_time_features(daily)

    # 5. 타겟
    daily_model, q30, q70 = make_binary_target(daily, q_low=q_low, q_high=q_high)
    feature_cols = [
        c for c in daily_model.columns
        if c not in ["date", TARGET_COL, CLASS_COL]
        and daily_model[c].dtype != object
    ]
    X     = daily_model[feature_cols].values
    y     = daily_model[CLASS_COL].values
    dates = daily_model["date"].values

    # 6. 학습
    cv_results, best_base    = cross_validate(X, y, dates)
    trained, scaler_final    = train_final(X, y)
    voting_clf, stacking_clf, ens_cv, best_ensemble = build_ensembles(X, y, trained)
    final_ensemble = voting_clf if best_ensemble == "SoftVoting" else stacking_clf

    # 7. 예측
    prob_dict, last_date, next_date = predict_tomorrow(
        daily_model, feature_cols, trained, scaler_final,
        voting_clf, stacking_clf, final_ensemble, best_ensemble,
    )

    # 8. SHAP
    shap_results = compute_shap(trained, X, scaler_final)

    return {
        "daily_model"   : daily_model,
        "feature_cols"  : feature_cols,
        "X"             : X,
        "y"             : y,
        "q30"           : q30,
        "q70"           : q70,
        "cv_results"    : cv_results,
        "ens_cv"        : ens_cv,
        "best_ensemble" : best_ensemble,
        "final_ensemble": final_ensemble,
        "prob_dict"     : prob_dict,
        "last_date"     : last_date,
        "next_date"     : next_date,
        "shap_results"  : shap_results,
        "top50"         : top50,
    }


# ════════════════════════════════════════════════════════════
#  사이드바
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=60)
    st.title("⚙️ 설정")
    st.markdown("---")

    st.subheader("📂 데이터 경로")
    short_path = st.text_input("공매도 CSV", value="data/df_short.csv")
    price_path = st.text_input("주가 CSV",   value="data/df_price.csv")

    st.subheader("🔧 모델 파라미터")
    top_n  = st.slider("시가총액 Top N 종목", 10, 100, TOP_N, step=10)
    q_low  = st.slider("저압력 분위수 (하위)", 0.10, 0.40, Q_LOW,  step=0.05)
    q_high = st.slider("고압력 분위수 (상위)", 0.60, 0.90, Q_HIGH, step=0.05)

    st.markdown("---")
    run_btn = st.button("🚀 분석 실행", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("KOSPI 공매도 영향 예측 v3\nStreamlit Dashboard")


# ════════════════════════════════════════════════════════════
#  메인 화면
# ════════════════════════════════════════════════════════════
st.title("📉 KOSPI Top 50 공매도 영향 예측")
st.markdown("시가총액 가중 공매도 지수를 기반으로 **내일의 공매도 압력 방향**을 예측합니다.")

# 실행 전 안내
if not run_btn and "result" not in st.session_state:
    st.info("👈 사이드바에서 데이터 경로를 확인하고 **분석 실행** 버튼을 누르세요.")
    st.stop()

# 파이프라인 실행
if run_btn:
    with st.spinner("⏳ 파이프라인 실행 중... (최초 실행 시 수 분 소요)"):
        try:
            result = run_pipeline(short_path, price_path, top_n, q_low, q_high)
            st.session_state["result"] = result
            st.success("✅ 분석 완료!")
        except FileNotFoundError as e:
            st.error(f"❌ 파일을 찾을 수 없습니다: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
            st.stop()

result = st.session_state.get("result")
if result is None:
    st.stop()

# ── 변수 언패킹 ──────────────────────────────────────────────
daily_model    = result["daily_model"]
feature_cols   = result["feature_cols"]
X, y           = result["X"], result["y"]
cv_results     = result["cv_results"]
ens_cv         = result["ens_cv"]
best_ensemble  = result["best_ensemble"]
final_ensemble = result["final_ensemble"]
prob_dict      = result["prob_dict"]
last_date      = result["last_date"]
next_date      = result["next_date"]
shap_results   = result["shap_results"]
top50          = result["top50"]
q30, q70       = result["q30"], result["q70"]
final_probs    = prob_dict["Final_Ens"]


# ════════════════════════════════════════════════════════════
#  탭 구성
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 예측 결과",
    "📊 모델 성능",
    "🔍 SHAP 해석",
    "📈 데이터 탐색",
])


# ──────────────────────────────────────────────────────────
#  Tab 1. 예측 결과
# ──────────────────────────────────────────────────────────
with tab1:
    # 헤더 카드
    pred_dir  = "📈 고압력" if final_probs[1] > 0.5 else "📉 저압력"
    color_hex = "#d73027"   if final_probs[1] > 0.5 else "#4575b4"
    conf      = max(final_probs)

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color_hex}22, {color_hex}11);
        border: 2px solid {color_hex};
        border-radius: 16px;
        padding: 28px 36px;
        text-align: center;
        margin-bottom: 24px;
    ">
        <p style="font-size:14px; color:#888; margin:0;">예측 기준일: {last_date.strftime('%Y-%m-%d')}</p>
        <h1 style="font-size:48px; color:{color_hex}; margin:8px 0;">{pred_dir}</h1>
        <p style="font-size:20px; margin:0;">내일 <b>{next_date.strftime('%Y-%m-%d')} ({next_date.strftime('%a')})</b> 공매도 방향</p>
        <p style="font-size:15px; color:#666; margin-top:8px;">신뢰도: <b>{conf:.1%}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # KPI 카드
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(고압력)",  f"{final_probs[1]:.1%}", delta=f"{final_probs[1]-0.5:+.1%} vs 기준선")
    c2.metric("P(저압력)",  f"{final_probs[0]:.1%}")
    c3.metric("신뢰도",     f"{conf:.1%}")
    c4.metric("학습 데이터", f"{len(X):,} 거래일")

    st.markdown("---")

    # 모델별 예측 확률 테이블
    st.subheader("🤖 모델별 예측 확률")
    rows = []
    for name, probs in prob_dict.items():
        rows.append({
            "모델"      : name,
            "P(저압력)" : f"{probs[0]:.4f}",
            "P(고압력)" : f"{probs[1]:.4f}",
            "예측"      : "📈 고압력" if probs[1] > 0.5 else "📉 저압력",
            "최종앙상블" : "✅" if name == "Final_Ens" else "",
        })
    df_pred = pd.DataFrame(rows)
    st.dataframe(df_pred, use_container_width=True, hide_index=True)

    # 막대 차트 (plotly 대신 st.bar_chart)
    st.subheader("📊 최종 앙상블 예측 확률")
    chart_df = pd.DataFrame({
        "확률": [final_probs[0], final_probs[1]]
    }, index=["📉 저압력", "📈 고압력"])
    st.bar_chart(chart_df, color=["#4575b4"])


# ──────────────────────────────────────────────────────────
#  Tab 2. 모델 성능
# ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("📋 교차검증 성능 요약")

    from src.models.definitions import LINEAR_MODELS

    perf_rows = []
    for name, res in cv_results.items():
        perf_rows.append({
            "모델"        : name,
            "유형"        : "선형" if name in LINEAR_MODELS else "비선형",
            "Accuracy"    : f"{np.mean(res['acc']):.4f}",
            "F1-binary"   : f"{np.mean(res['f1_binary']):.4f}",
            "F1-macro"    : f"{np.mean(res['f1_macro']):.4f}",
            "F1-weighted" : f"{np.mean(res['f1_weighted']):.4f}",
        })
    for ens_name, res in ens_cv.items():
        perf_rows.append({
            "모델"        : ens_name,
            "유형"        : "앙상블",
            "Accuracy"    : "-",
            "F1-binary"   : "-",
            "F1-macro"    : "-",
            "F1-weighted" : f"{np.mean(res['f1_weighted']):.4f}",
        })

    df_perf = pd.DataFrame(perf_rows)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📈 F1-weighted 비교")

    all_names = list(cv_results.keys()) + list(ens_cv.keys())
    all_f1w   = (
        [np.mean(cv_results[n]["f1_weighted"]) for n in cv_results]
        + [np.mean(ens_cv[n]["f1_weighted"])   for n in ens_cv]
    )
    chart_f1 = pd.DataFrame({"F1-weighted": all_f1w}, index=all_names)
    st.bar_chart(chart_f1)

    st.info(f"🏆 최종 선택된 앙상블: **{best_ensemble}**  "
            f"(F1-weighted = {np.mean(ens_cv[best_ensemble]['f1_weighted']):.4f})")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🗓️ 요일별 클래스 분포")
        dow_class = pd.crosstab(daily_model["dayofweek"], daily_model[CLASS_COL])
        dow_class.index   = ["월", "화", "수", "목", "금"][: len(dow_class)]
        dow_class.columns = ["저압력(0)", "고압력(1)"]
        st.bar_chart(dow_class)

    with col_b:
        st.subheader("📦 클래스 분포")
        cls_cnt = pd.Series(y).value_counts().rename({0: "저압력(0)", 1: "고압력(1)"})
        st.bar_chart(cls_cnt)
        st.caption(f"q30 기준값: {q30:.6f}  |  q70 기준값: {q70:.6f}")


# ──────────────────────────────────────────────────────────
#  Tab 3. SHAP 해석
# ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("🔍 SHAP 피처 중요도 (Top 20)")

    model_tab = st.selectbox(
        "분석 모델 선택",
        ["XGBoost", "LightGBM", "CatBoost", "LogisticRegression"],
    )
    key_map = {
        "XGBoost"           : "xgb",
        "LightGBM"          : "lgb",
        "CatBoost"          : "cat",
        "LogisticRegression": "lr",
    }

    _, shap_vals = shap_results[key_map[model_tab]]
    mi      = get_mean_abs_shap(shap_vals)
    top_idx = np.argsort(mi)[-20:][::-1]

    shap_df = pd.DataFrame({
        "피처"       : [feature_cols[i] for i in top_idx],
        "|SHAP| 평균": mi[top_idx],
    }).set_index("피처")

    st.bar_chart(shap_df)

    st.markdown("---")
    st.subheader("📋 피처 중요도 상세 테이블")
    full_shap_df = pd.DataFrame({
        "피처"       : feature_cols,
        "|SHAP| 평균": mi,
    }).sort_values("|SHAP| 평균", ascending=False).reset_index(drop=True)
    full_shap_df.index += 1
    st.dataframe(full_shap_df, use_container_width=True)


# ──────────────────────────────────────────────────────────
#  Tab 4. 데이터 탐색
# ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("📈 공매도 영향 지수 시계열")

    impact_df = daily_model[["date", TARGET_COL, CLASS_COL]].copy()
    impact_df = impact_df.set_index("date")
    st.line_chart(impact_df[[TARGET_COL]])

    st.markdown("---")
    col_x, col_y = st.columns(2)

    with col_x:
        st.subheader("🏢 Top 10 종목 (시가총액)")
        st.dataframe(
            top50.head(10)[["ISU_NM", "MKTCAP"]]
            .rename(columns={"ISU_NM": "종목명", "MKTCAP": "시가총액(평균)"})
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    with col_y:
        st.subheader("📊 데이터 기본 통계")
        st.dataframe(
            daily_model[[TARGET_COL, "wt_short_ratio", "wt_balance_ratio"]]
            .describe()
            .round(6),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("🗃️ 원본 일별 데이터 (최근 30일)")
    st.dataframe(
        daily_model.tail(30)[["date", TARGET_COL, CLASS_COL,
                               "wt_short_ratio", "wt_balance_ratio", "dayofweek"]]
        .sort_values("date", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
