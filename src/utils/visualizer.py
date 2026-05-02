# ============================================================
#  src/utils/visualizer.py
#  모든 시각화 함수 모음
# ============================================================

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit

matplotlib.rcParams["font.family"]        = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from config.config import CLASS_COLORS, FIGURE_DPI, N_SPLITS, TARGET_COL, CLASS_COL
from src.models.definitions import LINEAR_MODELS, NONLINEAR_MODELS
from src.models.explainer import get_mean_abs_shap

logger = logging.getLogger(__name__)


# ── 메인 대시보드 ─────────────────────────────────────────────
def plot_dashboard(
    daily_model: pd.DataFrame,
    cv_results: dict,
    ens_cv: dict,
    prob_dict: dict,
    shap_results: dict,
    feature_cols: list[str],
    X: np.ndarray,
    y: np.ndarray,
    next_date: pd.Timestamp,
    best_ensemble: str,
    final_ensemble: Any,
    output_path: str = "outputs/short_selling_v3.png",
) -> None:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fig = plt.figure(figsize=(24, 28))
    fig.suptitle(
        "KOSPI Top 50 공매도 영향 분류 모델 (Extreme-State Binary)",
        fontsize=18, fontweight="bold", y=0.99,
    )

    # 13-1. 공매도 영향 지수 + 클래스별 색상
    ax1 = fig.add_subplot(4, 3, 1)
    ax1.plot(daily_model["date"], daily_model[TARGET_COL], color="gray", lw=1, alpha=0.5, zorder=1)
    for cls, label, col in zip([0, 1], ["저압력(0)", "고압력(1)"], CLASS_COLORS):
        mask = daily_model[CLASS_COL] == cls
        ax1.scatter(daily_model.loc[mask, "date"], daily_model.loc[mask, TARGET_COL],
                    c=col, s=20, zorder=2 + cls, label=label)
    ax1.set_title("공매도 영향 지수 & Binary 타겟", fontsize=10)
    ax1.set_ylabel("영향 지수"); ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

    # 13-2. 클래스 분포
    ax2 = fig.add_subplot(4, 3, 2)
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    bars2 = ax2.bar(["저압력(0)", "고압력(1)"], counts, color=CLASS_COLORS, alpha=0.85)
    ax2.set_title("Binary 클래스 분포 (이상적: 50:50)", fontsize=10); ax2.set_ylabel("샘플 수")
    for b, v in zip(bars2, counts):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                 f"{v}\n({v/len(y)*100:.0f}%)", ha="center", fontsize=9)

    # 13-3. 교차검증 F1-weighted
    ax3 = fig.add_subplot(4, 3, 3)
    all_model_names = list(cv_results.keys()) + ["SoftVoting", "Stacking"]
    f1w_vals = (
        [np.mean(cv_results[n]["f1_weighted"]) for n in cv_results]
        + [np.mean(ens_cv["SoftVoting"]["f1_weighted"]),
           np.mean(ens_cv["Stacking"]["f1_weighted"])]
    )
    bar_colors3 = ["#74add1"] * 2 + ["#d73027"] * 4 + ["#1a9850", "#762a83"]
    bars3 = ax3.bar(all_model_names, f1w_vals, color=bar_colors3, alpha=0.85)
    ax3.axhline(0.8, color="red", linestyle="--", lw=1.5, label="목표 F1=0.8")
    ax3.set_title("교차검증 F1-weighted", fontsize=10)
    ax3.set_xticks(range(len(all_model_names)))
    ax3.set_xticklabels(all_model_names, rotation=20, fontsize=7)
    ax3.set_ylabel("F1-weighted"); ax3.set_ylim(0, 1.1); ax3.legend(fontsize=8)
    for b, v in zip(bars3, f1w_vals):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    # 13-4. 내일 예측 확률
    ax4 = fig.add_subplot(4, 3, 4)
    final_probs = prob_dict["Final_Ens"]
    bars4 = ax4.bar(["📉 저압력(하락)", "📈 고압력(상승)"], final_probs, color=CLASS_COLORS,
                    alpha=0.85, edgecolor="white")
    for b, v in zip(bars4, final_probs):
        ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    pred_label = "📈 고압력" if final_probs[1] > 0.5 else "📉 저압력"
    ax4.set_title(f"내일({next_date.strftime('%Y-%m-%d')}) 공매도 예측\n★ {pred_label}", fontsize=10)
    ax4.set_ylabel("확률"); ax4.set_ylim(0, 1.1)
    ax4.axhline(0.5, color="black", linestyle="--", lw=1, alpha=0.5)

    # 13-5. 모델별 P(고압력) 비교
    ax5 = fig.add_subplot(4, 3, 5)
    model_names_p = list(prob_dict.keys())
    p_up_list = [prob_dict[n][1] for n in model_names_p]
    bar_c5 = ["#d73027" if p > 0.5 else "#4575b4" for p in p_up_list]
    bars5 = ax5.bar(model_names_p, p_up_list, color=bar_c5, alpha=0.85)
    ax5.axhline(0.5, color="black", linestyle="--", lw=1.5, label="기준선 0.5")
    ax5.set_title("모델별 P(고압력/상승) 비교", fontsize=10)
    ax5.set_xticks(range(len(model_names_p)))
    ax5.set_xticklabels(model_names_p, rotation=20, fontsize=7)
    ax5.set_ylabel("P(고압력)"); ax5.set_ylim(0, 1.1); ax5.legend(fontsize=8)
    for b, v in zip(bars5, p_up_list):
        ax5.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    # 13-6. Confusion Matrix (마지막 Fold)
    ax6 = fig.add_subplot(4, 3, 6)
    last_tr, last_val = list(tscv.split(X))[-1]
    pred_last = final_ensemble.predict(X[last_val])
    cm = confusion_matrix(y[last_val], pred_last)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax6,
                xticklabels=["예측:저압력", "예측:고압력"],
                yticklabels=["실제:저압력", "실제:고압력"])
    f1_last = f1_score(y[last_val], pred_last, average="weighted", zero_division=0)
    ax6.set_title(f"Confusion Matrix ({best_ensemble}, 마지막 Fold)\nF1-weighted={f1_last:.3f}", fontsize=9)

    # 13-7~10. SHAP 피처 중요도
    shap_configs = [
        ("xgb", "XGBoost SHAP Top 15",          "#d73027", 7),
        ("lgb", "LightGBM SHAP Top 15",          "#762a83", 8),
        ("cat", "CatBoost SHAP Top 15",           "#1a9850", 9),
        ("lr",  "LogisticRegression SHAP Top 15", "#74add1", 10),
    ]
    for key, title, color, pos in shap_configs:
        ax = fig.add_subplot(4, 3, pos)
        _, shap_vals = shap_results[key]
        mi = get_mean_abs_shap(shap_vals)
        top_idx = np.argsort(mi)[-15:][::-1]
        ax.barh(np.arange(15), mi[top_idx], color=color, alpha=0.85)
        ax.set_yticks(np.arange(15))
        ax.set_yticklabels([feature_cols[i] for i in top_idx], fontsize=7)
        ax.set_title(title, fontsize=10); ax.set_xlabel("|SHAP|"); ax.invert_yaxis()

    # 13-11. 선형 vs 비선형 F1 비교
    ax11 = fig.add_subplot(4, 3, 11)
    lin_names  = LINEAR_MODELS
    nlin_names = NONLINEAR_MODELS
    all_n = lin_names + nlin_names
    all_f = ([np.mean(cv_results[n]["f1_weighted"]) for n in lin_names]
             + [np.mean(cv_results[n]["f1_weighted"]) for n in nlin_names])
    ax11.bar(all_n, all_f, color=["#74add1"] * 2 + ["#d73027"] * 4, alpha=0.85)
    ax11.axhline(0.8, color="black", linestyle="--", lw=1.5, label="목표 F1=0.8")
    ax11.set_title("선형 vs 비선형 F1-weighted", fontsize=10)
    ax11.set_xticks(range(len(all_n)))
    ax11.set_xticklabels(all_n, rotation=20, fontsize=8)
    ax11.set_ylabel("F1-weighted"); ax11.set_ylim(0, 1.1); ax11.legend(fontsize=8)
    for i, v in enumerate(all_f):
        ax11.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    # 13-12. 요일별 클래스 분포
    ax12 = fig.add_subplot(4, 3, 12)
    dow_class = pd.crosstab(daily_model["dayofweek"], daily_model[CLASS_COL])
    dow_class.index   = ["월", "화", "수", "목", "금"][: len(dow_class)]
    dow_class.columns = ["저압력(0)", "고압력(1)"]
    dow_class.plot(kind="bar", ax=ax12, color=CLASS_COLORS, alpha=0.85, edgecolor="white")
    ax12.set_title("요일별 고압력/저압력 분포", fontsize=10)
    ax12.set_xlabel(""); ax12.set_ylabel("빈도"); ax12.legend(fontsize=8)
    plt.setp(ax12.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    logger.info("  ✔ 저장: %s", output_path)


# ── SHAP Waterfall ───────────────────────────────────────────
def plot_shap_waterfall(
    shap_results: dict,
    X: np.ndarray,
    feature_cols: list[str],
    output_path: str = "outputs/shap_waterfall_v3.png",
) -> None:
    exp_xgb, shap_xgb = shap_results["xgb"]

    shap_vals_last = shap_xgb[-1] if not isinstance(shap_xgb, list) else shap_xgb[1][-1]
    base_val = (
        exp_xgb.expected_value[1]
        if isinstance(exp_xgb.expected_value, (list, np.ndarray))
        else exp_xgb.expected_value
    )

    shap_exp = shap.Explanation(
        values=shap_vals_last,
        base_values=base_val,
        data=X[-1],
        feature_names=feature_cols,
    )

    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_exp, max_display=20, show=False)
    plt.title("SHAP Waterfall (XGBoost) - 고압력(1) 클래스 기여도", fontsize=12)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    logger.info("  ✔ 저장: %s", output_path)
