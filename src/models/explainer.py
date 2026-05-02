# ============================================================
#  src/models/explainer.py
#  SHAP 기반 모델 해석
# ============================================================

import logging
from typing import Any

import numpy as np
import shap
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


def compute_shap(
    trained: dict[str, Any],
    X: np.ndarray,
    scaler_final: RobustScaler,
) -> dict[str, Any]:
    """
    XGBoost / LightGBM / CatBoost / LogisticRegression 에 대해
    SHAP 값을 계산하여 반환한다.

    Returns
    -------
    shap_results : {
        "xgb"  : (explainer, shap_values),
        "lgb"  : (explainer, shap_values),
        "cat"  : (explainer, shap_values),
        "lr"   : (explainer, shap_values),
    }
    """
    X_s = scaler_final.transform(X)
    results: dict[str, tuple] = {}

    logger.info("  ▷ XGBoost SHAP ...")
    exp_xgb = shap.TreeExplainer(trained["XGBoost"])
    results["xgb"] = (exp_xgb, exp_xgb.shap_values(X))

    logger.info("  ▷ LightGBM SHAP ...")
    exp_lgb = shap.TreeExplainer(trained["LightGBM"])
    results["lgb"] = (exp_lgb, exp_lgb.shap_values(X))

    logger.info("  ▷ CatBoost SHAP ...")
    exp_cat = shap.TreeExplainer(trained["CatBoost"])
    results["cat"] = (exp_cat, exp_cat.shap_values(X))

    logger.info("  ▷ LogisticRegression SHAP ...")
    exp_lr = shap.LinearExplainer(
        trained["LogisticRegression"],
        X_s,
        feature_perturbation="interventional",
    )
    results["lr"] = (exp_lr, exp_lr.shap_values(X_s))

    logger.info("  ✔ SHAP 계산 완료")
    return results


def get_mean_abs_shap(shap_vals: Any) -> np.ndarray:
    """SHAP 값의 절댓값 평균을 반환한다 (다중 클래스 호환)."""
    if isinstance(shap_vals, list):
        return np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(0)
    return np.abs(shap_vals).mean(0)
