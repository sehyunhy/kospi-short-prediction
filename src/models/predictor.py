# ============================================================
#  src/models/predictor.py
#  내일 공매도 영향 방향 예측
# ============================================================

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.models.definitions import LINEAR_MODELS

logger = logging.getLogger(__name__)


def predict_tomorrow(
    daily_model: pd.DataFrame,
    feature_cols: list[str],
    trained: dict[str, Any],
    scaler_final: RobustScaler,
    voting_clf: Any,
    stacking_clf: Any,
    final_ensemble: Any,
    best_ensemble: str,
) -> tuple[dict, pd.Timestamp, pd.Timestamp]:
    """
    가장 최근 데이터를 기준으로 내일의 공매도 압력 방향을 예측한다.

    Returns
    -------
    prob_dict  : {모델명: [P(저압력), P(고압력)]}
    last_date  : 예측 기준일
    next_date  : 예측 대상일 (다음 영업일)
    """
    last_feat = daily_model[feature_cols].iloc[-1].values.reshape(1, -1)
    last_date = daily_model["date"].iloc[-1]
    next_date = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=1)[0]

    last_feat_s = scaler_final.transform(last_feat)

    prob_dict: dict[str, np.ndarray] = {}
    for name, model in trained.items():
        if name in LINEAR_MODELS:
            prob_dict[name] = model.predict_proba(last_feat_s)[0]
        else:
            prob_dict[name] = model.predict_proba(last_feat)[0]

    prob_dict["SoftVoting"] = voting_clf.predict_proba(last_feat)[0]
    prob_dict["Stacking"]   = stacking_clf.predict_proba(last_feat)[0]
    prob_dict["Final_Ens"]  = final_ensemble.predict_proba(last_feat)[0]

    # 결과 로깅
    logger.info("\n  예측 기준일: %s  →  예측 대상일: %s", last_date.date(), next_date.date())
    logger.info("  %-22s %-10s %-10s %s", "모델", "P(저압력)", "P(고압력)", "예측")
    logger.info("  " + "-" * 55)

    for name, probs in prob_dict.items():
        direction = "📈 고압력" if probs[1] > 0.5 else "📉 저압력"
        tag = " ◀ 최종" if name == "Final_Ens" else ""
        logger.info("  %-22s %-10.4f %-10.4f %s%s", name, probs[0], probs[1], direction, tag)

    final_probs = prob_dict["Final_Ens"]
    pred_label  = "📈 고압력 (공매도 집중 가능)" if final_probs[1] > 0.5 else "📉 저압력 (공매도 완화 가능)"
    logger.info(
        "\n★★★ 내일(%s) 공매도 영향: %s ★★★\n    P(고압력)=%.4f | P(저압력)=%.4f | 신뢰도=%.4f",
        next_date.strftime("%Y-%m-%d"),
        pred_label,
        final_probs[1],
        final_probs[0],
        max(final_probs),
    )

    return prob_dict, last_date, next_date
