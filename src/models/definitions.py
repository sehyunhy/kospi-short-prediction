# ============================================================
#  src/models/definitions.py
#  모델 인스턴스 생성 팩토리
# ============================================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from config.config import (
    RANDOM_SEED,
    XGB_PARAMS, LGB_PARAMS, CAT_PARAMS, LOGIT_PARAMS,
)

# 모델 그룹 정의
LINEAR_MODELS    = ["LogisticRegression", "LDA"]
NONLINEAR_MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]


def build_base_models() -> dict:
    """
    모든 베이스 모델 인스턴스를 새로 생성해 반환한다.
    (매 실행마다 fresh 인스턴스 → 교차검증 leak 방지)
    """
    return {
        "LogisticRegression": LogisticRegression(
            **LOGIT_PARAMS, random_state=RANDOM_SEED
        ),
        "LDA": LDA(solver="svd"),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            **XGB_PARAMS, random_state=RANDOM_SEED
        ),
        "LightGBM": lgb.LGBMClassifier(
            **LGB_PARAMS, random_state=RANDOM_SEED
        ),
        "CatBoost": CatBoostClassifier(
            **CAT_PARAMS, random_seed=RANDOM_SEED
        ),
    }
