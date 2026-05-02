# ============================================================
#  src/models/trainer.py
#  시계열 교차검증 + 전체 데이터 최종 학습 + 앙상블 구성
# ============================================================

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

from config.config import RANDOM_SEED, N_SPLITS
from src.models.definitions import (
    build_base_models, LINEAR_MODELS, NONLINEAR_MODELS,
)

logger = logging.getLogger(__name__)


# ── 교차검증 ─────────────────────────────────────────────────
def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    n_splits: int = N_SPLITS,
) -> tuple[dict, str]:
    """
    TimeSeriesSplit 교차검증을 수행하고 각 모델의 성능 지표를 반환한다.

    Returns
    -------
    cv_results : {모델명: {metric: [fold별 값]}}
    best_base  : 비선형 모델 중 F1-binary 최고 모델명
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = {
        name: {"acc": [], "f1_binary": [], "f1_macro": [], "f1_weighted": []}
        for name in build_base_models()
    }

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler = RobustScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # fold마다 fresh 인스턴스
        models = build_base_models()

        for name, model in models.items():
            try:
                if name in LINEAR_MODELS:
                    model.fit(X_tr_s, y_tr)
                    pred = model.predict(X_val_s)
                else:
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_val)

                cv_results[name]["acc"].append(accuracy_score(y_val, pred))
                cv_results[name]["f1_binary"].append(
                    f1_score(y_val, pred, average="binary", zero_division=0)
                )
                cv_results[name]["f1_macro"].append(
                    f1_score(y_val, pred, average="macro", zero_division=0)
                )
                cv_results[name]["f1_weighted"].append(
                    f1_score(y_val, pred, average="weighted", zero_division=0)
                )
            except Exception as exc:
                logger.warning("Fold %d | %s 오류: %s", fold, name, exc)
                for k in cv_results[name]:
                    cv_results[name][k].append(0.0)

        logger.info(
            "Fold %d 완료 | %s ~ %s",
            fold,
            dates[val_idx[0]],
            dates[val_idx[-1]],
        )

    _print_cv_summary(cv_results)
    best_base = max(
        NONLINEAR_MODELS,
        key=lambda n: np.mean(cv_results[n]["f1_binary"]),
    )
    logger.info("★ 비선형 최고 모델 (F1-binary): %s", best_base)
    return cv_results, best_base


def _print_cv_summary(cv_results: dict) -> None:
    header = f"{'모델':<22} {'Accuracy':<12} {'F1-binary':<12} {'F1-macro':<12} {'F1-weighted'}"
    logger.info("\n%s\n%s", header, "-" * 72)
    for name, res in cv_results.items():
        tag = " ← 선형" if name in LINEAR_MODELS else ""
        logger.info(
            "  %-20s  %-12.4f %-12.4f %-12.4f %.4f%s",
            name,
            np.mean(res["acc"]),
            np.mean(res["f1_binary"]),
            np.mean(res["f1_macro"]),
            np.mean(res["f1_weighted"]),
            tag,
        )


# ── 최종 학습 ────────────────────────────────────────────────
def train_final(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[dict, RobustScaler]:
    """
    전체 데이터로 각 모델을 최종 학습한다.

    Returns
    -------
    trained      : {모델명: 학습된 모델}
    scaler_final : 선형 모델용 스케일러
    """
    scaler_final = RobustScaler()
    X_s = scaler_final.fit_transform(X)

    base_models = build_base_models()
    trained: dict[str, Any] = {}

    for name, model in base_models.items():
        if name in LINEAR_MODELS:
            model.fit(X_s, y)
        else:
            model.fit(X, y)
        trained[name] = model
        logger.info("  ✔ %s 학습 완료", name)

    return trained, scaler_final


# ── 앙상블 ───────────────────────────────────────────────────
def build_ensembles(
    X: np.ndarray,
    y: np.ndarray,
    trained: dict,
    n_splits: int = N_SPLITS,
) -> tuple[Any, Any, dict, str]:
    """
    Soft Voting + Stacking 앙상블을 구성하고 CV 성능을 비교한다.

    Returns
    -------
    voting_clf     : VotingClassifier (학습 완료)
    stacking_clf   : StackingClassifier (학습 완료)
    ens_cv         : 앙상블 CV 결과
    best_ensemble  : 성능이 더 좋은 앙상블 이름
    """
    # ── Soft Voting ───────────────────────────────────────────
    voting_clf = VotingClassifier(
        estimators=[(n, trained[n]) for n in NONLINEAR_MODELS],
        voting="soft", n_jobs=-1,
    )
    voting_clf.fit(X, y)
    logger.info("  ✔ Soft Voting 앙상블 학습 완료")

    # ── Stacking ──────────────────────────────────────────────
    stacking_clf = StackingClassifier(
        estimators=[(n, trained[n]) for n in NONLINEAR_MODELS],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=RANDOM_SEED,
        ),
        cv=3, passthrough=False, n_jobs=-1,
    )
    stacking_clf.fit(X, y)
    logger.info("  ✔ Stacking 앙상블 학습 완료")

    # ── 앙상블 CV ─────────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ens_cv: dict[str, dict] = {
        "SoftVoting": {"f1_weighted": []},
        "Stacking":   {"f1_weighted": []},
    }

    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        fresh_models = build_base_models()

        v_tmp = VotingClassifier(
            estimators=[(n, fresh_models[n]) for n in NONLINEAR_MODELS],
            voting="soft", n_jobs=-1,
        )
        v_tmp.fit(X_tr, y_tr)
        ens_cv["SoftVoting"]["f1_weighted"].append(
            f1_score(y_val, v_tmp.predict(X_val), average="weighted", zero_division=0)
        )

        s_tmp = StackingClassifier(
            estimators=[(n, fresh_models[n]) for n in NONLINEAR_MODELS],
            final_estimator=LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED,
            ),
            cv=3, n_jobs=-1,
        )
        s_tmp.fit(X_tr, y_tr)
        ens_cv["Stacking"]["f1_weighted"].append(
            f1_score(y_val, s_tmp.predict(X_val), average="weighted", zero_division=0)
        )

    for name, res in ens_cv.items():
        logger.info("  %-15s F1-weighted=%.4f", name, np.mean(res["f1_weighted"]))

    best_ensemble = max(ens_cv, key=lambda n: np.mean(ens_cv[n]["f1_weighted"]))
    logger.info("★ 최종 앙상블: %s", best_ensemble)

    return voting_clf, stacking_clf, ens_cv, best_ensemble
