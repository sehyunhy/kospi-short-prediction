# ============================================================
#  src/features/engineer.py
#  파생 피처 생성, 시가총액 가중 집계, 시계열 피처 엔지니어링
# ============================================================

import logging

import numpy as np
import pandas as pd

from config.config import (
    TARGET_COL, WT_SRC_COLS,
    LAG_PERIODS, ROLLING_WINS, ZSCORE_WIN,
)

logger = logging.getLogger(__name__)


# ── 1. 종목 수준 파생 피처 ───────────────────────────────────
def add_stock_features(
    df_short: pd.DataFrame,
    df_price: pd.DataFrame,
) -> pd.DataFrame:
    """
    공매도·주가 데이터를 종목 수준에서 병합하고
    파생 피처(intensity, pressure, uptick ratio 등)를 추가한다.
    """
    df_short = df_short.sort_values(["ISU_CD", "date"]).copy()

    df_short["short_intensity"]  = df_short["short_val"]  / (df_short["acc_trdval"] + 1)
    df_short["balance_pressure"] = df_short["net_short_balance_qty"] / (df_short["LIST_SHRS"] + 1)
    df_short["uptick_ratio"]     = df_short["uptick_appl_qty"]  / (df_short["short_qty"] + 1)
    df_short["excpt_ratio"]      = df_short["uptick_excpt_qty"] / (df_short["short_qty"] + 1)

    df_price = df_price.copy()
    df_price["price_range_rt"] = (
        (df_price["TDD_HGPRC"] - df_price["TDD_LWPRC"]) / (df_price["TDD_OPNPRC"] + 1)
    )
    df_price["close_open_rt"] = (
        (df_price["TDD_CLSPRC"] - df_price["TDD_OPNPRC"]) / (df_price["TDD_OPNPRC"] + 1)
    )
    df_price["vol_log"] = np.log1p(df_price["ACC_TRDVOL"])

    merge_cols = ["date", "ISU_CD", "CHG_RT", "price_range_rt", "close_open_rt", "vol_log", "TDD_CLSPRC"]
    df_m = df_short.merge(df_price[merge_cols], on=["date", "ISU_CD"], how="left")

    logger.info("종목 수준 파생 피처 생성 완료 | shape: %s", df_m.shape)
    return df_m


# ── 2. 시가총액 가중 일별 집계 ──────────────────────────────
def aggregate_daily(df_m: pd.DataFrame, top50: pd.DataFrame) -> pd.DataFrame:
    """
    종목별 가중치(시가총액 비중)를 적용해 일별 시장 수준 피처를 집계한다.
    """
    mktcap_dict  = top50.set_index("ISU_CD")["MKTCAP"].to_dict()
    total_mktcap = sum(mktcap_dict.values())
    df_m = df_m.copy()
    df_m["wt"] = df_m["ISU_CD"].map(mktcap_dict) / total_mktcap

    for col in WT_SRC_COLS:
        if col in df_m.columns:
            df_m[f"wt_{col}"] = df_m[col] * df_m["wt"]

    agg = {f"wt_{c}": (f"wt_{c}", "sum") for c in WT_SRC_COLS if f"wt_{c}" in df_m.columns}
    agg.update(
        {
            "short_qty_sum":   ("short_qty",            "sum"),
            "short_val_sum":   ("short_val",             "sum"),
            "acc_trdval_sum":  ("acc_trdval",            "sum"),
            "net_bal_qty_sum": ("net_short_balance_qty", "sum"),
            "n_stocks":        ("ISU_CD",                "count"),
        }
    )

    daily = (
        df_m.groupby("date").agg(**agg).reset_index().sort_values("date").reset_index(drop=True)
    )

    # 공매도 영향 지수 (연속값 → 이후 분위수 기반 라벨링 사용)
    daily[TARGET_COL] = daily["wt_short_ratio"] + 0.5 * daily["wt_balance_ratio"]

    logger.info("일별 집계 완료: %d 거래일", len(daily))
    return daily


# ── 3. 시계열 피처 엔지니어링 ───────────────────────────────
def add_time_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    달력 피처 / Lag / Rolling / 모멘텀 / Z-score 피처를 추가한다.
    """
    daily = daily.copy()

    # 달력
    daily["dayofweek"]    = daily["date"].dt.dayofweek
    daily["is_monday"]    = (daily["dayofweek"] == 0).astype(int)
    daily["is_friday"]    = (daily["dayofweek"] == 4).astype(int)
    daily["month"]        = daily["date"].dt.month
    daily["quarter"]      = daily["date"].dt.quarter
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["is_month_end"] = daily["date"].dt.is_month_end.astype(int)
    daily["is_qtr_end"]   = daily["date"].dt.is_quarter_end.astype(int)

    # Lag 피처
    lag_base = [
        TARGET_COL, "wt_short_ratio", "wt_balance_ratio",
        "wt_short_intensity", "wt_balance_pressure",
        "wt_uptick_ratio", "wt_CHG_RT", "wt_vol_log", "wt_excpt_ratio",
    ]
    for col in lag_base:
        if col not in daily.columns:
            continue
        for lag in LAG_PERIODS:
            daily[f"{col}_lag{lag}"] = daily[col].shift(lag)

    # Rolling 피처
    roll_base = [
        TARGET_COL, "wt_short_ratio", "wt_balance_ratio",
        "wt_CHG_RT", "wt_vol_log", "wt_short_intensity",
    ]
    for col in roll_base:
        if col not in daily.columns:
            continue
        s = daily[col].shift(1)
        for win in ROLLING_WINS:
            daily[f"{col}_roll{win}_mean"] = s.rolling(win).mean()
            daily[f"{col}_roll{win}_std"]  = s.rolling(win).std()
            daily[f"{col}_roll{win}_max"]  = s.rolling(win).max()
            daily[f"{col}_roll{win}_min"]  = s.rolling(win).min()

    # 모멘텀 / 가속도 / 교차항
    daily["short_mom_5d"]   = daily[TARGET_COL].shift(1) - daily[TARGET_COL].shift(6)
    daily["short_mom_3d"]   = daily[TARGET_COL].shift(1) - daily[TARGET_COL].shift(4)
    daily["short_accel"]    = (
        daily[TARGET_COL].shift(1)
        - 2 * daily[TARGET_COL].shift(2)
        + daily[TARGET_COL].shift(3)
    )
    daily["vol_chg_cross"]   = daily["wt_vol_log"].shift(1) * daily["wt_CHG_RT"].shift(1)
    daily["short_bal_cross"] = daily["wt_short_ratio"].shift(1) * daily["wt_balance_ratio"].shift(1)

    # Z-score (rolling 20일)
    for col in ["wt_short_ratio", "wt_balance_ratio", TARGET_COL]:
        if col not in daily.columns:
            continue
        mu  = daily[col].shift(1).rolling(ZSCORE_WIN).mean()
        sig = daily[col].shift(1).rolling(ZSCORE_WIN).std()
        daily[f"{col}_zscore"] = (daily[col].shift(1) - mu) / (sig + 1e-9)

    daily.dropna(inplace=True)
    daily.reset_index(drop=True, inplace=True)

    logger.info("시계열 피처 생성 완료: %d 거래일, 피처 수: %d", len(daily), daily.shape[1])
    return daily


# ── 4. Binary 타겟 생성 (Extreme-State) ─────────────────────
def make_binary_target(
    daily: pd.DataFrame,
    q_low: float = 0.30,
    q_high: float = 0.70,
) -> tuple[pd.DataFrame, float, float]:
    """
    공매도 영향 지수의 분위수를 기준으로 Binary 라벨을 생성한다.
    중립 구간(q_low < val < q_high)은 제거한다.

    Returns
    -------
    daily_model : 라벨이 붙은 DataFrame (중립 제거)
    q30, q70    : 분위수 기준값
    """
    from config.config import CLASS_COL, TARGET_COL

    impact_vals = daily[TARGET_COL].values
    q30 = daily[TARGET_COL].quantile(q_low)
    q70 = daily[TARGET_COL].quantile(q_high)

    target_labels, valid_idx = [], []
    for i in range(len(daily) - 1):
        v = impact_vals[i + 1]
        if v >= q70:
            target_labels.append(1)
            valid_idx.append(i)
        elif v <= q30:
            target_labels.append(0)
            valid_idx.append(i)

    daily_model = daily.iloc[valid_idx].copy().reset_index(drop=True)
    daily_model[CLASS_COL] = target_labels

    from collections import Counter
    dist = Counter(target_labels)
    total_excl = len(daily) - 1 - len(valid_idx)
    logger.info(
        "Binary 타겟 생성 | q30=%.6f  q70=%.6f | "
        "저압력=%d  고압력=%d  제외=%d (%.1f%%)",
        q30, q70,
        dist[0], dist[1],
        total_excl, total_excl / (len(daily) - 1) * 100,
    )
    return daily_model, q30, q70
