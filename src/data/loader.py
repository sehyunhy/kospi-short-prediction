# ============================================================
#  src/data/loader.py
#  데이터 로드 & 결측치 처리
# ============================================================

import logging
import pandas as pd

from config.config import SHORT_NUM_COLS, PRICE_NUM_COLS, TOP_N

logger = logging.getLogger(__name__)


def load_raw(short_path: str, price_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    공매도·주가 CSV를 로드하고 date 컬럼을 datetime으로 변환한다.

    Returns
    -------
    df_short, df_price
    """
    df_short = pd.read_csv(short_path, parse_dates=["date"])
    df_price = pd.read_csv(price_path, parse_dates=["date"])

    logger.info(
        "df_short %s  (%s ~ %s)",
        df_short.shape,
        df_short["date"].min().date(),
        df_short["date"].max().date(),
    )
    logger.info(
        "df_price %s  (%s ~ %s)",
        df_price.shape,
        df_price["date"].min().date(),
        df_price["date"].max().date(),
    )
    return df_short, df_price


def select_top_n(df_short: pd.DataFrame, n: int = TOP_N) -> tuple[list, pd.DataFrame]:
    """
    시가총액 평균 기준 상위 N개 종목 코드와 필터링된 DataFrame을 반환한다.

    Returns
    -------
    top_codes : list[str]
    top_df    : pd.DataFrame
    """
    mktcap_mean = (
        df_short.dropna(subset=["MKTCAP"])
        .groupby(["ISU_CD", "ISU_NM"])["MKTCAP"]
        .mean()
        .reset_index()
        .sort_values("MKTCAP", ascending=False)
    )
    top_df   = mktcap_mean.head(n)
    top_codes = top_df["ISU_CD"].tolist()

    logger.info("Top %d 종목 선정 완료 (상위 10개):", n)
    logger.info("\n%s", top_df.head(10)[["ISU_NM", "MKTCAP"]].to_string(index=False))
    return top_codes, top_df


def fill_missing(
    df_short: pd.DataFrame,
    df_price: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    공매도: forward-fill → 0 채우기
    주가  : forward-fill → backward-fill
    """
    for col in SHORT_NUM_COLS:
        if col in df_short.columns:
            df_short[col] = df_short.groupby("ISU_CD")[col].transform(
                lambda x: x.ffill().fillna(0)
            )

    for col in PRICE_NUM_COLS:
        if col in df_price.columns:
            df_price[col] = df_price.groupby("ISU_CD")[col].transform(
                lambda x: x.ffill().bfill()
            )

    remaining = df_short[SHORT_NUM_COLS].isna().sum().sum()
    logger.info("결측치 처리 완료 | 잔여 NaN: %d", remaining)
    return df_short, df_price
