# ============================================================
#  config/config.py
#  프로젝트 전역 설정 관리
# ============================================================

from pathlib import Path

# ── 프로젝트 루트 ─────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# ── 데이터 경로 ───────────────────────────────────────────
SHORT_PATH = DATA_DIR / "df_short.csv"
PRICE_PATH = DATA_DIR / "df_price.csv"

# ── 모델 설정 ─────────────────────────────────────────────
TOP_N       = 50          # 시가총액 상위 N개 종목
TARGET_COL  = "short_impact"
CLASS_COL   = "extreme_state"
RANDOM_SEED = 42
N_SPLITS    = 5           # TimeSeriesSplit fold 수

# ── 타겟 분위수 (중립 구간 제거) ──────────────────────────
Q_LOW  = 0.30             # 하위 30% → 저압력(0)
Q_HIGH = 0.70             # 상위 30% → 고압력(1)

# ── 피처 엔지니어링 설정 ──────────────────────────────────
LAG_PERIODS    = [1, 2, 3, 4, 5, 10]
ROLLING_WINS   = [5, 10, 20]
ZSCORE_WIN     = 20

# ── 컬럼 정의 ─────────────────────────────────────────────
SHORT_NUM_COLS = [
    "short_qty", "uptick_appl_qty", "uptick_excpt_qty",
    "short_val", "short_ratio", "short_val_wt",
    "net_short_balance_qty", "net_short_balance_amt", "balance_ratio",
]
PRICE_NUM_COLS = [
    "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC",
    "TDD_CLSPRC", "ACC_TRDVOL", "CHG_RT",
]
WT_SRC_COLS = [
    "short_ratio", "short_val_wt", "balance_ratio", "short_intensity",
    "balance_pressure", "uptick_ratio", "excpt_ratio",
    "CHG_RT", "price_range_rt", "close_open_rt", "vol_log",
]

# ── 모델 하이퍼파라미터 ───────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=600, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.5,
    objective="binary:logistic", eval_metric="logloss",
    verbosity=0,
)
LGB_PARAMS = dict(
    n_estimators=600, max_depth=6, learning_rate=0.03,
    num_leaves=63, subsample=0.8, colsample_bytree=0.7,
    verbose=-1,
)
CAT_PARAMS = dict(
    iterations=600, depth=6, learning_rate=0.03,
    loss_function="Logloss", eval_metric="AUC",
    verbose=0,
)
LOGIT_PARAMS = dict(C=0.5, max_iter=3000, solver="lbfgs")

# ── 시각화 ────────────────────────────────────────────────
FIGURE_DPI    = 150
CLASS_COLORS  = ["#4575b4", "#d73027"]   # 저압력=파랑, 고압력=빨강
LINEAR_COLOR  = "#74add1"
