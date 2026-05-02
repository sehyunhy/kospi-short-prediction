# ============================================================
#  src/utils/logger.py
#  프로젝트 공통 로거 설정
# ============================================================

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "kospi_short",
    level: int = logging.INFO,
    log_file: str | None = "outputs/run.log",
) -> logging.Logger:
    """
    콘솔 + 파일(선택) 핸들러를 가진 로거를 생성·반환한다.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:          # 중복 핸들러 방지
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 파일 핸들러 (옵션)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
