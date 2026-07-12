"""價差 z-score 序列計算（供回測與即時訊號共用）。"""
import numpy as np
import pandas as pd


def spread_series(log_price_a: pd.Series, log_price_b: pd.Series, hedge_ratio: float) -> pd.Series:
    """對數價差 spread = log(A) - hedge_ratio * log(B)。"""
    return log_price_a - hedge_ratio * log_price_b


def rolling_zscore(spread: pd.Series, lookback: int = 60) -> pd.Series:
    """滾動 z-score：(spread - 滾動均值) / 滾動標準差。

    用滾動窗口（而非全樣本）計算，避免用到未來資訊（look-ahead bias）。
    前 lookback-1 期因窗口不足為 NaN。
    """
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std.replace(0, np.nan)
