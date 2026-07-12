"""停損模組：以價差 z-score 判斷是否觸發停損。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SpreadState:
    zscore: float          # 當前價差 z-score
    spread: float          # 當前價差
    spread_mean: float     # 回看期間價差均值
    spread_std: float      # 回看期間價差標準差
    stop_triggered: bool   # 是否觸發停損


def spread_zscore(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    hedge_ratio: float,
    lookback: int = 60,
    stop_z: float = 3.0,
) -> SpreadState:
    """計算 spread = log(A) - hedge_ratio * log(B) 的當前 z-score。

    z-score 絕對值超過 stop_z 即觸發停損——代表價差偏離已超出
    進場假設的回歸範圍，該認錯出場而不是凹單。
    """
    spread = log_price_a - hedge_ratio * log_price_b
    recent = spread.iloc[-lookback:]
    mean, std = recent.mean(), recent.std()
    z = (spread.iloc[-1] - mean) / std if std > 0 else np.nan
    return SpreadState(
        zscore=z,
        spread=spread.iloc[-1],
        spread_mean=mean,
        spread_std=std,
        stop_triggered=bool(abs(z) > stop_z) if not np.isnan(z) else False,
    )
