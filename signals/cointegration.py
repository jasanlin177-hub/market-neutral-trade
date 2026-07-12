"""Engle-Granger 共整合檢定模組。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def to_log(prices: pd.Series) -> pd.Series:
    """轉成對數價格。"""
    return np.log(prices)


@dataclass
class CointegrationResult:
    stat: float
    p_value: float
    is_cointegrated: bool
    hedge_ratio: float


def test_cointegration(price_a: pd.Series, price_b: pd.Series, significance_level: float = 0.05) -> CointegrationResult:
    """對兩檔股票的收盤價序列做 Engle-Granger 共整合檢定。

    hedge_ratio 來自 price_a 對 price_b 的線性迴歸係數，
    代表建立價差 spread = price_a - hedge_ratio * price_b 時的比例。
    """
    stat, p_value, _ = coint(price_a, price_b)

    x = sm.add_constant(price_b.values)
    model = sm.OLS(price_a.values, x).fit()
    hedge_ratio = model.params[1]

    return CointegrationResult(
        stat=stat,
        p_value=p_value,
        is_cointegrated=p_value < significance_level,
        hedge_ratio=hedge_ratio,
    )


def rolling_cointegration(
    dates: pd.Series,
    price_a: pd.Series,
    price_b: pd.Series,
    window: int = 120,
    step: int = 5,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """滾動窗口 Engle-Granger 檢定。

    每隔 step 個交易日取最近 window 天做一次檢定，
    回傳每個窗口的結束日、p-value、hedge ratio 與是否通過。
    """
    rows = []
    for end in range(window, len(price_a) + 1, step):
        start = end - window
        pa = price_a.iloc[start:end]
        pb = price_b.iloc[start:end]
        result = test_cointegration(pa, pb, significance_level)
        rows.append(
            {
                "window_start": dates.iloc[start],
                "window_end": dates.iloc[end - 1],
                "stat": result.stat,
                "p_value": result.p_value,
                "hedge_ratio": result.hedge_ratio,
                "is_cointegrated": result.is_cointegrated,
            }
        )
    return pd.DataFrame(rows)
