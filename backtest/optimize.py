"""參數最佳化與診斷：網格搜尋（含樣本外驗證）+ 均值回歸半衰期。

核心誠實原則：任何「最佳」參數都必須用樣本外(out-of-sample)驗證，
否則只是在歷史雜訊上 curve-fitting。本模組刻意同時輸出樣本內與樣本外績效，
讓「樣本內調出來的好參數在樣本外是否還靈」無所遁形。
"""
import math
from itertools import product

import numpy as np
import pandas as pd

from backtest.engine import _ols_hedge_ratio, walk_forward_backtest
from signals.zscore import spread_series

TRADING_DAYS = 252


def half_life(spread: pd.Series) -> float:
    """均值回歸半衰期（交易日）：價差偏離後回到一半距離所需天數。

    用 AR(1) 迴歸 Δspread_t = a + b·spread_{t-1}，
    若 b<0（均值回歸）則 half-life = -ln(2)/b；b>=0（發散或隨機漫步）回傳 NaN。
    半衰期越短＝回歸越快，可作為 lookback / 持有期的自然尺度。
    """
    s = spread.dropna()
    lagged = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lagged, delta = lagged.align(delta, join="inner")
    if len(lagged) < 10:
        return float("nan")
    b = _ols_hedge_ratio(delta, lagged)  # Δ 對 lagged 的斜率
    if b >= 0:
        return float("nan")
    return -math.log(2) / b


def _sharpe(daily: pd.Series) -> float:
    d = daily.dropna()
    if len(d) == 0 or d.std() == 0:
        return 0.0
    return float(d.mean() / d.std() * math.sqrt(TRADING_DAYS))


def _total_return(daily: pd.Series) -> float:
    d = daily.dropna()
    return float((1 + d).prod() - 1) if len(d) else 0.0


def grid_search(
    log_a: pd.Series,
    log_b: pd.Series,
    entry_grid=(1.5, 2.0, 2.5, 3.0),
    exit_grid=(0.0, 0.5, 1.0),
    stop_grid=(3.0, 4.0, 5.0),
    split_frac: float = 0.7,
    estimation_window: int = 120,
    reestimate_every: int = 20,
    lookback: int = 60,
    cost_per_turn: float = 0.002,
) -> pd.DataFrame:
    """對 (entry, exit, stop) 網格做 walk-forward 回測，分別回報樣本內/樣本外 Sharpe。

    每組參數只跑一次 walk-forward（hedge ratio 本就滾動估計），
    再把每日報酬序列切成前 split_frac（樣本內，模擬「用來調參的歷史」）
    與後段（樣本外，模擬「調完才遇到的未來」）分別計算績效。
    """
    n = len(log_a)
    split = int(n * split_frac)
    rows = []
    for entry, exit_, stop in product(entry_grid, exit_grid, stop_grid):
        if exit_ >= entry or stop <= entry:
            continue  # 不合理組合：出場門檻須低於進場、停損須高於進場
        res = walk_forward_backtest(
            log_a, log_b,
            estimation_window=estimation_window, reestimate_every=reestimate_every,
            lookback=lookback, entry_z=entry, exit_z=exit_, stop_z=stop,
            cost_per_turn=cost_per_turn,
        )
        is_ret = res.daily_returns.iloc[:split]
        oos_ret = res.daily_returns.iloc[split:]
        n_trades = int((res.positions.diff().abs() > 0).sum())
        rows.append({
            "entry_z": entry, "exit_z": exit_, "stop_z": stop,
            "in_sample_sharpe": _sharpe(is_ret),
            "in_sample_return": _total_return(is_ret),
            "out_sample_sharpe": _sharpe(oos_ret),
            "out_sample_return": _total_return(oos_ret),
            "n_trades": n_trades,
        })
    df = pd.DataFrame(rows)
    # 依「樣本內 Sharpe」排序——這正是天真最佳化會挑的順序，方便對照樣本外
    return df.sort_values("in_sample_sharpe", ascending=False).reset_index(drop=True)
