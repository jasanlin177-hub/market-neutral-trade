"""績效指標：Sharpe ratio、最大回撤、勝率等。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceStats:
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    win_rate: float          # 有損益的交易中獲利比例
    n_trades: int
    avg_trade_return: float


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（負值，例如 -0.15 代表 -15%）。"""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def compute_stats(daily_returns: pd.Series, trade_returns: list, rf: float = 0.0) -> PerformanceStats:
    """由每日報酬序列與每筆交易報酬計算績效。

    daily_returns：策略每日報酬率序列（已含成本）。
    trade_returns：每筆完整交易（進場到出場）的報酬率，用於勝率。
    """
    daily = daily_returns.dropna()
    equity = (1 + daily).cumprod()

    total_return = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    ann_return = float((1 + daily.mean()) ** TRADING_DAYS - 1) if len(daily) else 0.0
    ann_vol = float(daily.std() * np.sqrt(TRADING_DAYS)) if len(daily) else 0.0
    sharpe = float((daily.mean() - rf / TRADING_DAYS) / daily.std() * np.sqrt(TRADING_DAYS)) \
        if len(daily) and daily.std() > 0 else 0.0
    mdd = max_drawdown(equity) if len(equity) else 0.0

    wins = [r for r in trade_returns if r > 0]
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
    avg_trade = float(np.mean(trade_returns)) if trade_returns else 0.0

    return PerformanceStats(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=mdd,
        win_rate=win_rate,
        n_trades=len(trade_returns),
        avg_trade_return=avg_trade,
    )
