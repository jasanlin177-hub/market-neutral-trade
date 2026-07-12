"""配對交易回測引擎（z-score 均值回歸策略）。

策略邏輯（價差 spread = log(A) - hedge_ratio * log(B)）：
- z > entry_z            → 放空價差（空 A、多 B），position = -1
- z < -entry_z           → 做多價差（多 A、空 B），position = +1
- |z| < exit_z           → 平倉，position = 0
- |z| > stop_z           → 強制停損平倉，position = 0
訊號用「前一日」z-score 決定當日部位，避免用到未來資訊。

每日策略報酬 = position(前一日) * Δspread。
Δspread ≈ A 日報酬 - hedge_ratio * B 日報酬（對數價差近似），
代表「多 1 單位 A、空 hedge_ratio 單位 B」組合的報酬。
換倉當日按周轉扣除來回交易成本（cost_per_turn，單位為報酬率）。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.zscore import rolling_zscore, spread_series


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    positions: pd.Series
    zscore: pd.Series
    trade_returns: list
    equity: pd.Series
    hedge_ratio: pd.Series = None   # walk-forward 下為逐日（分段常數）序列
    trade_days: list = None         # 每筆交易的持有天數（交易日數，非日曆天）


def _ols_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """y 對 x 的 OLS 斜率（含截距），即 hedge ratio。"""
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    return float(((x - xm) * (y - ym)).sum() / denom) if denom > 0 else float("nan")


def _generate_positions(z: pd.Series, entry_z: float, exit_z: float, stop_z: float,
                        direction: str = "both") -> pd.Series:
    """依 z-score 逐日推進部位狀態（帶狀態機：進場後續抱到出場/停損）。

    direction：
      "both"       —— 雙向：z<-entry 做多價差(+1)、z>+entry 放空價差(-1)
      "long_only"  —— 只做多價差(+1)：只在 z<-entry 進場，z>+entry 完全不碰
      "short_only" —— 只放空價差(-1)：只在 z>+entry 進場，z<-entry 完全不碰

    出場採「方向性」判斷（正確處理 exit_z=0 = 回到均值才出）：
      多方部位(+1，進場時 z 很負)：z 回升至 >= -exit_z 出場；z 再跌破 -stop_z 停損
      空方部位(-1，進場時 z 很正)：z 回落至 <= +exit_z 出場；z 再升破 +stop_z 停損
    """
    allow_long = direction in ("both", "long_only")
    allow_short = direction in ("both", "short_only")
    pos = pd.Series(0, index=z.index, dtype=float)
    current = 0.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            pos.iloc[i] = 0.0
            continue
        if current == 0.0:
            if allow_short and zi > entry_z:
                current = -1.0            # 價差過高 → 放空價差
            elif allow_long and zi < -entry_z:
                current = 1.0             # 價差過低 → 做多價差
        elif current == 1.0:              # 多方價差部位
            if zi >= -exit_z:
                current = 0.0             # 回歸至均值 → 出場
            elif zi < -stop_z:
                current = 0.0             # 續跌超標 → 停損
        else:                            # 空方價差部位(-1)
            if zi <= exit_z:
                current = 0.0             # 回歸至均值 → 出場
            elif zi > stop_z:
                current = 0.0             # 續漲超標 → 停損
        pos.iloc[i] = current
    return pos


def run_backtest(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    hedge_ratio: float,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.0,
    cost_per_turn: float = 0.0,
    direction: str = "both",
) -> BacktestResult:
    spread = spread_series(log_price_a, log_price_b, hedge_ratio)
    z = rolling_zscore(spread, lookback)
    pos = _generate_positions(z, entry_z, exit_z, stop_z, direction)

    d_spread = spread.diff()
    gross_ret = pos.shift(1) * d_spread          # 前一日部位 * 當日價差變動

    turnover = pos.diff().abs().fillna(0)
    cost = turnover * cost_per_turn
    net_ret = (gross_ret - cost).fillna(0)

    equity = (1 + net_ret).cumprod()
    trade_returns, trade_days = _extract_trades(pos, net_ret)

    return BacktestResult(
        daily_returns=net_ret,
        positions=pos,
        zscore=z,
        trade_returns=trade_returns,
        equity=equity,
        trade_days=trade_days,
    )


def _extract_trades(pos: pd.Series, net_ret: pd.Series) -> tuple:
    """從部位序列拆出每筆交易（開倉到平倉）的累積報酬與持有天數（交易日數）。

    回傳 (trade_returns, trade_days)，兩個 list 一一對應同一筆交易。
    """
    trade_returns = []
    trade_days = []
    in_trade = False
    trade_cum = 0.0
    trade_len = 0
    prev_pos = 0.0
    for i in range(len(pos)):
        p = pos.iloc[i]
        if not in_trade and prev_pos == 0.0 and p != 0.0:
            in_trade = True
            trade_cum = 0.0
            trade_len = 0
        if in_trade:
            trade_cum += net_ret.iloc[i]
            trade_len += 1
            if p == 0.0:
                trade_returns.append(trade_cum)
                trade_days.append(trade_len)
                in_trade = False
        prev_pos = p
    if in_trade:
        trade_returns.append(trade_cum)
        trade_days.append(trade_len)
    return trade_returns, trade_days


def walk_forward_backtest(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    estimation_window: int = 120,
    reestimate_every: int = 20,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.0,
    cost_per_turn: float = 0.0,
    direction: str = "both",
) -> BacktestResult:
    """Walk-forward 回測：hedge_ratio 只用過去資料估，滾動往前，無前視偏誤。

    - 每 reestimate_every 天，用前 estimation_window 天重估 hedge_ratio；
      該比例僅套用在「之後」的日子（分段常數的逐日序列）。
    - 每日報酬 = position(前一日) * (r_A - hr(前一日) * r_B)，
      hr 用持倉當時鎖定的比例，避免重估跳動污染報酬。
    - z-score 用滾動窗口（本身只看過去）判斷進出場。
    """
    n = len(log_price_a)
    hr_series = pd.Series(np.nan, index=log_price_a.index, dtype=float)

    # 從累積足夠估計資料後開始，每 reestimate_every 天重估一次
    for t in range(estimation_window, n, reestimate_every):
        win_a = log_price_a.iloc[t - estimation_window:t]
        win_b = log_price_b.iloc[t - estimation_window:t]
        hr = _ols_hedge_ratio(win_a, win_b)
        end = min(t + reestimate_every, n)
        hr_series.iloc[t:end] = hr  # 只套用在估計期「之後」

    spread = log_price_a - hr_series * log_price_b
    z = rolling_zscore(spread, lookback)
    pos = _generate_positions(z, entry_z, exit_z, stop_z, direction)

    ret_a = log_price_a.diff()
    ret_b = log_price_b.diff()
    # 用前一日鎖定的 hedge ratio 計算持倉報酬
    gross_ret = pos.shift(1) * (ret_a - hr_series.shift(1) * ret_b)

    turnover = pos.diff().abs().fillna(0)
    net_ret = (gross_ret - turnover * cost_per_turn).fillna(0)
    equity = (1 + net_ret).cumprod()
    trade_returns, trade_days = _extract_trades(pos, net_ret)

    return BacktestResult(
        daily_returns=net_ret,
        positions=pos,
        zscore=z,
        trade_returns=trade_returns,
        equity=equity,
        hedge_ratio=hr_series,
        trade_days=trade_days,
    )
