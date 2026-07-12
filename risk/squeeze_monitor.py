"""軋空預警模組：監控融券餘額暴增、券資比與融券使用率。"""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SqueezeAlert:
    triggered: bool
    reasons: list = field(default_factory=list)
    short_balance_now: float = 0.0
    short_balance_surge: float = 0.0   # 最近 surge_days 的融券餘額倍數
    short_margin_ratio: float = 0.0    # 券資比
    short_utilization: float = 0.0     # 融券使用率


def check_squeeze_risk(
    margin_df: pd.DataFrame,
    surge_days: int = 3,
    surge_multiplier: float = 3.0,
    ratio_threshold: float = 0.3,
    utilization_threshold: float = 0.8,
) -> SqueezeAlert:
    """對放空腳做軋空風險檢查。

    - 融券餘額在 surge_days 天內暴增超過 surge_multiplier 倍 → 警示
      （擁擠的空單是軋空的燃料）
    - 券資比超過 ratio_threshold → 警示
    - 融券使用率超過 utilization_threshold → 警示（快沒券可空、也可能被強制回補）
    """
    reasons = []
    latest = margin_df.iloc[-1]

    surge = float("nan")
    if len(margin_df) > surge_days:
        base = margin_df["short_balance"].iloc[-1 - surge_days]
        if base > 0:
            surge = latest["short_balance"] / base
            if surge >= surge_multiplier:
                reasons.append(
                    f"融券餘額 {surge_days} 天內增為 {surge:.1f} 倍"
                    f"（{base:.0f} -> {latest['short_balance']:.0f} 張）"
                )

    ratio = latest["short_margin_ratio"]
    if pd.notna(ratio) and ratio >= ratio_threshold:
        reasons.append(f"券資比 {ratio:.1%} 超過門檻 {ratio_threshold:.0%}")

    util = latest["short_utilization"]
    if pd.notna(util) and util >= utilization_threshold:
        reasons.append(f"融券使用率 {util:.1%} 超過門檻 {utilization_threshold:.0%}")

    return SqueezeAlert(
        triggered=bool(reasons),
        reasons=reasons,
        short_balance_now=float(latest["short_balance"]),
        short_balance_surge=surge,
        short_margin_ratio=float(ratio) if pd.notna(ratio) else float("nan"),
        short_utilization=float(util) if pd.notna(util) else float("nan"),
    )
