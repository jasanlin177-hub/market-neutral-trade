"""部位配置模組：計算多空兩腳的股數，維持金額中性。"""
from dataclasses import dataclass


@dataclass
class PositionPlan:
    shares_long: int      # 多方股數
    shares_short: int     # 空方股數
    notional_long: float  # 多方金額
    notional_short: float # 空方金額
    net_exposure: float   # 淨曝險（多方金額 - 空方金額）


def short_anchored_neutral(capital: float, price_long: float, price_short: float,
                           short_lot: int = 1000) -> PositionPlan:
    """空方腳整股、多方腳零股配平（符合台股放空只能整股的限制）。

    台股融券/借券都只能整股，零股無法放空；只有現股（多方）能零股。
    因此以空方腳的整張融券金額為基準，多方腳用現股零股去精準對齊其市值，
    達到金額中性。適合高價股（多方整股顆粒度太粗）。
    """
    per_leg = capital / 2
    lots_short = max(round(per_leg / (price_short * short_lot)), 1)
    shares_short = lots_short * short_lot
    notional_short = shares_short * price_short
    # 多方現股零股，配平空方市值
    shares_long = round(notional_short / price_long)
    notional_long = shares_long * price_long
    return PositionPlan(
        shares_long=shares_long,
        shares_short=shares_short,
        notional_long=notional_long,
        notional_short=notional_short,
        net_exposure=notional_long - notional_short,
    )


def dollar_neutral(capital: float, price_long: float, price_short: float, lot_size: int = 1000) -> PositionPlan:
    """金額中性配置：多空各配 capital/2，股數取整張（lot_size 股）。

    台股整股交易單位為 1000 股；若要用零股，lot_size 傳 1。
    """
    per_leg = capital / 2
    lots_long = int(per_leg / (price_long * lot_size))
    lots_short = int(per_leg / (price_short * lot_size))
    shares_long = lots_long * lot_size
    shares_short = lots_short * lot_size
    notional_long = shares_long * price_long
    notional_short = shares_short * price_short
    return PositionPlan(
        shares_long=shares_long,
        shares_short=shares_short,
        notional_long=notional_long,
        notional_short=notional_short,
        net_exposure=notional_long - notional_short,
    )
