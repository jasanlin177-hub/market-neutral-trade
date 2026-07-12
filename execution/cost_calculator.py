"""成本試算模組：把各工具的交易成本算進去，估算損益兩平點。

費率為台股一般預設值，實際以你券商與主管機關公告為準，皆可在 config 覆寫。
- 股票手續費：成交金額 0.1425%（買賣各收），多數券商有折讓
- 證交稅：賣出時 0.3%（現股）；當沖 0.15%
- 融資利息：年化約 6%，按持有天數計
- 借券費（融券）：年化約 0.08%~0.16%（券商收），另有標借費（軋空時飆高）
- 期貨手續費：每口固定（非比例），交易稅 0.002%
- 權證：手續費同股票；證交稅賣出 0.1%（低於現股）；無利息但有「時間價值衰減(theta)」
  這個隱形成本——持有越久、越接近到期，光是時間流逝就會侵蝕權證價值。
"""
from dataclasses import dataclass

from execution.warrant_pricing import bs_price


@dataclass
class LegCost:
    tool: str
    entry_cost: float      # 進場成本（手續費等）
    exit_cost: float       # 出場成本（手續費 + 稅）
    holding_cost: float    # 持有成本（利息/借券費，依天數）
    total_cost: float


def stock_long_cost(notional: float, days: int, fee_rate: float, tax_rate: float,
                    margin_interest_rate: float, financing_pct: float, fee_discount: float) -> LegCost:
    """融資做多的成本。notional = 進場市值。"""
    fee_entry = notional * fee_rate * fee_discount
    fee_exit = notional * fee_rate * fee_discount
    tax = notional * tax_rate
    interest = notional * financing_pct * margin_interest_rate * days / 365
    total = fee_entry + fee_exit + tax + interest
    return LegCost("融資買進", fee_entry, fee_exit + tax, interest, total)


def stock_cash_long_cost(notional: float, fee_rate: float, tax_rate: float,
                         fee_discount: float) -> LegCost:
    """現股（現金）買進的成本，無融資利息。零股買進適用（零股不可融資）。"""
    fee_entry = notional * fee_rate * fee_discount
    fee_exit = notional * fee_rate * fee_discount
    tax = notional * tax_rate
    total = fee_entry + fee_exit + tax
    return LegCost("現股買進", fee_entry, fee_exit + tax, 0.0, total)


def stock_short_cost(notional: float, days: int, fee_rate: float, tax_rate: float,
                     borrow_rate: float, fee_discount: float) -> LegCost:
    """融券放空的成本。"""
    fee_entry = notional * fee_rate * fee_discount
    fee_exit = notional * fee_rate * fee_discount
    tax = notional * tax_rate
    borrow = notional * borrow_rate * days / 365
    total = fee_entry + fee_exit + tax + borrow
    return LegCost("融券賣出", fee_entry, fee_exit + tax, borrow, total)


def futures_cost(notional: float, contracts: int, fee_per_contract: float, futures_tax_rate: float) -> LegCost:
    """個股期貨的成本（進出各一次手續費 + 稅），無持有利息。"""
    fee = fee_per_contract * contracts * 2
    tax = notional * futures_tax_rate * 2
    total = fee + tax
    return LegCost("個股期貨", fee, tax, 0.0, total)


@dataclass
class WarrantLegCost:
    tool: str
    entry_cost: float       # 進場手續費
    exit_cost: float        # 出場手續費 + 證交稅
    spread_cost: float      # 買賣價差成本（進出各吃半個價差）
    theta_cost: float       # 持有期間時間價值衰減（BS 估算，假設現價與 IV 不變）
    total_cost: float
    decayed_price: float     # 持有 days 天後的理論權證價（現價與 IV 不變下）


def warrant_leg_cost(
    warrant_price: float,
    n_units: int,             # 權證單位數（1 張 = 1000 單位）
    days: int,
    spot: float,
    strike: float,
    implied_vol: float,
    exercise_ratio: float,    # 每 1 單位權證可換股數
    days_to_expiry: int,
    is_call: bool,
    fee_rate: float,
    fee_discount: float,
    warrant_tax_rate: float,
    spread_pct: float,
    r: float = 0.015,
) -> WarrantLegCost:
    """權證單腳成本：手續費 + 證交稅 + 買賣價差 + theta 時間衰減。

    theta 成本 = 現在權證價 - 持有 days 天後的理論價（現價與 IV 不變下純時間衰減），
    用 Black-Scholes 以「每股口徑」估算再乘行使比例還原成每單位權證。
    """
    notional = warrant_price * n_units
    fee_entry = notional * fee_rate * fee_discount
    fee_exit = notional * fee_rate * fee_discount
    tax = notional * warrant_tax_rate
    # 進場買在賣價、出場賣在買價，來回各吃半個價差 ≈ 一個完整價差
    spread_cost = notional * spread_pct

    # theta：以 BS 算現在與 days 天後的每股理論價（IV、spot 固定），差額即時間價值流失
    t_now = max(days_to_expiry, 0) / 365.0
    t_later = max(days_to_expiry - days, 0) / 365.0
    if implied_vol == implied_vol and exercise_ratio > 0:  # IV 非 NaN
        price_now_per_share = bs_price(spot, strike, t_now, r, implied_vol, is_call)
        price_later_per_share = bs_price(spot, strike, t_later, r, implied_vol, is_call)
        decayed_unit_price = price_later_per_share * exercise_ratio
        theta_cost = max((price_now_per_share - price_later_per_share) * exercise_ratio, 0.0) * n_units
    else:
        decayed_unit_price = float("nan")
        theta_cost = float("nan")

    total = fee_entry + fee_exit + tax + spread_cost + (theta_cost if theta_cost == theta_cost else 0.0)
    return WarrantLegCost(
        tool="認購/認售權證",
        entry_cost=fee_entry,
        exit_cost=fee_exit + tax,
        spread_cost=spread_cost,
        theta_cost=theta_cost,
        total_cost=total,
        decayed_price=decayed_unit_price,
    )


@dataclass
class PairCostResult:
    long_cost: LegCost
    short_cost: LegCost
    total_cost: float
    breakeven_move_pct: float   # 價差需朝有利方向變動多少 % 才回本


def pair_breakeven(long_cost: LegCost, short_cost: LegCost, gross_notional: float) -> PairCostResult:
    """以雙腳總成本估算損益兩平：價差需變動多少比例才能覆蓋成本。

    gross_notional = 單腳平均市值（作為獲利基準）。
    """
    total = long_cost.total_cost + short_cost.total_cost
    breakeven = total / gross_notional if gross_notional > 0 else float("inf")
    return PairCostResult(long_cost, short_cost, total, breakeven)
