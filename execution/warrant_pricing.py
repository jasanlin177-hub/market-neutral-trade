"""權證定價與選券因子：Black-Scholes 定價 + 隱含波動率反解。

隱含波動率用二分法反解（不依賴 scipy），對「一般型」認購/認售權證適用；
牛熊證等上下限型（category 非一般型）不套用標準 BS，IV 回傳 NaN。
"""
import math
from dataclasses import dataclass


def _norm_cdf(x: float) -> float:
    """標準常態累積分佈（用 erf，避免額外相依）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, t: float, r: float, sigma: float, is_call: bool, q: float = 0.0) -> float:
    """Black-Scholes 單股選擇權理論價（每 1 股標的）。t 為年化到期時間。"""
    if t <= 0 or sigma <= 0:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return intrinsic
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if is_call:
        return spot * math.exp(-q * t) * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * math.exp(-q * t) * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, t: float, r: float, sigma: float, is_call: bool, q: float = 0.0) -> float:
    """Black-Scholes delta（每 1 股）。認購 0~1，認售 -1~0。"""
    if t <= 0 or sigma <= 0:
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    if is_call:
        return math.exp(-q * t) * _norm_cdf(d1)
    return -math.exp(-q * t) * _norm_cdf(-d1)


def implied_vol(option_price: float, spot: float, strike: float, t: float, r: float,
                is_call: bool, q: float = 0.0, tol: float = 1e-5, max_iter: int = 100) -> float:
    """二分法反解隱含波動率（每 1 股口徑的權證價）。反解不出時回傳 NaN。"""
    intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    if option_price < intrinsic - tol or option_price <= 0 or t <= 0:
        return float("nan")
    lo, hi = 1e-4, 5.0
    # 確保區間包住解
    if bs_price(spot, strike, t, r, hi, is_call, q) < option_price:
        return float("nan")  # 波動率超過 500%，視為無效
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = bs_price(spot, strike, t, r, mid, is_call, q)
        if abs(price - option_price) < tol:
            return mid
        if price > option_price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@dataclass
class WarrantFactors:
    warrant_id: str
    name: str
    type: str
    is_plain: bool          # 是否一般型（IV 才有效）
    warrant_price: float
    moneyness: float        # 現價 / 履約價（認購 >1 為價內；認售 <1 為價內）
    itm: bool               # 是否價內
    intrinsic_value: float  # 每 1 單位權證的內含價值
    time_value: float       # 每 1 單位權證的時間價值 = 市價 - 內含價值
    time_value_pct: float   # 時間價值佔權證價比例（越高＝時間價值消耗負擔越重）
    days_to_expiry: int
    implied_vol: float
    avg_volume: float       # 近期日均成交量（流動性）


def compute_factors(
    warrant_row: dict,
    spot: float,
    warrant_price: float,
    avg_volume: float,
    as_of,
    r: float = 0.015,
    q: float = 0.0,
) -> WarrantFactors:
    """對單一權證計算選券四因子。warrant_row 來自 get_warrant_details 的一列。"""
    is_call = warrant_row["type"] == "認購"
    strike = warrant_row["strike"]
    ratio = warrant_row["exercise_ratio"]
    is_plain = warrant_row["category"] == "一般型"

    days = (warrant_row["expiry_date"] - as_of).days
    t = max(days, 0) / 365.0

    # 內含價值（每 1 單位權證）= 每股內含 * 行使比例
    intrinsic_per_share = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    intrinsic = intrinsic_per_share * ratio
    time_value = warrant_price - intrinsic
    tv_pct = time_value / warrant_price if warrant_price > 0 else float("nan")
    moneyness = spot / strike if strike > 0 else float("nan")
    itm = (spot > strike) if is_call else (spot < strike)

    # IV：把權證價還原成「每 1 股口徑」再反解（僅一般型）
    iv = float("nan")
    if is_plain and ratio > 0:
        option_price_per_share = warrant_price / ratio
        iv = implied_vol(option_price_per_share, spot, strike, t, r, is_call, q)

    return WarrantFactors(
        warrant_id=warrant_row["warrant_id"],
        name=warrant_row["name"],
        type=warrant_row["type"],
        is_plain=is_plain,
        warrant_price=warrant_price,
        moneyness=moneyness,
        itm=itm,
        intrinsic_value=intrinsic,
        time_value=time_value,
        time_value_pct=tv_pct,
        days_to_expiry=days,
        implied_vol=iv,
        avg_volume=avg_volume,
    )
