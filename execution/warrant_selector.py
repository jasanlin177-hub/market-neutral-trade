"""權證選券引擎：對某標的的存續權證逐檔算因子、過濾、排序，挑出最適交易的權證。

篩選 / 排序邏輯（可在 config 調整）：
- 硬性過濾：一般型、日均量足夠、距到期天數足夠、非過度價外。
- 排序偏好：時間價值佔比低（不想付太多時間價值）、隱含波動率低（不想買貴）、
  流動性高（好進出）、價內外適中（避免深價外的高槓桿高風險）。
綜合分數越低越好（成本/風險越小）。
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd

from data.fetchers.finmind_client import get_stock_price
from data.fetchers.twse_client import get_warrant_details
from execution.warrant_pricing import compute_factors


@dataclass
class WarrantScreenConfig:
    """台股掛牌權證絕大多數為價外，故預設區間允許價外，並以 IV/流動性選券。"""
    direction: str = "long"          # long -> 認購, short -> 認售
    min_avg_volume: float = 100      # 日均量下限（張，流動性）
    min_days_to_expiry: int = 30     # 距到期最少天數（避免時間價值急速衰減）
    # 現價/履約價 可接受區間：認購價外時 <1，太低（深價外）槓桿過高風險大，故設下限 0.75；
    # 上限 1.20 容許適度價內。認售方向可另調。
    moneyness_band: tuple = (0.75, 1.20)
    max_time_value_pct: float = 1.05  # 時間價值佔比上限（價外必為~100%，預設幾乎不擋，保留給價內情境）
    max_implied_vol: float = 1.20     # 隱含波動率上限（過濾定價過貴的權證）
    lookback_days: int = 20          # 計算日均量的回看天數


class FinMindRateLimitError(RuntimeError):
    """FinMind API 額度用盡（HTTP 402），逐檔查詢應整批停止，不要繼續一檔檔重試。"""


def _fetch_warrant_price_volume(warrant_id: str, start: str, end: str, lookback: int):
    """抓權證的最新收盤價與近期日均量。回傳 (last_close, avg_volume)。

    單一權證查無資料時回傳 (NaN, 0)，由呼叫端跳過該檔；
    額度用盡（402）視為不可恢復錯誤，往上拋出讓整批選券中止。
    """
    import os
    import requests
    token = os.environ.get("FINMIND_TOKEN", "")
    params = {"dataset": "TaiwanStockPrice", "data_id": warrant_id,
              "start_date": start, "end_date": end}
    if token:
        params["token"] = token
    resp = requests.get("https://api.finmindtrade.com/api/v4/data", params=params, timeout=30)
    if resp.status_code == 402:
        raise FinMindRateLimitError(
            f"FinMind API 額度已用盡（HTTP 402），查 {warrant_id} 時觸發。"
            f"請稍後再試或減少同時篩選的權證數量。"
        )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        return float("nan"), 0.0
    df = pd.DataFrame(data).sort_values("date")
    df = df[df["close"] > 0]
    if df.empty:
        return float("nan"), 0.0
    last_close = float(df["close"].iloc[-1])
    # Trading_Volume 單位為股，換算成張（/1000）
    avg_vol = float(df["Trading_Volume"].tail(lookback).mean()) / 1000.0
    return last_close, avg_vol


def screen_warrants(
    underlying_name: str,
    underlying_id: str,
    as_of: pd.Timestamp,
    screen: WarrantScreenConfig,
    r: float = 0.015,
    max_candidates: int = 40,
    max_workers: int = 8,
    progress_callback=None,
) -> tuple:
    """對標的的存續權證做選券，回傳 (依綜合分數排序的 DataFrame, 是否額度用盡提前中止)。

    先用不需查價的欄位（一般型、到期天數、moneyness）過濾，
    再對候選檔逐一查價——避免對注定被篩掉的牛熊證/深價外檔浪費 API 額度。
    max_candidates 進一步限制實際查價的檔數上限（依 moneyness 接近 1 排序取前 N）。

    查價階段用 ThreadPoolExecutor 平行處理（I/O bound，FinMind 一檔一檔序列查
    實測近 40 檔要 2 分鐘以上，平行後大幅縮短）。
    progress_callback(done, total) 若提供，每查完一檔會呼叫一次，供 UI 顯示進度。
    """
    wtype = "認購" if screen.direction == "long" else "認售"
    details = get_warrant_details(underlying_name, as_of=as_of)
    details = details[details["type"] == wtype]
    details = details[details["category"] == "一般型"]  # 牛熊證等非 BS 適用，查價前先排除
    days_left = (details["expiry_date"] - as_of).dt.days
    details = details[days_left >= screen.min_days_to_expiry]
    if details.empty:
        return pd.DataFrame(), False

    # 標的現價
    spot_df = get_stock_price(underlying_id, "TaiwanStockPrice",
                              (as_of - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                              as_of.strftime("%Y-%m-%d"))
    spot = float(spot_df[underlying_id].iloc[-1])

    # 查價前先用履約價做 moneyness 過濾（不需要權證市價）
    moneyness = spot / details["strike"].replace(0, float("nan"))
    details = details[(moneyness >= screen.moneyness_band[0]) & (moneyness <= screen.moneyness_band[1])]
    if details.empty:
        return pd.DataFrame(), False

    # 候選過多時，優先查價 moneyness 最接近 1 的（越可能通過、越具代表性）
    moneyness = spot / details["strike"]
    details = details.assign(_m=moneyness).sort_values(
        by="_m", key=lambda s: (s - 1).abs()
    ).head(max_candidates).drop(columns="_m")

    start = (as_of - pd.Timedelta(days=screen.lookback_days * 2 + 10)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")

    warrant_rows = list(details.iterrows())
    total = len(warrant_rows)
    price_map = {}     # warrant_id -> (price, avg_vol)
    rate_limited = False
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_warrant_price_volume, w["warrant_id"], start, end,
                       screen.lookback_days): w["warrant_id"]
            for _, w in warrant_rows
        }
        for fut in as_completed(futures):
            wid = futures[fut]
            try:
                price_map[wid] = fut.result()
            except FinMindRateLimitError:
                rate_limited = True
                # 額度用盡：不再等其餘 future（它們多半也會 402），但已送出的請求仍讓其完成
            except Exception:
                price_map[wid] = (float("nan"), 0.0)
            done += 1
            if progress_callback:
                progress_callback(done, total)

    rows = []
    for _, w in warrant_rows:
        price, avg_vol = price_map.get(w["warrant_id"], (float("nan"), 0.0))
        if pd.isna(price):
            continue
        f = compute_factors(w.to_dict(), spot, price, avg_vol, as_of, r=r)

        # 硬性過濾（is_plain 已在查價前濾過，這裡保留 as safety net）
        if not f.is_plain:
            continue
        if f.avg_volume < screen.min_avg_volume:
            continue
        if f.days_to_expiry < screen.min_days_to_expiry:
            continue
        if not (screen.moneyness_band[0] <= f.moneyness <= screen.moneyness_band[1]):
            continue
        if pd.notna(f.time_value_pct) and f.time_value_pct > screen.max_time_value_pct:
            continue
        if pd.notna(f.implied_vol) and f.implied_vol > screen.max_implied_vol:
            continue

        rows.append({
            "warrant_id": f.warrant_id, "name": f.name, "type": f.type,
            "price": f.warrant_price, "strike": w["strike"],
            "exercise_ratio": w["exercise_ratio"],
            "moneyness": f.moneyness, "itm": f.itm,
            "time_value_pct": f.time_value_pct, "days_to_expiry": f.days_to_expiry,
            "implied_vol": f.implied_vol, "avg_volume": f.avg_volume,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, rate_limited

    # 綜合分數（越低越好）：隱含波動率為主（買貴的懲罰）- 流動性分數 + 深價外懲罰。
    # 價外權證時間價值佔比恆為~100%無鑑別度，故不納入評分，改以 IV 衡量貴／便宜。
    iv_fill = df["implied_vol"].fillna(df["implied_vol"].median() if df["implied_vol"].notna().any() else 0.8)
    liq_score = (df["avg_volume"] / df["avg_volume"].max()).fillna(0)
    otm_penalty = (1.0 - df["moneyness"]).clip(lower=0)  # 認購越深價外（moneyness 越低）懲罰越大
    df["score"] = iv_fill - 0.3 * liq_score + 0.5 * otm_penalty
    return df.sort_values("score").reset_index(drop=True), rate_limited
