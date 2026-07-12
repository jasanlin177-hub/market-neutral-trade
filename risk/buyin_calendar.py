"""強制回補日曆：TWSE 官方停券預告優先，除權息公告推算為備援。

資料來源分兩層：
1. TWSE 停資停券預告表（BFI84U）——官方權威來源，含停券起訖日與原因。
   融券部位必須在停券起日前回補完畢，本模組以停券起日作為回補期限。
2. FinMind 除權息公告推算——預告表通常只涵蓋近期，更遠的事件用
   「除權息交易日往前推 N 個營業日」估算（僅排除週末、未排台股假日），
   結果標示為估算值。
"""
import os
from dataclasses import dataclass, field

import pandas as pd
import requests

from data.fetchers.finmind_client import FINMIND_URL
from data.fetchers.twse_client import get_suspension_calendar


@dataclass
class BuyinAlert:
    has_upcoming_event: bool
    triggered: bool                 # 今天已進入回補風險區（回補期限前 warning_days 天內）
    events: list = field(default_factory=list)  # [{kind, ex_date, est_last_buyin, days_left, source}]


def get_dividend_events(stock_id: str, start_date: str, end_date: str) -> list:
    """抓取已公告的除權息交易日（現金股息與股票股利分列）。"""
    token = os.environ.get("FINMIND_TOKEN", "")
    params = {
        "dataset": "TaiwanStockDividend",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    if token:
        params["token"] = token
    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    events = []
    for row in data:
        for key, kind in [
            ("CashExDividendTradingDate", "除息"),
            ("StockExDividendTradingDate", "除權"),
        ]:
            d = row.get(key, "")
            if d:
                events.append({"ex_date": pd.Timestamp(d), "kind": kind})
    # 去重（同一除權息日可能出現在多筆公告）
    seen = set()
    unique = []
    for e in sorted(events, key=lambda x: x["ex_date"]):
        k = (e["ex_date"], e["kind"])
        if k not in seen:
            seen.add(k)
            unique.append(e)
    return unique


def get_official_suspensions(stock_id: str, today: pd.Timestamp) -> list:
    """從 TWSE 停券預告表撈出該股票尚未結束的停券事件（官方權威來源）。"""
    df = get_suspension_calendar()
    df = df[(df["stock_id"] == stock_id) & (df["end_date"] >= today)]
    return [
        {
            "kind": f"停券（{row['reason']}）",
            "ex_date": row["end_date"].date(),        # 停券迄日
            "est_last_buyin": row["start_date"].date(),  # 停券起日＝回補期限
            "days_left": (row["start_date"] - today).days,
            "source": "TWSE官方預告",
        }
        for _, row in df.iterrows()
    ]


def check_buyin_risk(
    stock_id: str,
    today: pd.Timestamp,
    lookahead_days: int = 90,
    buyin_offset_bdays: int = 6,
    warning_days: int = 10,
) -> BuyinAlert:
    """檢查放空腳未來 lookahead_days 內是否有強制回補風險。

    官方停券預告優先；沒被預告表涵蓋的除權息事件，
    用「除權息交易日往前推 buyin_offset_bdays 個營業日」估算補上。
    今天距離回補期限不足 warning_days 天即觸發。
    """
    upcoming = []

    # 第一層：TWSE 官方停券預告
    official = get_official_suspensions(stock_id, today)
    upcoming.extend(official)
    official_windows = [(e["est_last_buyin"], e["ex_date"]) for e in official]

    # 第二層：FinMind 除權息公告推算（補預告表沒涵蓋的較遠事件）
    end = (today + pd.Timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
    ann_start = (today - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    for e in get_dividend_events(stock_id, ann_start, end):
        if e["ex_date"] < today:
            continue
        # 已被官方停券窗口涵蓋的除權息事件不重複列出
        # （停券迄日通常是除權息日前一日，窗口尾端放寬 3 天避免漏去重）
        if any(
            start <= e["ex_date"].date() <= end_ + pd.Timedelta(days=3)
            for start, end_ in official_windows
        ):
            continue
        est_last_buyin = e["ex_date"] - pd.offsets.BDay(buyin_offset_bdays)
        upcoming.append(
            {
                "kind": e["kind"],
                "ex_date": e["ex_date"].date(),
                "est_last_buyin": est_last_buyin.date(),
                "days_left": (est_last_buyin - today).days,
                "source": "除權息推算（估算值）",
            }
        )

    triggered = any(e["days_left"] <= warning_days for e in upcoming)
    return BuyinAlert(
        has_upcoming_event=bool(upcoming),
        triggered=triggered,
        events=sorted(upcoming, key=lambda e: e["days_left"]),
    )
