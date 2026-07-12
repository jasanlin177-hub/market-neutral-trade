"""FinMind API 股價資料抓取模組。"""
import os
import datetime
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # 從專案根目錄 .env 讀取 FINMIND_TOKEN

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def resolve_date(d: str) -> str:
    """把特殊值 "today" 解析成當天日期字串；其餘原樣回傳。

    讓 config 的 end_date 可寫 "today"，避免寫死日期導致資料停在過去某日。
    """
    if isinstance(d, str) and d.strip().lower() == "today":
        return datetime.date.today().strftime("%Y-%m-%d")
    return d


def get_stock_price(stock_id: str, dataset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """抓取單一股票的歷史資料，回傳含 date/close 欄位的 DataFrame。

    沒有設定 FINMIND_TOKEN 環境變數時，仍會嘗試用公開額度呼叫；
    只有在 API 明確回傳需要授權的錯誤時才會拋出說明性錯誤。
    """
    token = os.environ.get("FINMIND_TOKEN", "")
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": resolve_date(start_date),
        "end_date": resolve_date(end_date),
    }
    if token:
        params["token"] = token

    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    try:
        payload = resp.json()
    except ValueError:
        # FinMind 回傳非 JSON（多半是限流/IP 封鎖時的空內容或 HTML），
        # 印出診斷到伺服器端 log（有無 token、HTTP 狀態、回應前 200 字），方便排查。
        import sys
        snippet = resp.text[:200].replace("\n", " ")
        print(f"[finmind] 非 JSON 回應 stock={stock_id} has_token={bool(token)} "
              f"http={resp.status_code} body[:200]={snippet!r}", file=sys.stderr)
        raise RuntimeError(
            f"FinMind API 回傳非 JSON 內容（HTTP {resp.status_code}）。"
            f"{'（目前無 FINMIND_TOKEN，可能走公開額度被限流）' if not token else ''}"
            f"常見於雲端共用 IP 被資料源限流。"
        )

    status = payload.get("status")
    if status == 402 or "token" in str(payload.get("msg", "")).lower():
        raise RuntimeError(
            f"FinMind API 需要授權才能取得 {stock_id} 的資料，"
            f"請設定環境變數 FINMIND_TOKEN。原始訊息: {payload.get('msg')}"
        )

    data = payload.get("data", [])
    if not data:
        raise ValueError(
            f"股票代號 {stock_id} 在 {start_date}~{end_date} 區間查無資料，"
            f"請確認代號是否正確。API 訊息: {payload.get('msg')}"
        )

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
    # FinMind 偶爾會回傳收盤價 0 的錯誤資料列（非真實停牌/全額交割），過濾避免污染下游對數/迴歸計算
    bad = df["close"] <= 0
    if bad.any():
        print(f"[警告] {stock_id} 有 {bad.sum()} 筆收盤價 <=0 的異常資料已過濾: "
              f"{df.loc[bad, 'date'].dt.date.tolist()}")
        df = df[~bad]
    df = df.rename(columns={"close": stock_id})
    return df


def get_margin_short_data(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """抓取融資融券餘額，回傳含券資比的 DataFrame。

    欄位：date, margin_balance（融資餘額）, short_balance（融券餘額）,
    short_limit（融券限額）, short_margin_ratio（券資比）, short_utilization（融券使用率）。
    """
    token = os.environ.get("FINMIND_TOKEN", "")
    params = {
        "dataset": "TaiwanStockMarginPurchaseShortSale",
        "data_id": stock_id,
        "start_date": resolve_date(start_date),
        "end_date": resolve_date(end_date),
    }
    if token:
        params["token"] = token

    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        raise ValueError(
            f"股票代號 {stock_id} 在 {start_date}~{end_date} 查無融資融券資料。"
            f"API 訊息: {payload.get('msg')}"
        )

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(
        columns={
            "MarginPurchaseTodayBalance": "margin_balance",
            "ShortSaleTodayBalance": "short_balance",
            "ShortSaleLimit": "short_limit",
        }
    )[["date", "margin_balance", "short_balance", "short_limit"]].sort_values("date")
    df["short_margin_ratio"] = df["short_balance"] / df["margin_balance"].replace(0, pd.NA)
    df["short_utilization"] = df["short_balance"] / df["short_limit"].replace(0, pd.NA)
    return df.reset_index(drop=True)


def resolve_stock(query: str) -> dict:
    """依代號或名稱查回 {'id':..., 'name':...}。優先比對代號，找不到再比對名稱。

    多筆同代號（上市/上櫃/興櫃重複列）取第一筆；查無資料拋 ValueError。
    """
    query = query.strip()
    token = os.environ.get("FINMIND_TOKEN", "")
    params = {"dataset": "TaiwanStockInfo"}
    if token:
        params["token"] = token
    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    by_id = [r for r in data if r["stock_id"] == query]
    if by_id:
        return {"id": by_id[0]["stock_id"], "name": by_id[0]["stock_name"]}
    by_name = [r for r in data if r["stock_name"] == query]
    if by_name:
        return {"id": by_name[0]["stock_id"], "name": by_name[0]["stock_name"]}
    raise ValueError(f"查無股票「{query}」，請確認代號或名稱是否正確。")


def get_pair_prices(stock_id_a: str, stock_id_b: str, dataset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """抓取兩檔股票的收盤價並依日期對齊，回傳合併後的 DataFrame。"""
    df_a = get_stock_price(stock_id_a, dataset, start_date, end_date)
    df_b = get_stock_price(stock_id_b, dataset, start_date, end_date)
    merged = pd.merge(df_a, df_b, on="date", how="inner")
    return merged
