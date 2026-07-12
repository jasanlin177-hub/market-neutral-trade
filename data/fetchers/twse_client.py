"""TWSE OpenAPI 客戶端：停資停券預告表、股東會公告。"""
import pandas as pd
import requests

from data.fetchers.http_util import get_json

TWSE_OPENAPI = "https://openapi.twse.com.tw/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def roc_to_date(roc: str) -> pd.Timestamp:
    """民國年日期字串（如 '1150703'）轉西元 Timestamp。"""
    roc = roc.strip()
    year = int(roc[:-4]) + 1911
    return pd.Timestamp(year=year, month=int(roc[-4:-2]), day=int(roc[-2:]))


def get_suspension_calendar() -> pd.DataFrame:
    """抓取集中市場停資停券預告表（BFI84U，官方權威來源）。

    回傳欄位：stock_id, name, start_date（停券起日）, end_date（停券迄日）, reason。
    """
    rows = get_json(f"{TWSE_OPENAPI}/exchangeReport/BFI84U", params=None,
                    source="twse-suspension", headers=HEADERS, timeout=30)
    df = pd.DataFrame(rows).rename(
        columns={"Code": "stock_id", "Name": "name", "Reason": "reason"}
    )
    df["start_date"] = df["StartDate"].map(roc_to_date)
    df["end_date"] = df["EndDate"].map(roc_to_date)
    return df[["stock_id", "name", "start_date", "end_date", "reason"]]


def get_warrant_map(as_of: pd.Timestamp = None) -> dict:
    """抓取上市權證基本資料，依標的證券名稱彙整認購/認售可用性。

    回傳 {標的名稱: {'has_call_warrant': bool, 'has_put_warrant': bool,
                    'n_call': int, 'n_put': int}}。
    只計入未到期（最後交易日 >= as_of）的存續權證。
    認購權證 = 看多工具，認售權證 = 看空工具。
    """
    if as_of is None:
        as_of = pd.Timestamp.now().normalize()
    rows = get_json(f"{TWSE_OPENAPI}/opendata/t187ap37_L", params=None,
                    source="twse-warrant-map", headers=HEADERS, timeout=60)

    agg = {}
    for w in rows:
        try:
            last_trade = roc_to_date(w["最後交易日"])
        except (ValueError, KeyError):
            continue
        if last_trade < as_of:
            continue  # 已到期
        underlying = w.get("標的證券/指數", "").strip()
        if not underlying:
            continue
        rec = agg.setdefault(
            underlying,
            {"has_call_warrant": False, "has_put_warrant": False, "n_call": 0, "n_put": 0},
        )
        if w.get("權證類型", "").strip() == "認購":
            rec["has_call_warrant"] = True
            rec["n_call"] += 1
        elif w.get("權證類型", "").strip() == "認售":
            rec["has_put_warrant"] = True
            rec["n_put"] += 1
    return agg


def get_warrant_details(underlying_name: str, as_of: pd.Timestamp = None) -> pd.DataFrame:
    """取得單一標的的逐檔存續權證明細（供選券因子計算）。

    回傳欄位：warrant_id, name, type（認購/認售）, category（類別，如一般型/牛證）,
    strike（最新履約價）, last_trade_date, expiry_date, exercise_ratio（每 1 單位權證可換股數）,
    upper_limit, lower_limit（上下限型才有）。
    """
    if as_of is None:
        as_of = pd.Timestamp.now().normalize()
    rows = get_json(f"{TWSE_OPENAPI}/opendata/t187ap37_L", params=None,
                    source="twse-warrant-details", headers=HEADERS, timeout=60)

    def _to_float(s):
        try:
            return float(str(s).replace(",", ""))
        except (ValueError, TypeError):
            return float("nan")

    records = []
    for w in rows:
        if w.get("標的證券/指數", "").strip() != underlying_name:
            continue
        try:
            last_trade = roc_to_date(w["最後交易日"])
            expiry = roc_to_date(w["履約截止日"])
        except (ValueError, KeyError):
            continue
        if last_trade < as_of:
            continue
        # 履約配發數量是「每仟單位權證」，換算成每 1 單位權證可換股數
        ratio = _to_float(w.get("最新標的履約配發數量(每仟單位權證)")) / 1000.0
        records.append(
            {
                "warrant_id": w.get("權證代號", "").strip(),
                "name": w.get("權證簡稱", "").strip(),
                "type": w.get("權證類型", "").strip(),
                "category": w.get("類別", "").strip(),
                "strike": _to_float(w.get("最新履約價格(元)/履約指數")),
                "last_trade_date": last_trade,
                "expiry_date": expiry,
                "exercise_ratio": ratio,
                "upper_limit": _to_float(w.get("最新上限價格(元)/上限指數")),
                "lower_limit": _to_float(w.get("最新下限價格(元)/下限指數")),
            }
        )
    return pd.DataFrame(records)


def get_shareholder_meetings() -> pd.DataFrame:
    """抓取上市公司股東會公告（t187ap38_L）。

    回傳欄位：stock_id, name, meeting_date（股東會日期）,
    transfer_stop_start（停止過戶起日）。用於比預告表更早看到股東會回補時點。
    """
    rows = get_json(f"{TWSE_OPENAPI}/opendata/t187ap38_L", params=None,
                    source="twse-shareholder-meeting", headers=HEADERS, timeout=30)
    df = pd.DataFrame(rows).rename(
        columns={
            "公司代號": "stock_id",
            "公司名稱": "name",
            "股東常(臨時)會日期-日期": "meeting_date_roc",
            "停止過戶起訖日期-起": "transfer_stop_start_roc",
        }
    )
    df = df[df["meeting_date_roc"].str.strip() != ""]
    df["meeting_date"] = df["meeting_date_roc"].map(roc_to_date)
    df["transfer_stop_start"] = df["transfer_stop_start_roc"].map(roc_to_date)
    return df[["stock_id", "name", "meeting_date", "transfer_stop_start"]]
