"""TAIFEX 個股期貨/選擇權標的清單抓取。

來源：https://www.taifex.com.tw/cht/2/stockLists （HTML 表格）
更新頻率低（每季檢查即可），可快取。
"""
import re

import pandas as pd
import requests

TAIFEX_URL = "https://www.taifex.com.tw/cht/2/stockLists"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _clean(cell: str) -> str:
    return re.sub(r"<[^>]+>", "", cell).replace("\r", "").replace("\n", "").replace("\t", "").strip()


def get_futures_options_list() -> pd.DataFrame:
    """回傳個股期貨/選擇權標的清單。

    欄位：stock_id, name, product_code（期貨商品代碼前綴，如 IM）,
    has_futures（是否為股票期貨標的）, has_options（是否為股票選擇權標的）。
    """
    resp = requests.get(TAIFEX_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    records = []
    for row in rows:
        cells = [_clean(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        # 列結構：商品代碼 | 公司全名 | 證券代號 | 簡稱 | 期貨標記 | 選擇權標記 | ...
        if len(cells) < 6:
            continue
        stock_id = cells[2]
        # 只收 4~6 碼數字證券代號，濾掉表頭與雜列
        if not re.fullmatch(r"\d{4,6}", stock_id):
            continue
        records.append(
            {
                "stock_id": stock_id,
                "name": cells[3],
                "product_code": cells[0] if cells[0] not in ("-", "") else None,
                "has_futures": "●" in cells[4],
                "has_options": "●" in cells[5],
            }
        )
    return pd.DataFrame(records)


def get_tradable_map() -> dict:
    """回傳 {stock_id: {'has_futures': bool, 'has_options': bool, ...}} 供工具選擇引擎查詢。"""
    df = get_futures_options_list()
    return {r["stock_id"]: r for r in df.to_dict("records")}
