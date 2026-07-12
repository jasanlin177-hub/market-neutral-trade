"""共用的 HTTP JSON 抓取工具，帶診斷用途的錯誤處理。

各資料源（FinMind / TWSE OpenAPI）呼叫 API 拿 JSON 時共用這個函式，
遇到非 JSON 回應（限流、IP 封鎖常見）時，會把「哪個來源、host、HTTP 狀態、
回應前 200 字」印到伺服器端 log（token 已在 URL 中，故只印 host 不印完整 URL），
方便在雲端部署時定位到底是哪個資料源、哪個 host 出問題。
"""
import sys
from urllib.parse import urlparse

import requests


def get_json(url: str, params: dict, source: str, timeout: int = 30, headers: dict = None):
    """GET 一個回傳 JSON 的 API。非 JSON 時拋 RuntimeError 並記錄診斷。

    source：來源名稱（如 "finmind-price"），只用於 log 辨識。
    """
    resp = requests.get(url, params=params, timeout=timeout, headers=headers)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        host = urlparse(url).netloc
        snippet = resp.text[:200].replace("\n", " ")
        print(f"[http_util] 非 JSON 回應 source={source} host={host} "
              f"http={resp.status_code} body[:200]={snippet!r}", file=sys.stderr)
        raise RuntimeError(
            f"資料源 {source}（{host}）回傳非 JSON（HTTP {resp.status_code}）；"
            f"常見於雲端共用 IP 被資料源限流／封鎖。"
        )
