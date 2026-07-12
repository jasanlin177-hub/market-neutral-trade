"""Telegram 通知模組。

需要在 .env 設定：
  TELEGRAM_BOT_TOKEN=（BotFather 建立 bot 後取得）
  TELEGRAM_CHAT_ID=（跟 bot 對話後從 getUpdates 取得）
未設定時自動退回主控台輸出，不會中斷流程。
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API = "https://api.telegram.org"
MAX_RETRIES = 4       # 對 Telegram 的連線會間歇性被重置（實測 5 次約中 2 次），重試可解
RETRY_WAIT_SEC = 3


def is_configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN")) and bool(os.environ.get("TELEGRAM_CHAT_ID"))


def send_message(text: str) -> bool:
    """發送 Telegram 訊息；未設定憑證時印到主控台並回傳 False。

    連線層錯誤（TLS 被重置等）會自動重試 MAX_RETRIES 次。
    """
    if not is_configured():
        print("[Telegram 未設定，訊息改印主控台]")
        print(text)
        return False

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"{TELEGRAM_API}/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.ok:
                return True
            print(f"[Telegram 發送失敗] HTTP {resp.status_code}: {resp.text[:200]}")
            return False  # HTTP 層錯誤（token/chat_id 有問題）重試也沒用
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SEC)

    print(f"[Telegram 發送失敗] 重試 {MAX_RETRIES} 次仍連線被重置: {last_error}")
    return False
