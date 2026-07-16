"""Telegram 互動指令 bot：手機用 Telegram 直接下指令查詢，不用開 dashboard。

用法：python telegram_bot.py（前景常駐，Ctrl+C 停止）
只回應 .env 裡 TELEGRAM_CHAT_ID 設定的那個 chat，避免陌生人亂用。

指令：
  /start, /help             顯示可用指令
  /check 多方 空方           跑六道風控守門（代號或名稱皆可，如 /check 3019 4585）
  /check                    跑 config.yaml 預設 pairs 清單的風控守門
  /signal 多方 空方          查共整合 + z-score + 進場建議
  /scan 股票池名稱           跑該股票池的共整合掃描，回傳前 5 名
"""
import os
import sys
import time

import requests
from dotenv import load_dotenv

from alerts.telegram_notifier import send_message
from check_risk import check_pair, load_config
from data.fetchers.finmind_client import get_pair_prices, resolve_stock
from scan import scan_pairs
from signals.cointegration import test_cointegration, to_log
from signals.zscore import rolling_zscore, spread_series

load_dotenv()
TELEGRAM_API = "https://api.telegram.org"

HELP_TEXT = (
    "可用指令：\n"
    "/check 多方 空方 — 對指定配對跑六道風控守門（代號或名稱皆可，如 /check 3019 4585）\n"
    "/check — 跑 config.yaml 預設配對清單\n"
    "/signal 多方 空方 — 查共整合 + z-score + 進場建議\n"
    "/scan 股票池名稱 — 跑批次共整合掃描（如 /scan OPTICAL）\n"
    "/help — 顯示本說明"
)


def _token() -> str:
    t = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not t:
        print("錯誤：未設定 TELEGRAM_BOT_TOKEN", file=sys.stderr)
        sys.exit(1)
    return t


def send(chat_id, text: str):
    # 重用 telegram_notifier 的 send_message：這台機器 requests 對 Telegram API
    # 偶發 TLS 連線被重置，該函式已內建 curl 備援，這裡不重寫一份沒有備援的版本。
    # 注意：send_message 是對 .env 裡的 TELEGRAM_CHAT_ID 發送，此處 bot 已限定
    # 只回應同一個 chat_id（見 run() 的過濾），故直接沿用不需再傳 chat_id。
    if not send_message(text):
        print(f"[telegram_bot] 發送失敗（chat_id={chat_id}）", file=sys.stderr)


def fmt_gate(decision) -> str:
    lines = [f"結論：{'✅ 放行' if decision.approved else '⛔ 擋下'}"]
    for c in decision.checks:
        tag = "PASS" if c.passed else ("BLOCK" if c.level.value == "block" else "WARN")
        lines.append(f"[{tag}] {c.name}：{c.detail}")
    return "\n".join(lines)


def handle_check(chat_id, args, config):
    if len(args) >= 2:
        try:
            long_leg = resolve_stock(args[0])
            short_leg = resolve_stock(args[1])
        except ValueError as e:
            send(chat_id, str(e))
            return
        try:
            decision = check_pair(long_leg, short_leg, config)
        except Exception as e:
            send(chat_id, f"查詢失敗：{type(e).__name__}，請稍後再試")
            return
        send(chat_id, f"{long_leg['name']} x {short_leg['name']}\n\n{fmt_gate(decision)}")
    else:
        pairs = config.get("pairs", [])
        if not pairs:
            send(chat_id, "config.yaml 沒有預設配對，請帶入兩個代號：/check 3019 4585")
            return
        for p in pairs:
            try:
                decision = check_pair(p["stock_a"], p["stock_b"], config)
                send(chat_id, f"[{p['name']}]\n{fmt_gate(decision)}")
            except Exception as e:
                send(chat_id, f"[{p['name']}] 查詢失敗：{type(e).__name__}")


def handle_signal(chat_id, args, config):
    if len(args) < 2:
        send(chat_id, "用法：/signal 多方代號或名稱 空方代號或名稱")
        return
    try:
        long_leg = resolve_stock(args[0])
        short_leg = resolve_stock(args[1])
    except ValueError as e:
        send(chat_id, str(e))
        return

    data_cfg, coint_cfg = config["data"], config["cointegration"]
    try:
        prices = get_pair_prices(long_leg["id"], short_leg["id"], data_cfg["dataset"],
                                 data_cfg["start_date"], data_cfg["end_date"])
        log_a, log_b = to_log(prices[long_leg["id"]]), to_log(prices[short_leg["id"]])
        coint = test_cointegration(log_a, log_b, coint_cfg["significance_level"])
        z = rolling_zscore(spread_series(log_a, log_b, coint.hedge_ratio),
                           coint_cfg["rolling"]["window"])
        latest_z = z.iloc[-1]
        entry_z = config["backtest"]["entry_z"]
    except Exception as e:
        send(chat_id, f"查詢失敗：{type(e).__name__}，請稍後再試")
        return

    lines = [
        f"{long_leg['name']} x {short_leg['name']}",
        f"共整合 p-value={coint.p_value:.4f}（{'通過' if coint.is_cointegrated else '未通過'}）",
        f"hedge ratio={coint.hedge_ratio:.4f}",
        f"z-score={latest_z:.2f}" if latest_z == latest_z else "z-score=N/A",
    ]
    if latest_z == latest_z:
        if latest_z <= -entry_z:
            lines.append(f"👉 建議：做多 {long_leg['name']}、放空 {short_leg['name']}")
        elif latest_z >= entry_z:
            lines.append(f"👉 建議：放空 {long_leg['name']}、做多 {short_leg['name']}（與輸入順序相反）")
        else:
            lines.append("👉 未達進場門檻，暫不建議進場")
    send(chat_id, "\n".join(lines))


def handle_scan(chat_id, args, config):
    if not args:
        names = [s["name"] for s in config["scans"]]
        send(chat_id, "用法：/scan 股票池名稱\n可用股票池：" + ", ".join(names))
        return
    name = args[0]
    scan_cfg = next((s for s in config["scans"] if s["name"] == name), None)
    if not scan_cfg:
        send(chat_id, f"查無股票池「{name}」")
        return
    send(chat_id, f"開始掃描 {name}，請稍候…")
    try:
        df = scan_pairs(config, scan_cfg)
    except Exception as e:
        send(chat_id, f"掃描失敗：{type(e).__name__}，請稍後再試")
        return
    if df.empty:
        send(chat_id, "沒有任何配對完成檢定")
        return
    lines = [f"{name} 共整合掃描前 5 名："]
    for _, r in df.head(5).iterrows():
        lines.append(f"{r['pair']}：p={r['full_pvalue']:.4f}（{'過' if r['full_pass'] else '未過'}）")
    send(chat_id, "\n".join(lines))


def dispatch(chat_id, text, config):
    parts = text.strip().split()
    if not parts:
        return
    cmd, args = parts[0].lower(), parts[1:]
    if cmd in ("/start", "/help"):
        send(chat_id, HELP_TEXT)
    elif cmd == "/check":
        handle_check(chat_id, args, config)
    elif cmd == "/signal":
        handle_signal(chat_id, args, config)
    elif cmd == "/scan":
        handle_scan(chat_id, args, config)
    else:
        send(chat_id, f"不認得指令「{cmd}」，輸入 /help 看可用指令")


def run():
    token = _token()
    config = load_config()
    allowed_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not allowed_chat_id:
        print("錯誤：未設定 TELEGRAM_CHAT_ID，為避免陌生人操控你的 bot，拒絕啟動。",
              file=sys.stderr)
        sys.exit(1)

    offset = None
    print("Telegram bot 已啟動，等待指令…（Ctrl+C 停止）")
    while True:
        try:
            resp = requests.get(
                f"{TELEGRAM_API}/bot{token}/getUpdates",
                params={"timeout": 30, "offset": offset}, timeout=40,
            )
            resp.raise_for_status()
            updates = resp.json().get("result", [])
        except requests.exceptions.RequestException as e:
            print(f"[telegram_bot] 輪詢失敗: {type(e).__name__}，5 秒後重試", file=sys.stderr)
            time.sleep(5)
            continue

        for u in updates:
            offset = u["update_id"] + 1
            msg = u.get("message", {})
            text = msg.get("text", "")
            chat_id = msg.get("chat", {}).get("id")
            if not text or chat_id is None:
                continue
            if str(chat_id) != str(allowed_chat_id):
                continue  # 不是設定的 chat_id，忽略（避免陌生人操控）
            print(f"[telegram_bot] 收到指令: {text}")
            try:
                dispatch(chat_id, text, config)
            except Exception as e:
                print(f"[telegram_bot] 處理指令失敗: {type(e).__name__}: {e}", file=sys.stderr)
                send(chat_id, "處理指令時發生錯誤，請稍後再試")


if __name__ == "__main__":
    run()
