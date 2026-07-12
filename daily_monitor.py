"""每日監控任務：跑風控守門 -> 組報告 -> Telegram 推播。

可直接手動執行（python daily_monitor.py），也由 scheduler.py 排程呼叫。
"""
from datetime import datetime

import yaml

from alerts.telegram_notifier import send_message
from check_risk import load_config, run as run_risk_checks


def build_report(results: list, alerts_only: bool = False) -> str:
    """把守門結果組成通知文字。alerts_only=True 時只回報有 BLOCK/WARN 的配對。"""
    lines = [f"配對交易每日監控 {datetime.now():%Y-%m-%d %H:%M}"]
    reported = 0
    for name, decision in results:
        issues = decision.blocks + decision.warnings
        if alerts_only and not issues:
            continue
        reported += 1
        status = "放行" if decision.approved else "擋下"
        lines.append(f"\n[{name}] => {status}")
        for c in issues:
            tag = "BLOCK" if c in decision.blocks else "WARN"
            lines.append(f"  ({tag}) {c.name}: {c.detail}")
    if reported == 0:
        lines.append("\n所有配對檢查全數通過，無警示。")
    return "\n".join(lines)


def main():
    config = load_config()
    notify_cfg = config.get("notification", {})
    try:
        results = run_risk_checks()
    except Exception as e:
        # 資料抓取失敗也要通知，排程環境沒人看主控台
        send_message(f"每日監控執行失敗: {e}")
        raise

    report = build_report(results, alerts_only=notify_cfg.get("alerts_only", False))
    send_message(report)


if __name__ == "__main__":
    main()
