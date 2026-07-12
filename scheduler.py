"""排程器：每個交易日固定時間執行 daily_monitor。

用法：python scheduler.py（前景常駐，Ctrl+C 停止）
執行時間在 config/settings.yaml 的 notification.daily_run_at 設定。
注意：融資融券餘額約在盤後晚間公布，執行時間不要設太早。
"""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from check_risk import load_config
from daily_monitor import main as run_daily_monitor


def start():
    config = load_config()
    notify_cfg = config.get("notification", {})
    run_at = notify_cfg.get("daily_run_at", "20:00")
    hour, minute = (int(x) for x in run_at.split(":"))

    scheduler = BlockingScheduler(timezone="Asia/Taipei")
    scheduler.add_job(
        run_daily_monitor,
        CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute),
        id="daily_monitor",
        misfire_grace_time=3600,  # 錯過排程 1 小時內補跑
    )
    print(f"排程已啟動：週一至週五 {run_at}（Asia/Taipei）執行每日監控，Ctrl+C 停止。")
    scheduler.start()


if __name__ == "__main__":
    start()
