"""批次 walk-forward 回測：對股票池兩兩組合跑無前視回測，找真正可行的配對。

跟 scan.py（純統計檢定）的差異：這裡直接跑交易模擬，
用 Sharpe / 勝率 / 最大回撤做最終篩選——因為就算共整合檢定通過，
也可能扣掉成本、抓對進出場時機後就不划算。

篩選邏輯：先過共整合關（scan_cfg 沿用 cointegration_gate 概念），
沒過的配對直接跳過不浪費算力跑回測；過關的才跑 walk-forward。
"""
import sys

import pandas as pd
import yaml
from itertools import combinations

from backtest.engine import walk_forward_backtest
from backtest.performance import compute_stats
from scan import fetch_universe
from signals.cointegration import test_cointegration, to_log


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def scan_walkforward(config: dict, scan_cfg: dict, require_cointegration: bool = True) -> pd.DataFrame:
    data_cfg = config["data"]
    coint_cfg = config["cointegration"]
    bt_cfg = config["backtest"]
    alpha = coint_cfg["significance_level"]
    name_map = {s["id"]: s["name"] for s in scan_cfg["universe"]}

    print(f"[{scan_cfg['name']}] 抓取股票池 {len(scan_cfg['universe'])} 檔...")
    prices = fetch_universe(
        scan_cfg["universe"],
        data_cfg["dataset"],
        scan_cfg.get("start_date", data_cfg["start_date"]),
        data_cfg["end_date"],
    )

    rows = []
    for sid_a, sid_b in combinations(sorted(prices.keys()), 2):
        merged = pd.merge(prices[sid_a], prices[sid_b], on="date", how="inner")
        min_needed = bt_cfg["estimation_window"] + bt_cfg["lookback"]
        if len(merged) < min_needed:
            print(f"  {sid_a}x{sid_b}: 對齊後僅 {len(merged)} 筆，不足 {min_needed}，略過。")
            continue

        log_a = to_log(merged[sid_a]) if coint_cfg.get("use_log_prices", False) else merged[sid_a]
        log_b = to_log(merged[sid_b]) if coint_cfg.get("use_log_prices", False) else merged[sid_b]

        full = test_cointegration(log_a, log_b, alpha)
        if require_cointegration and not full.is_cointegrated:
            continue  # 沒過共整合關，不浪費算力跑回測

        result = walk_forward_backtest(
            log_a, log_b,
            estimation_window=bt_cfg["estimation_window"],
            reestimate_every=bt_cfg["reestimate_every"],
            lookback=bt_cfg["lookback"],
            entry_z=bt_cfg["entry_z"],
            exit_z=bt_cfg["exit_z"],
            stop_z=bt_cfg["stop_z"],
            cost_per_turn=bt_cfg["cost_per_turn"],
        )
        stats = compute_stats(result.daily_returns, result.trade_returns)

        rows.append(
            {
                "pair": f"{sid_a} {name_map.get(sid_a, '')} x {sid_b} {name_map.get(sid_b, '')}",
                "n_obs": len(merged),
                "full_pvalue": full.p_value,
                "n_trades": stats.n_trades,
                "win_rate": stats.win_rate,
                "total_return": stats.total_return,
                "sharpe": stats.sharpe,
                "max_drawdown": stats.max_drawdown,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("sharpe", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    config = load_config()
    pd.set_option("display.width", 200)

    for scan_cfg in config["scans"]:
        df = scan_walkforward(config, scan_cfg)
        if df.empty:
            print(f"[{scan_cfg['name']}] 沒有配對通過共整合關，無回測結果。")
            continue

        print(f"\n=== [{scan_cfg['name']}] Walk-forward 回測結果（依 Sharpe 排序，"
              f"僅含通過共整合的配對）===")
        print(
            df.to_string(
                index=False,
                formatters={
                    "full_pvalue": "{:.4f}".format,
                    "win_rate": "{:.0%}".format,
                    "total_return": "{:.2%}".format,
                    "sharpe": "{:.2f}".format,
                    "max_drawdown": "{:.2%}".format,
                },
            )
        )
        out = f"walkforward_{scan_cfg['name']}.csv"
        df.to_csv(out, index=False)
        print(f"結果已存至 {out}")
