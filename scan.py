"""批次掃描同產業股票池，兩兩組合跑共整合檢定，找穩定配對。

排序邏輯：
1. 全區間（對數價格）p-value 是否通過
2. 滾動窗口通過率（越高越好）
3. hedge ratio 變異係數 CV（越低代表比例越穩定）
"""
import sys
from itertools import combinations

import pandas as pd
import yaml

from data.fetchers.finmind_client import get_stock_price
from signals.cointegration import rolling_cointegration, test_cointegration, to_log


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_universe(universe: list, dataset: str, start_date: str, end_date: str) -> dict:
    """抓取股票池所有標的的收盤價，回傳 {stock_id: DataFrame}。抓不到的印警告後略過。"""
    prices = {}
    for stock in universe:
        sid = stock["id"]
        try:
            prices[sid] = get_stock_price(sid, dataset, start_date, end_date)
            print(f"  {sid} {stock['name']}: {len(prices[sid])} 筆")
        except (ValueError, RuntimeError) as e:
            print(f"  {sid} {stock['name']}: 抓取失敗，略過。{e}")
    return prices


def scan_pairs(config: dict, scan_cfg: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    coint_cfg = config["cointegration"]
    rolling_cfg = coint_cfg.get("rolling", {})
    window = rolling_cfg.get("window", 120)
    step = rolling_cfg.get("step", 5)
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
        if len(merged) < window:
            print(f"  {sid_a}x{sid_b}: 對齊後僅 {len(merged)} 筆，不足 window={window}，略過。")
            continue

        pa, pb = merged[sid_a], merged[sid_b]
        if coint_cfg.get("use_log_prices", False):
            pa, pb = to_log(pa), to_log(pb)

        full = test_cointegration(pa, pb, alpha)
        roll = rolling_cointegration(merged["date"], pa, pb, window=window, step=step, significance_level=alpha)

        pass_rate = roll["is_cointegrated"].mean()
        hr_mean = roll["hedge_ratio"].mean()
        hr_cv = roll["hedge_ratio"].std() / abs(hr_mean) if hr_mean != 0 else float("inf")
        # 最近一個窗口是否仍通過（關係當下是否有效）
        latest_pass = bool(roll.iloc[-1]["is_cointegrated"]) if not roll.empty else False

        rows.append(
            {
                "pair": f"{sid_a} {name_map.get(sid_a, '')} x {sid_b} {name_map.get(sid_b, '')}",
                "n_obs": len(merged),
                "full_pvalue": full.p_value,
                "full_pass": full.is_cointegrated,
                "rolling_pass_rate": pass_rate,
                "hedge_ratio_mean": hr_mean,
                "hedge_ratio_cv": hr_cv,
                "latest_window_pass": latest_pass,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 排序：全區間通過 > 滾動通過率高 > hedge ratio 穩定
    df = df.sort_values(
        ["full_pass", "rolling_pass_rate", "hedge_ratio_cv"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return df


if __name__ == "__main__":
    config = load_config()
    pd.set_option("display.width", 200)

    for scan_cfg in config["scans"]:
        df = scan_pairs(config, scan_cfg)
        if df.empty:
            print(f"[{scan_cfg['name']}] 沒有任何配對完成檢定。")
            continue

        print(f"\n=== [{scan_cfg['name']}] 掃描結果（依穩定度排序）===")
        print(
            df.to_string(
                index=False,
                formatters={
                    "full_pvalue": "{:.4f}".format,
                    "rolling_pass_rate": "{:.0%}".format,
                    "hedge_ratio_mean": "{:.4f}".format,
                    "hedge_ratio_cv": "{:.2f}".format,
                },
            )
        )
        out = f"scan_{scan_cfg['name']}.csv"
        df.to_csv(out, index=False)
        print(f"結果已存至 {out}")
