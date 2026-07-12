"""MVP 入口：讀取配對清單 -> 抓資料 -> 跑共整合檢定 -> 輸出結果。"""
import sys

import yaml

from data.fetchers.finmind_client import get_pair_prices
from signals.cointegration import rolling_cointegration, test_cointegration, to_log


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run():
    config = load_config()
    data_cfg = config["data"]
    coint_cfg = config["cointegration"]

    results = []
    for pair in config["pairs"]:
        stock_a = pair["stock_a"]
        stock_b = pair["stock_b"]
        print(f"[{pair['name']}] 抓取 {stock_a['id']} / {stock_b['id']} 收盤價...")

        try:
            prices = get_pair_prices(
                stock_a["id"],
                stock_b["id"],
                data_cfg["dataset"],
                data_cfg["start_date"],
                data_cfg["end_date"],
            )
        except (ValueError, RuntimeError) as e:
            print(f"[{pair['name']}] 資料抓取失敗: {e}")
            continue

        if len(prices) < 30:
            print(f"[{pair['name']}] 對齊後資料筆數過少（{len(prices)} 筆），略過共整合檢定。")
            continue

        pa = prices[stock_a["id"]]
        pb = prices[stock_b["id"]]
        if coint_cfg.get("use_log_prices", False):
            pa, pb = to_log(pa), to_log(pb)
            price_label = "對數價格"
        else:
            price_label = "原始價格"

        result = test_cointegration(
            pa, pb, significance_level=coint_cfg["significance_level"]
        )

        print(
            f"[{pair['name']}] 全區間共整合檢定（{price_label}）: stat={result.stat:.4f}, "
            f"p-value={result.p_value:.4f}, hedge_ratio={result.hedge_ratio:.4f}, "
            f"通過檢定={result.is_cointegrated}"
        )
        results.append({"pair": pair["name"], **vars(result)})

        rolling_cfg = coint_cfg.get("rolling", {})
        if rolling_cfg.get("enabled", False):
            window = rolling_cfg.get("window", 120)
            if len(prices) < window:
                print(f"[{pair['name']}] 資料不足 {window} 筆，略過滾動檢定。")
                continue
            df_roll = rolling_cointegration(
                prices["date"],
                pa,
                pb,
                window=window,
                step=rolling_cfg.get("step", 5),
                significance_level=coint_cfg["significance_level"],
            )
            n_pass = int(df_roll["is_cointegrated"].sum())
            print(
                f"[{pair['name']}] 滾動檢定（window={window}）: "
                f"共 {len(df_roll)} 個窗口，通過 {n_pass} 個"
                f"（{n_pass / len(df_roll):.0%}）"
            )
            passed = df_roll[df_roll["is_cointegrated"]]
            if not passed.empty:
                print(f"[{pair['name']}] 通過檢定的窗口（結束日 / p-value / hedge_ratio）:")
                for _, row in passed.iterrows():
                    print(
                        f"  {row['window_end'].date()}  p={row['p_value']:.4f}  "
                        f"hr={row['hedge_ratio']:.4f}"
                    )
            out_path = f"rolling_{pair['name']}.csv"
            df_roll.to_csv(out_path, index=False)
            print(f"[{pair['name']}] 滾動檢定明細已存至 {out_path}")

    return results


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError as e:
        print(f"設定檔讀取失敗: {e}", file=sys.stderr)
        sys.exit(1)
