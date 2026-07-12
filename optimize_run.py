"""參數最佳化診斷入口：對 config 配對跑網格搜尋 + 半衰期，示範樣本外驗證。"""
import sys

import pandas as pd
import yaml

from backtest.engine import _ols_hedge_ratio
from backtest.optimize import grid_search, half_life
from data.fetchers.finmind_client import get_pair_prices
from signals.cointegration import test_cointegration, to_log
from signals.zscore import spread_series


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run():
    config = load_config()
    data_cfg = config["data"]
    bt_cfg = config["backtest"]

    for pair in config["pairs"]:
        stock_a, stock_b = pair["stock_a"], pair["stock_b"]
        print(f"\n{'='*70}\n[{pair['name']}] {stock_a['name']} / {stock_b['name']}")

        prices = get_pair_prices(stock_a["id"], stock_b["id"],
                                 data_cfg["dataset"], data_cfg["start_date"], data_cfg["end_date"])
        log_a, log_b = to_log(prices[stock_a["id"]]), to_log(prices[stock_b["id"]])

        coint = test_cointegration(log_a, log_b, config["cointegration"]["significance_level"])
        hr = coint.hedge_ratio
        spread = spread_series(log_a, log_b, hr)
        hl = half_life(spread)
        print(f"共整合 p-value={coint.p_value:.4f}（通過={coint.is_cointegrated}）")
        print(f"均值回歸半衰期：{hl:.1f} 個交易日" if hl == hl else
              "均值回歸半衰期：無（價差未呈均值回歸，b>=0）")

        print("\n網格搜尋（依『樣本內 Sharpe』由高到低，前 8 名）：")
        df = grid_search(
            log_a, log_b,
            estimation_window=bt_cfg["estimation_window"],
            reestimate_every=bt_cfg["reestimate_every"],
            lookback=bt_cfg["lookback"], cost_per_turn=bt_cfg["cost_per_turn"],
        )
        pd.set_option("display.width", 200)
        print(df.head(8).to_string(index=False, formatters={
            "in_sample_sharpe": "{:.2f}".format, "in_sample_return": "{:.1%}".format,
            "out_sample_sharpe": "{:.2f}".format, "out_sample_return": "{:.1%}".format,
        }))

        best = df.iloc[0]
        print(f"\n樣本內最佳參數：entry±{best['entry_z']}, exit{best['exit_z']}, stop±{best['stop_z']}")
        print(f"  樣本內 Sharpe {best['in_sample_sharpe']:.2f} / 報酬 {best['in_sample_return']:.1%}")
        print(f"  → 樣本外 Sharpe {best['out_sample_sharpe']:.2f} / 報酬 {best['out_sample_return']:.1%}")
        corr = df["in_sample_sharpe"].corr(df["out_sample_sharpe"])
        print(f"\n所有參數組的『樣本內 Sharpe vs 樣本外 Sharpe』相關係數：{corr:.2f}")
        print("（接近 0 或負值＝樣本內好的參數在樣本外沒用，就是 curve-fitting 的鐵證）")


if __name__ == "__main__":
    try:
        run()
    except (ValueError, RuntimeError) as e:
        print(f"執行失敗: {e}", file=sys.stderr)
        sys.exit(1)
