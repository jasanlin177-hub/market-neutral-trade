"""回測入口：對 config 配對跑 z-score 均值回歸回測並輸出績效。"""
import sys

import yaml

from backtest.engine import run_backtest, walk_forward_backtest
from backtest.performance import compute_stats
from data.fetchers.finmind_client import get_pair_prices
from signals.cointegration import test_cointegration, to_log


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run():
    config = load_config()
    data_cfg = config["data"]
    coint_cfg = config["cointegration"]
    bt_cfg = config["backtest"]

    for pair in config["pairs"]:
        stock_a, stock_b = pair["stock_a"], pair["stock_b"]
        print(f"\n[{pair['name']}] 回測 {stock_a['name']}({stock_a['id']}) / "
              f"{stock_b['name']}({stock_b['id']})")

        prices = get_pair_prices(
            stock_a["id"], stock_b["id"],
            data_cfg["dataset"], data_cfg["start_date"], data_cfg["end_date"],
        )
        log_a, log_b = to_log(prices[stock_a["id"]]), to_log(prices[stock_b["id"]])

        coint = test_cointegration(log_a, log_b, coint_cfg["significance_level"])
        print(f"  共整合 p-value={coint.p_value:.4f}（通過={coint.is_cointegrated}）")

        mode = bt_cfg.get("mode", "walk_forward")
        if mode == "walk_forward":
            print(f"  回測模式: walk-forward（估計窗 {bt_cfg['estimation_window']} 天，"
                  f"每 {bt_cfg['reestimate_every']} 天重估）")
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
        else:
            print(f"  回測模式: simple（全樣本 hedge_ratio={coint.hedge_ratio:.4f}，會高估）")
            result = run_backtest(
                log_a, log_b, coint.hedge_ratio,
                lookback=bt_cfg["lookback"],
                entry_z=bt_cfg["entry_z"],
                exit_z=bt_cfg["exit_z"],
                stop_z=bt_cfg["stop_z"],
                cost_per_turn=bt_cfg["cost_per_turn"],
            )
        stats = compute_stats(result.daily_returns, result.trade_returns)

        print(f"  交易次數: {stats.n_trades}，勝率: {stats.win_rate:.0%}，"
              f"平均每筆報酬: {stats.avg_trade_return:.2%}")
        print(f"  總報酬: {stats.total_return:.2%}，年化報酬: {stats.annualized_return:.2%}，"
              f"年化波動: {stats.annualized_vol:.2%}")
        print(f"  Sharpe: {stats.sharpe:.2f}，最大回撤: {stats.max_drawdown:.2%}")

        out = f"backtest_{pair['name']}.csv"
        df = result.daily_returns.to_frame("daily_return")
        df["position"] = result.positions
        df["zscore"] = result.zscore
        df["equity"] = result.equity
        df.index = prices["date"]
        df.to_csv(out)
        print(f"  每日明細已存至 {out}")


if __name__ == "__main__":
    try:
        run()
    except (ValueError, RuntimeError) as e:
        print(f"回測失敗: {e}", file=sys.stderr)
        sys.exit(1)
