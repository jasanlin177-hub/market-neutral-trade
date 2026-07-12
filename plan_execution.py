"""執行層示範：對配對計算可用工具與成本試算。

流程：工具選擇引擎（多空各腳可用工具） -> 若放空腳可執行 -> 成本試算 + 損益兩平點。
沿用 check_risk 的部位配置得出實際市值。
"""
import sys

import yaml

from data.fetchers.finmind_client import get_pair_prices
from execution.cost_calculator import (
    pair_breakeven,
    stock_long_cost,
    stock_short_cost,
    warrant_leg_cost,
)
from execution.instrument_selector import select_pair_instruments
from execution.warrant_pricing import bs_delta
from execution.warrant_selector import WarrantScreenConfig, screen_warrants
from risk.position_sizing import dollar_neutral


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run():
    config = load_config()
    data_cfg = config["data"]
    risk_cfg = config["risk"]
    cost_cfg = config["execution"]["cost"]

    for pair in config["pairs"]:
        stock_a, stock_b = pair["stock_a"], pair["stock_b"]
        print(f"\n[{pair['name']}] 做多 {stock_a['name']}({stock_a['id']})、"
              f"放空 {stock_b['name']}({stock_b['id']})")

        # 1) 工具選擇引擎
        sel = select_pair_instruments(
            {"id": stock_a["id"], "name": stock_a["name"]},
            {"id": stock_b["id"], "name": stock_b["name"]},
        )
        lo, so = sel["long_leg"], sel["short_leg"]
        print(f"  多方腳可用工具: {' / '.join(lo.long_tools)}")
        print(f"  空方腳可用工具: {' / '.join(so.short_tools) if so.short_tools else '（無法放空！）'}")

        if not sel["short_leg_executable"]:
            print(f"  => 放空腳 {so.name} 無任何可執行工具，此配對無法建立空單。")
            continue

        # 2) 部位配置（取現價）
        prices = get_pair_prices(
            stock_a["id"], stock_b["id"],
            data_cfg["dataset"], data_cfg["start_date"], data_cfg["end_date"],
        )
        price_a = float(prices[stock_a["id"]].iloc[-1])
        price_b = float(prices[stock_b["id"]].iloc[-1])
        plan = dollar_neutral(risk_cfg["capital"], price_a, price_b, lot_size=risk_cfg["lot_size"])

        # 3) 成本試算（示範：多腳融資、空腳融券）
        days = cost_cfg["hold_days"]
        long_cost = stock_long_cost(
            plan.notional_long, days,
            cost_cfg["fee_rate"], cost_cfg["stock_tax_rate"],
            cost_cfg["margin_interest_rate"], cost_cfg["financing_pct"], cost_cfg["fee_discount"],
        )
        short_cost = stock_short_cost(
            plan.notional_short, days,
            cost_cfg["fee_rate"], cost_cfg["stock_tax_rate"],
            cost_cfg["borrow_rate"], cost_cfg["fee_discount"],
        )
        gross = (plan.notional_long + plan.notional_short) / 2
        result = pair_breakeven(long_cost, short_cost, gross)

        print(f"  [方案A：多腳融資 / 空腳融券]")
        print(f"    多方成本({long_cost.tool}): {long_cost.total_cost:,.0f} 元"
              f"（含 {days} 天融資息 {long_cost.holding_cost:,.0f}）")
        print(f"    空方成本({short_cost.tool}): {short_cost.total_cost:,.0f} 元"
              f"（含 {days} 天借券費 {short_cost.holding_cost:,.0f}）")
        print(f"    雙腳總成本: {result.total_cost:,.0f} 元，"
              f"損益兩平: 價差需變動 {result.breakeven_move_pct:.2%}")

        # 4) 方案B：多腳改用認購權證（若有），把價差 + theta 衰減算進成本
        if lo.has_call_warrant:
            ws_cfg = config["execution"]["warrant_screen"]
            screen = WarrantScreenConfig(
                direction="long",
                min_avg_volume=ws_cfg["min_avg_volume"],
                min_days_to_expiry=ws_cfg["min_days_to_expiry"],
                moneyness_band=(ws_cfg["moneyness_low"], ws_cfg["moneyness_high"]),
                max_implied_vol=ws_cfg["max_implied_vol"],
            )
            as_of = prices["date"].max()
            wdf, rate_limited = screen_warrants(stock_a["name"], stock_a["id"], as_of, screen)
            if rate_limited:
                print(f"  [警告] FinMind API 額度用盡，權證選券結果可能不完整。")
            if wdf.empty:
                print(f"  [方案B：多腳認購權證] 無權證通過選券篩選，略過。")
                continue
            best = wdf.iloc[0]
            # Delta-matched 部位：讓權證的「等效股數」對上方案A多方腳的股數，
            # 而非花等額權利金（那會因權證槓桿造成曝險爆量、theta 虛胖）。
            # 等效股數 = n_units * exercise_ratio * delta
            delta = bs_delta(
                spot=price_a, strike=best["strike"],
                t=int(best["days_to_expiry"]) / 365.0, r=0.015,
                sigma=best["implied_vol"], is_call=True,
            )
            target_shares = plan.shares_long          # 對齊方案A的多方股數
            n_units = int(target_shares / (best["exercise_ratio"] * delta))
            wcost = warrant_leg_cost(
                warrant_price=best["price"], n_units=n_units, days=days,
                spot=price_a, strike=best["strike"],
                implied_vol=best["implied_vol"],
                exercise_ratio=best["exercise_ratio"],
                days_to_expiry=int(best["days_to_expiry"]),
                is_call=True,
                fee_rate=cost_cfg["fee_rate"], fee_discount=cost_cfg["fee_discount"],
                warrant_tax_rate=cost_cfg["warrant_tax_rate"],
                spread_pct=cost_cfg["warrant_spread_pct"],
            )
            result_w = pair_breakeven(wcost, short_cost, gross)
            premium = best["price"] * n_units
            print(f"  [方案B：多腳認購權證 {best['name']}({best['warrant_id']}) / 空腳融券]"
                  f"（delta-matched）")
            print(f"    選中權證: IV {best['implied_vol']:.0%}、價內外 {best['moneyness']:.3f}、"
                  f"delta {delta:.3f}、到期 {int(best['days_to_expiry'])} 天、"
                  f"日均量 {best['avg_volume']:.0f} 張")
            print(f"    部位: {n_units:,} 單位（equiv. {target_shares:,} 股曝險），"
                  f"權利金僅 {premium:,.0f} 元（vs 方案A多腳 {plan.notional_long:,.0f}）")
            print(f"    權證成本 {wcost.total_cost:,.0f} 元 = 手續費 {wcost.entry_cost + wcost.exit_cost:,.0f}"
                  f" + 價差 {wcost.spread_cost:,.0f} + theta 衰減 {wcost.theta_cost:,.0f}")
            print(f"    雙腳總成本: {result_w.total_cost:,.0f} 元，"
                  f"損益兩平: 價差需變動 {result_w.breakeven_move_pct:.2%}")
            print(f"    => theta 是權證方案的隱形大宗成本，持有 {days} 天光時間衰減就吃掉 "
                  f"{wcost.theta_cost:,.0f} 元")


if __name__ == "__main__":
    try:
        run()
    except (ValueError, RuntimeError) as e:
        print(f"執行失敗: {e}", file=sys.stderr)
        sys.exit(1)
