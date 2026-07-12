"""風控守門：對指定配對跑完整的進場前檢查。

流程：抓價格與籌碼 -> 共整合檢定 -> 部位試算 -> 逐項風控檢查 -> 放行/擋下。
假設方向：做多 stock_a、放空 stock_b（軋空檢查針對放空腳 stock_b）。
"""
import sys

import yaml

from data.fetchers.finmind_client import get_margin_short_data, get_pair_prices
from risk.buyin_calendar import check_buyin_risk
from risk.gate import CheckLevel, CheckResult, evaluate
from risk.margin_monitor import pair_maintenance_ratio
from risk.position_sizing import dollar_neutral
from risk.squeeze_monitor import check_squeeze_risk
from risk.stop_loss import spread_zscore
from signals.cointegration import test_cointegration, to_log


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_pair(stock_a: dict, stock_b: dict, config: dict):
    """對單一配對（做多 stock_a、放空 stock_b）跑六道風控守門，回傳 GateDecision。

    stock_a / stock_b 格式：{'id':..., 'name':...}，不需預先寫在 config 的 pairs 清單裡，
    供批次掃描/手動輸入等動態選出的配對直接呼叫。
    """
    data_cfg = config["data"]
    coint_cfg = config["cointegration"]
    risk_cfg = config["risk"]

    prices = get_pair_prices(
        stock_a["id"], stock_b["id"],
        data_cfg["dataset"], data_cfg["start_date"], data_cfg["end_date"],
    )
    margin_b = get_margin_short_data(
        stock_b["id"], data_cfg["start_date"], data_cfg["end_date"]
    )

    log_a, log_b = to_log(prices[stock_a["id"]]), to_log(prices[stock_b["id"]])
    checks = []

    # 檢查 1：共整合（策略統計前提）
    coint = test_cointegration(log_a, log_b, coint_cfg["significance_level"])
    checks.append(CheckResult(
        name="共整合檢定",
        level=CheckLevel.BLOCK if risk_cfg.get("cointegration_gate", True) else CheckLevel.WARN,
        passed=coint.is_cointegrated,
        detail=f"p-value={coint.p_value:.4f}（門檻 {coint_cfg['significance_level']}），"
               f"hedge_ratio={coint.hedge_ratio:.4f}",
    ))

    # 檢查 2：價差 z-score 停損
    sl_cfg = risk_cfg["stop_loss"]
    state = spread_zscore(log_a, log_b, coint.hedge_ratio,
                          lookback=sl_cfg["lookback"], stop_z=sl_cfg["stop_z"])
    checks.append(CheckResult(
        name="價差停損",
        level=CheckLevel.BLOCK,
        passed=not state.stop_triggered,
        detail=f"z-score={state.zscore:.2f}（停損門檻 ±{sl_cfg['stop_z']}）",
    ))

    # 檢查 3：部位配置與淨曝險
    price_a_now = float(prices[stock_a["id"]].iloc[-1])
    price_b_now = float(prices[stock_b["id"]].iloc[-1])
    plan = dollar_neutral(risk_cfg["capital"], price_a_now, price_b_now,
                          lot_size=risk_cfg["lot_size"])
    max_net = risk_cfg["capital"] * risk_cfg["max_net_exposure_pct"]
    checks.append(CheckResult(
        name="淨曝險",
        level=CheckLevel.BLOCK,
        passed=abs(plan.net_exposure) <= max_net,
        detail=f"多 {plan.shares_long} 股({plan.notional_long:,.0f}) / "
               f"空 {plan.shares_short} 股({plan.notional_short:,.0f})，"
               f"淨曝險 {plan.net_exposure:,.0f}（上限 ±{max_net:,.0f}）",
    ))

    # 檢查 4：放空腳軋空風險
    sq_cfg = risk_cfg["squeeze"]
    alert = check_squeeze_risk(
        margin_b,
        surge_days=sq_cfg["surge_days"],
        surge_multiplier=sq_cfg["surge_multiplier"],
        ratio_threshold=sq_cfg["ratio_threshold"],
        utilization_threshold=sq_cfg["utilization_threshold"],
    )
    util_blocked = alert.short_utilization >= sq_cfg["utilization_threshold"]
    checks.append(CheckResult(
        name="軋空預警（融券使用率）",
        level=CheckLevel.BLOCK,
        passed=not util_blocked,
        detail=f"融券使用率 {alert.short_utilization:.1%}"
               f"（門檻 {sq_cfg['utilization_threshold']:.0%}）",
    ))
    checks.append(CheckResult(
        name="軋空預警（餘額暴增/券資比）",
        level=CheckLevel.WARN,
        passed=not any("融券餘額" in r or "券資比" in r for r in alert.reasons),
        detail="; ".join(alert.reasons) if alert.reasons
               else f"券資比 {alert.short_margin_ratio:.1%}，融券餘額 {alert.short_balance_now:.0f} 張，無異常",
    ))

    # 檢查 5：放空腳強制回補日曆
    bc_cfg = risk_cfg["buyin_calendar"]
    today = prices["date"].max()  # 用資料最新交易日當作「今天」
    buyin = check_buyin_risk(
        stock_b["id"],
        today=today,
        lookahead_days=bc_cfg["lookahead_days"],
        buyin_offset_bdays=bc_cfg["buyin_offset_bdays"],
        warning_days=bc_cfg["warning_days"],
    )
    if buyin.has_upcoming_event:
        detail = "; ".join(
            f"{e['kind']} {e['ex_date']}，回補期限 {e['est_last_buyin']}"
            f"（剩 {e['days_left']} 天，{e['source']}）"
            for e in buyin.events
        )
    else:
        detail = f"未來 {bc_cfg['lookahead_days']} 天無官方停券預告或已公告除權息"
    checks.append(CheckResult(
        name="強制回補日曆",
        level=CheckLevel.BLOCK,
        passed=not buyin.triggered,
        detail=detail,
    ))

    # 檢查 6：保證金維持率（以現價進場模擬，看離追繳門檻多遠）
    mm_cfg = risk_cfg["margin_monitor"]
    margin_state = pair_maintenance_ratio(
        entry_price_long=price_a_now,
        entry_price_short=price_b_now,
        current_price_long=price_a_now,
        current_price_short=price_b_now,
        shares_long=max(plan.shares_long, risk_cfg["lot_size"]),
        shares_short=max(plan.shares_short, risk_cfg["lot_size"]),
        financing_pct=mm_cfg["financing_pct"],
        short_margin_pct=mm_cfg["short_margin_pct"],
        call_threshold=mm_cfg["call_threshold"],
    )
    checks.append(CheckResult(
        name="保證金維持率",
        level=CheckLevel.WARN,
        passed=not margin_state.margin_call
               and margin_state.short_move_to_call >= mm_cfg["min_short_buffer"],
        detail=f"進場維持率 {margin_state.maintenance_ratio:.0%}"
               f"（追繳門檻 {mm_cfg['call_threshold']:.0%}），"
               f"多方腳跌 {margin_state.long_move_to_call:.1%} 或"
               f"空方腳漲 {margin_state.short_move_to_call:.1%} 會觸及追繳",
    ))

    return evaluate(checks)


def run():
    """對 config 中所有預設配對跑風控守門，回傳 [(配對名稱, GateDecision), ...]。"""
    config = load_config()
    results = []
    for pair in config["pairs"]:
        stock_a, stock_b = pair["stock_a"], pair["stock_b"]
        print(f"\n[{pair['name']}] 假設交易：做多 {stock_a['name']}({stock_a['id']})、"
              f"放空 {stock_b['name']}({stock_b['id']})")
        decision = check_pair(stock_a, stock_b, config)
        print(decision.summary())
        results.append((pair["name"], decision))
    return results


if __name__ == "__main__":
    try:
        run()
    except (ValueError, RuntimeError) as e:
        print(f"資料抓取失敗: {e}", file=sys.stderr)
        sys.exit(1)
