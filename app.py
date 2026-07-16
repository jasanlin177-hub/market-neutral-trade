"""配對交易系統 Streamlit Dashboard。

啟動：streamlit run app.py

流程順序（符合實際決策順序，不是資料處理順序）：
  1. 🔍 找配對：批次掃描股票池找候選，或手動輸入兩檔代號/名稱 -> 存成「目前配對」
  2. 之後四個分頁（訊號/共整合、風控守門、回測、執行層）都針對「目前配對」分析
「目前配對」存在 st.session_state，不再依賴 config.yaml 寫死的 pairs 清單。
重運算（抓資料、共整合、回測、選券）都以按鈕觸發並快取，避免每次互動重跑。
"""
import os
import re
import sys

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml

# Streamlit Cloud 用 st.secrets（非 .env 檔）存機密資訊；把它轉成環境變數，
# 讓既有的 os.environ.get("FINMIND_TOKEN") 之類的邏輯本機/雲端都能用。
# 本機沒設 secrets.toml 時 st.secrets 會丟例外，忽略即可（改吃 .env）。
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass
# 診斷（只印「有沒有」，絕不印 token 值）：確認 secrets 是否成功注入
print(f"[startup] FINMIND_TOKEN loaded={bool(os.environ.get('FINMIND_TOKEN'))}",
      file=sys.stderr)

from backtest.engine import walk_forward_backtest
from backtest.performance import compute_stats
from data.fetchers.finmind_client import get_pair_prices, resolve_stock
from signals.cointegration import test_cointegration, to_log
from signals.zscore import rolling_zscore, spread_series

st.set_page_config(page_title="配對交易監控", layout="wide")

# 網路/HTTP 類例外的原始訊息可能包含含 token 的完整 URL，絕不可直接顯示給使用者
_NETWORK_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
    requests.exceptions.RequestException,
)


def data_date_caption(prices) -> str:
    """回傳資料日期提示字串。價格是 FinMind 日收盤，非即時報價。"""
    d = prices["date"].max().date()
    return f"📅 資料日期：{d} 收盤價（FinMind 日 K，**非即時報價**，僅供盤後規劃/監控）"


def _redact_token(text: str) -> str:
    """把 URL 裡的 token=xxx 遮掉，即使印到只有你自己看得到的伺服器端 log 也不留明文。"""
    return re.sub(r"(token=)[^&\s]+", r"\1***REDACTED***", text)


def show_data_error(e: Exception):
    """顯示乾淨的錯誤訊息，絕不輸出可能含 token 的原始 traceback / URL。

    - Streamlit 控制流例外（st.stop / rerun）：原樣重拋，不可吞掉。
    - 網路/HTTP 例外：只顯示通用「連線失敗」訊息（原始訊息含 token）。
    - ValueError：我們自己拋的、訊息安全（如查無股票），可直接顯示。
    - 其他：只顯示例外類型名稱，不顯示可能含敏感資訊的完整訊息。
    完整（token 已遮蔽）的錯誤內容會印到伺服器端 log（stderr），
    只有你自己在 Streamlit Cloud 的「Manage app → Logs」看得到，方便診斷。
    """
    if type(e).__name__ in ("StopException", "RerunException", "RerunData"):
        raise e
    print(f"[show_data_error] {type(e).__name__}: {_redact_token(str(e))}", file=sys.stderr)
    if isinstance(e, _NETWORK_EXCEPTIONS):
        st.error("資料源連線失敗，請稍後再試。（網路或資料源 API 暫時無法連線）")
    elif isinstance(e, ValueError):
        st.error(str(e))
    else:
        st.error(f"分析發生錯誤（{type(e).__name__}），請稍後再試或檢查設定。")


@st.cache_data(ttl=3600)
def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=3600)
def fetch_pair(id_a, id_b, dataset, start, end):
    return get_pair_prices(id_a, id_b, dataset, start, end)


@st.cache_data(ttl=3600)
def resolve_stock_cached(query: str) -> dict:
    return resolve_stock(query)


@st.cache_data(ttl=1800)
def run_cointegration_scan(scan_name: str):
    """即時跑共整合掃描（快取 30 分鐘），回傳 DataFrame。"""
    from scan import scan_pairs
    cfg = load_config()
    scan_cfg = next(s for s in cfg["scans"] if s["name"] == scan_name)
    return scan_pairs(cfg, scan_cfg)


@st.cache_data(ttl=1800)
def run_walkforward_scan(scan_name: str):
    """即時跑 walk-forward 批次回測（快取 30 分鐘），回傳 DataFrame。"""
    from scan_walkforward import scan_walkforward
    cfg = load_config()
    scan_cfg = next(s for s in cfg["scans"] if s["name"] == scan_name)
    return scan_walkforward(cfg, scan_cfg)


@st.cache_data(ttl=1800)
def run_param_optimization(id_a, id_b, dataset, start, end, est_win, reest, lookback, cost):
    """半衰期 + 參數網格搜尋（含樣本外驗證），回傳 (half_life, grid_df, corr)。"""
    from backtest.optimize import grid_search, half_life
    prices = get_pair_prices(id_a, id_b, dataset, start, end)
    log_a, log_b = to_log(prices[id_a]), to_log(prices[id_b])
    coint = test_cointegration(log_a, log_b, 0.05)
    hl = half_life(spread_series(log_a, log_b, coint.hedge_ratio))
    grid = grid_search(log_a, log_b, estimation_window=est_win,
                       reestimate_every=reest, lookback=lookback, cost_per_turn=cost)
    corr = grid["in_sample_sharpe"].corr(grid["out_sample_sharpe"])
    return hl, grid, corr


config = load_config()
data_cfg = config["data"]
coint_cfg = config["cointegration"]

st.title("同產業配對交易監控系統")
st.caption("找配對 → 訊號/共整合 → 風控守門 → 回測 → 執行層")

if "active_pair" not in st.session_state:
    st.session_state.active_pair = None

# 強制重新整理：清掉所有快取資料，下次抓取一律重打 API（盤後想立即看到最新收盤時用）
if st.sidebar.button("🔄 強制重新整理資料", help="清除快取，重新抓取最新收盤資料（否則預設快取 1 小時）"):
    st.cache_data.clear()
    st.sidebar.success("已清除快取，資料將重新抓取")
    st.rerun()

# 側欄只顯示目前選中的配對，不再提供固定下拉選單（配對來源改由分頁1決定）
if st.session_state.active_pair:
    ap = st.session_state.active_pair
    st.sidebar.success("目前配對")
    st.sidebar.write(f"做多：{ap['long']['name']} ({ap['long']['id']})")
    st.sidebar.write(f"放空：{ap['short']['name']} ({ap['short']['id']})")
    st.sidebar.caption(f"來源：{ap['source']}")
else:
    st.sidebar.info("尚未選擇配對，請至「🔍 找配對」分頁選擇")

tab_find, tab_signal, tab_risk, tab_backtest, tab_exec = st.tabs(
    ["🔍 找配對", "📈 訊號/共整合", "🛡️ 風控守門", "🧪 回測", "⚙️ 執行層"]
)

# ---------------- 分頁1：找配對（批次掃描 + 手動輸入） ----------------
with tab_find:
    st.subheader("第一步：決定要分析哪一組配對")
    sub_scan, sub_manual = st.tabs(["📊 批次掃描結果", "✍️ 手動輸入"])

    with sub_scan:
        scan_names = [s["name"] for s in config["scans"]]
        which = st.selectbox("股票池", scan_names, key="scan_pool")
        coint_path, wf_path = f"scan_{which}.csv", f"walkforward_{which}.csv"

        # 即時執行按鈕：直接在 dashboard 觸發，結果存 session_state 也落地成 CSV
        bcol1, bcol2 = st.columns(2)
        if bcol1.button("▶ 執行共整合掃描", key="run_scan"):
            with st.spinner(f"抓 {which} 股票池、兩兩跑共整合檢定..."):
                try:
                    res = run_cointegration_scan(which)
                    if not res.empty:
                        res.to_csv(coint_path, index=False)
                    st.session_state[f"scan_df_{which}"] = res
                except Exception as e:
                    st.error(f"共整合掃描失敗：{e}")
        if bcol2.button("▶ 執行 walk-forward 回測", key="run_wf"):
            with st.spinner(f"對 {which} 通過共整合的配對跑 walk-forward 回測..."):
                try:
                    res = run_walkforward_scan(which)
                    if not res.empty:
                        res.to_csv(wf_path, index=False)
                    st.session_state[f"wf_df_{which}"] = res
                except Exception as e:
                    st.error(f"walk-forward 回測失敗：{e}")

        # 顯示來源優先序：本次執行結果 > 既有 CSV
        df = st.session_state.get(f"scan_df_{which}")
        if df is None:
            try:
                df = pd.read_csv(coint_path)
            except FileNotFoundError:
                df = None
                st.info(f"尚無共整合掃描結果，請按上方「▶ 執行共整合掃描」")

        if df is not None and not df.empty:
            st.write("**共整合掃描結果**（選一列設為目前配對）")
            st.dataframe(df, height=300)
            pair_options = df["pair"].tolist()
            picked = st.selectbox("選擇配對", pair_options, key="scan_pick")
            # pair 欄位格式："3019 亞光 x 4585 達明"，用 " x " 拆兩腳
            left, right = picked.split(" x ")
            left_id, left_name = left.split(" ", 1)
            right_id, right_name = right.split(" ", 1)
            direction = st.radio(
                "做多哪一腳？（放空另一腳）",
                [f"做多 {left_name}", f"做多 {right_name}"],
                key="scan_direction", horizontal=True,
            )
            if direction == f"做多 {left_name}":
                long_leg, short_leg = {"id": left_id, "name": left_name}, {"id": right_id, "name": right_name}
            else:
                long_leg, short_leg = {"id": right_id, "name": right_name}, {"id": left_id, "name": left_name}

            if st.button("設為目前配對", key="use_scan_pick"):
                st.session_state.active_pair = {
                    "long": long_leg, "short": short_leg,
                    "source": f"批次掃描（{which}）",
                }
                st.success(f"已設定：做多 {long_leg['name']} / 放空 {short_leg['name']}")

        wf_df = st.session_state.get(f"wf_df_{which}")
        if wf_df is None:
            try:
                wf_df = pd.read_csv(wf_path)
            except FileNotFoundError:
                wf_df = None
                st.caption("尚無 walk-forward 結果，可按上方「▶ 執行 walk-forward 回測」產生")
        if wf_df is not None and not wf_df.empty:
            st.write("**Walk-forward 回測結果**（同一股票池，僅含通過共整合者，供參考）")
            st.dataframe(wf_df, height=300)

    with sub_manual:
        st.write("輸入股票代號或名稱皆可（例如 3019 或 亞光）")
        c1, c2 = st.columns(2)
        q_long = c1.text_input("做多標的", key="manual_long")
        q_short = c2.text_input("放空標的", key="manual_short")
        if st.button("查詢並設為目前配對", key="use_manual"):
            if not q_long or not q_short:
                st.error("請兩個欄位都輸入")
            else:
                try:
                    with st.spinner("查詢股票代號/名稱..."):
                        long_leg = resolve_stock_cached(q_long)
                        short_leg = resolve_stock_cached(q_short)
                    st.session_state.active_pair = {
                        "long": long_leg, "short": short_leg,
                        "source": "手動輸入",
                    }
                    st.success(f"已設定：做多 {long_leg['name']}({long_leg['id']}) / "
                              f"放空 {short_leg['name']}({short_leg['id']})")
                except ValueError as e:
                    st.error(str(e))

active = st.session_state.active_pair
if active is None:
    st.stop()  # 後面分頁都需要「目前配對」，尚未設定就不繼續渲染

stock_a, stock_b = active["long"], active["short"]

# ---------------- 分頁2：訊號 / 共整合 ----------------
with tab_signal:
    st.subheader(f"{stock_a['name']} × {stock_b['name']} 價差與 z-score")
    if st.button("載入資料並分析", key="sig"):
      try:
        with st.spinner("抓取價格、跑共整合檢定..."):
            prices = fetch_pair(stock_a["id"], stock_b["id"], data_cfg["dataset"],
                                data_cfg["start_date"], data_cfg["end_date"])
            log_a, log_b = to_log(prices[stock_a["id"]]), to_log(prices[stock_b["id"]])
            coint = test_cointegration(log_a, log_b, coint_cfg["significance_level"])
            spread = spread_series(log_a, log_b, coint.hedge_ratio)
            z = rolling_zscore(spread, coint_cfg["rolling"]["window"])

        st.caption(data_date_caption(prices))
        c1, c2, c3 = st.columns(3)
        c1.metric("共整合 p-value", f"{coint.p_value:.4f}",
                  "通過" if coint.is_cointegrated else "未通過")
        c2.metric("hedge ratio", f"{coint.hedge_ratio:.4f}")
        c3.metric("目前 z-score", f"{z.iloc[-1]:.2f}" if pd.notna(z.iloc[-1]) else "N/A")

        # z-score 正負號決定「當下」該做多哪一邊、放空哪一邊——跟找配對分頁選的
        # 「多方/空方」標籤方向可能相反（標籤只是固定 spread 公式裡誰是 A、誰是 B）。
        entry_z = config["backtest"]["entry_z"]
        exit_z = config["backtest"]["exit_z"]
        na, nb = stock_a["name"], stock_b["name"]
        latest_z = z.iloc[-1]

        st.markdown("#### 📌 進場建議")
        if coint.hedge_ratio <= 0:
            # hedge_ratio ≤ 0 時 spread = log(A) - hr*log(B) 退化：
            # B 上漲不再壓低 spread（甚至同向推高），「z 高=A相對貴」這個
            # 經濟意義不成立，z 可能只是兩邊剛好同向暴衝，不是誰貴誰便宜。
            # 這種情況不給方向建議，直接示警，避免機械套用規則反而誤導。
            st.error(
                f"⚠️ **hedge ratio 為負（{coint.hedge_ratio:.4f}），spread 結構退化，"
                f"不提供方向建議**\n\n"
                f"正常配對交易需要 hedge ratio > 0（B 漲則壓低 spread，"
                f"代表 A 相對變便宜）。hedge ratio ≤ 0 時，{nb} 上漲反而會**推高**"
                f"spread，此時 z 偏高很可能只是兩檔剛好同向大漲/大跌，"
                f"不代表「{na} 相對{nb}貴」。"
                f"{'共整合本身也未通過，' if not coint.is_cointegrated else ''}"
                f"這組配對的統計基礎不可信，不建議依 z-score 交易。"
            )
        elif pd.isna(latest_z):
            st.info("z-score 尚無足夠資料（滾動窗口未滿），暫無建議。")
        elif latest_z <= -entry_z:
            st.success(
                f"**✅ 建議進場**（z = {latest_z:.2f} ≤ −{entry_z:.1f}，價差偏低）\n\n"
                f"- 🔴 **做多：{na}**\n- 🟢 **放空：{nb}**\n\n"
                f"預期價差回歸至 0 附近時出場（|z| < {exit_z}）。"
            )
        elif latest_z >= entry_z:
            st.warning(
                f"**✅ 建議進場**（z = {latest_z:.2f} ≥ +{entry_z:.1f}，價差偏高）\n\n"
                f"- 🔴 **做多：{nb}**\n- 🟢 **放空：{na}**\n\n"
                f"⚠️ 方向與你在找配對選的標籤相反（此時該反過來做）。"
                f"預期價差回歸至 0 附近時出場（|z| < {exit_z}）。"
            )
        else:
            # 未達門檻：明確說「不進場」，並預告兩個門檻各自的方向，方便盯盤
            st.info(
                f"**⏸️ 暫不進場**（z = {latest_z:.2f}，在 ±{entry_z:.1f} 之間，價差未明顯偏離）\n\n"
                f"盯盤參考——之後若：\n"
                f"- z 跌破 **−{entry_z:.1f}** → 做多 {na}、放空 {nb}\n"
                f"- z 升破 **+{entry_z:.1f}** → 做多 {nb}、放空 {na}"
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices["date"], y=z, name="z-score"))
        for lvl, dash in [(2, "dash"), (-2, "dash"), (0, "dot")]:
            fig.add_hline(y=lvl, line_dash=dash, line_color="gray")
        fig.update_layout(title="價差 z-score（±2 進場、0 出場）", height=350,
                          margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        LONG_COLOR, SHORT_COLOR = "#d62728", "#2ca02c"  # 做多=紅、做空=綠
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=prices["date"], y=prices[stock_a["id"]],
                                  name=f"{stock_a['name']}（做多）",
                                  line=dict(color=LONG_COLOR)))
        fig2.add_trace(go.Scatter(x=prices["date"], y=prices[stock_b["id"]],
                                  name=f"{stock_b['name']}（做空）",
                                  line=dict(color=SHORT_COLOR), yaxis="y2"))
        fig2.update_layout(
            title="雙腳股價（左軸=做多／紅、右軸=做空／綠）", height=300,
            margin=dict(t=40, b=20),
            yaxis=dict(title=stock_a["name"], tickfont=dict(color=LONG_COLOR),
                      title_font=dict(color=LONG_COLOR)),
            yaxis2=dict(title=stock_b["name"], tickfont=dict(color=SHORT_COLOR),
                       title_font=dict(color=SHORT_COLOR),
                       overlaying="y", side="right"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 股價原始價差（A-B，非對數 spread）：純粹看兩檔實際股價差距的走勢
        price_diff = prices[stock_a["id"]] - prices[stock_b["id"]]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=prices["date"], y=price_diff,
                                  name=f"{stock_a['name']} − {stock_b['name']}",
                                  line=dict(color="#7f7f7f")))
        fig3.add_hline(y=0, line_dash="dot", line_color="lightgray")
        fig3.update_layout(
            title=f"股價原始價差：{stock_a['name']} − {stock_b['name']}（元）",
            height=250, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "此為兩檔股價直接相減（單位：元），與上方用來算 z-score 的對數 spread "
            "（含 hedge ratio 加權）不同——這條線純粹看「絕對價差」，受股價量級影響，"
            "不能直接當進出場依據。"
        )
      except Exception as e:
        show_data_error(e)

# ---------------- 分頁3：風控守門 ----------------
with tab_risk:
    st.subheader(f"{stock_a['name']} × {stock_b['name']} 進場前風控守門")
    st.caption("六道檢查，任一 BLOCK 未過即擋下，不得進入執行層")
    if st.button("執行風控守門", key="risk"):
        from check_risk import check_pair
        with st.spinner("跑共整合/停損/淨曝險/軋空/回補/保證金六道檢查..."):
            try:
                decision = check_pair(stock_a, stock_b, config)
            except Exception as e:
                show_data_error(e)
                decision = None
        if decision:
            st.write(f"**結論** → {'✅ 放行' if decision.approved else '⛔ 擋下'}")
            for c in decision.checks:
                if c.passed:
                    st.success(f"PASS｜{c.name}：{c.detail}")
                elif c.level.value == "block":
                    st.error(f"BLOCK｜{c.name}：{c.detail}")
                else:
                    st.warning(f"WARN｜{c.name}：{c.detail}")

# ---------------- 分頁4：回測 ----------------
with tab_backtest:
    st.subheader(f"{stock_a['name']} × {stock_b['name']} Walk-forward 回測")
    bt_cfg = config["backtest"]

    # 可調參數（預設沿用 config，可即時覆寫試不同進出場點）
    pc1, pc2, pc3 = st.columns(3)
    entry_z = pc1.number_input("進場門檻 |z|", 0.5, 5.0, float(bt_cfg["entry_z"]), 0.5, key="bt_entry")
    exit_z = pc2.number_input("出場門檻 |z|（0＝回到均值才出）", 0.0, 3.0, float(bt_cfg["exit_z"]), 0.5, key="bt_exit")
    stop_z = pc3.number_input("停損門檻 |z|", 1.0, 8.0, float(bt_cfg["stop_z"]), 0.5, key="bt_stop")

    if st.button("執行 walk-forward 回測（三方向對照）", key="bt"):
      try:
        with st.spinner("抓資料、滾動估 hedge ratio、模擬三種方向..."):
            # 回測固定看 5 年：起始日設「今天往前 5 年」；若標的掛牌不滿 5 年，
            # FinMind 會自動從掛牌日開始回傳，等同「以掛牌日為起點」。
            bt_start = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
            prices = fetch_pair(stock_a["id"], stock_b["id"], data_cfg["dataset"],
                                bt_start, data_cfg["end_date"])
            log_a, log_b = to_log(prices[stock_a["id"]]), to_log(prices[stock_b["id"]])
            dir_map = {
                "雙向": "both",
                f"正向（做多{stock_a['name']}/放空{stock_b['name']}）": "long_only",
                f"反向（放空{stock_a['name']}/做多{stock_b['name']}）": "short_only",
            }
            results = {}
            for label, d in dir_map.items():
                res = walk_forward_backtest(
                    log_a, log_b,
                    estimation_window=bt_cfg["estimation_window"],
                    reestimate_every=bt_cfg["reestimate_every"],
                    lookback=bt_cfg["lookback"], entry_z=entry_z,
                    exit_z=exit_z, stop_z=stop_z,
                    cost_per_turn=bt_cfg["cost_per_turn"], direction=d,
                )
                results[label] = (res, compute_stats(res.daily_returns, res.trade_returns))

        st.caption(
            f"📅 回測期間：{prices['date'].min().date()} ~ {prices['date'].max().date()}"
            f"（FinMind 日 K 收盤價；z 碰 ±{entry_z} 進場／回到 |z|≤{exit_z} 出場／"
            f"|z|>{stop_z} 停損；非每日交易）"
        )

        # 三方向績效對照表
        def avg_days(res):
            return f"{sum(res.trade_days) / len(res.trade_days):.1f}" if res.trade_days else "N/A"

        table = pd.DataFrame({
            label: {
                "總報酬": f"{s.total_return:.2%}", "Sharpe": f"{s.sharpe:.2f}",
                "勝率": f"{s.win_rate:.0%}", "最大回撤": f"{s.max_drawdown:.2%}",
                "交易次數": s.n_trades, "平均持有天數(交易日)": avg_days(r),
            } for label, (r, s) in results.items()
        }).T
        st.write("**三方向績效對照**")
        st.dataframe(table, use_container_width=True)

        # 三條權益曲線疊圖
        fig = go.Figure()
        colors = {"雙向": "#1f77b4"}
        for label, (res, s) in results.items():
            fig.add_trace(go.Scatter(x=prices["date"], y=res.equity, name=label,
                                     line=dict(color=colors.get(label))))
        fig.add_hline(y=1.0, line_dash="dot", line_color="lightgray")
        fig.update_layout(title="三方向權益曲線對照（1.0＝損益平衡）", height=380,
                          margin=dict(t=40, b=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "正向＝只做多價差（順配對標籤方向）；反向＝只放空價差（相反）；雙向＝兩邊都做。"
            "walk-forward 已排除前視偏誤。若某單一方向明顯優於雙向，"
            "可能代表這段期間有趨勢性偏移，而非穩定的均值回歸——仍須留意 curve-fitting。"
        )
      except Exception as e:
        show_data_error(e)

    st.markdown("---")
    st.markdown("### 參數最佳化診斷（半衰期 + 網格搜尋 + 樣本外驗證）")
    st.caption("找『最佳進出場點』的正確做法：樣本內找參數、樣本外驗證是否還靈——"
               "而不是在同一段歷史上挑最好看的（那是 curve-fitting）。")
    if st.button("執行參數網格搜尋", key="opt"):
      try:
        with st.spinner("算半衰期、掃 entry/exit/stop 網格、分樣本內外驗證..."):
            hl, grid, corr = run_param_optimization(
                stock_a["id"], stock_b["id"], data_cfg["dataset"],
                data_cfg["start_date"], data_cfg["end_date"],
                bt_cfg["estimation_window"], bt_cfg["reestimate_every"],
                bt_cfg["lookback"], bt_cfg["cost_per_turn"],
            )

        oc1, oc2 = st.columns(2)
        oc1.metric("均值回歸半衰期",
                   f"{hl:.1f} 交易日" if hl == hl else "無（非均值回歸）",
                   help="價差偏離後回到一半距離所需天數；無＝價差不回歸，配對交易前提不成立")
        oc2.metric("樣本內 vs 樣本外 Sharpe 相關", f"{corr:.2f}" if corr == corr else "N/A",
                   help="接近 1＝參數穩健可信；接近 0 或負＝樣本內好的參數樣本外沒用，是 curve-fitting")

        if corr == corr and corr <= 0.2:
            st.error(
                f"⚠️ 相關係數 {corr:.2f} 偏低／為負：樣本內調出來的『最佳參數』在樣本外並不成立，"
                f"這組配對用參數最佳化無法產生穩健 edge，不建議交易。"
            )
        elif corr == corr:
            st.success(f"相關係數 {corr:.2f} 為正：參數相對穩健，最佳化結果較可信。")

        st.write("**網格搜尋結果**（依樣本內 Sharpe 排序；比較同列的樣本內 vs 樣本外）")
        st.dataframe(
            grid.style.format({
                "in_sample_sharpe": "{:.2f}", "in_sample_return": "{:.1%}",
                "out_sample_sharpe": "{:.2f}", "out_sample_return": "{:.1%}",
            }),
            height=350,
        )
        st.caption(
            "若『樣本內 Sharpe 高』的那幾列，其『樣本外 Sharpe』不跟著高（甚至為負），"
            "就代表照樣本內挑參數會被誤導——這正是不能用歷史最佳化騙自己的原因。"
        )
      except Exception as e:
        show_data_error(e)

# ---------------- 分頁5：執行層 ----------------
with tab_exec:
    st.subheader("工具選擇 + 選券 + 部位配置 + 成本試算")
    if st.button("分析可用工具與成本", key="exec"):
      try:
        from execution.cost_calculator import (
            pair_breakeven,
            stock_cash_long_cost,
            stock_long_cost,
            stock_short_cost,
            warrant_leg_cost,
        )
        from execution.instrument_selector import select_pair_instruments
        from execution.warrant_pricing import bs_delta
        from risk.position_sizing import dollar_neutral, short_anchored_neutral

        risk_cfg = config["risk"]
        cost_cfg = config["execution"]["cost"]

        with st.spinner("查 TAIFEX 期貨/選擇權、TWSE 權證..."):
            sel = select_pair_instruments(
                {"id": stock_a["id"], "name": stock_a["name"]},
                {"id": stock_b["id"], "name": stock_b["name"]},
            )
        c1, c2 = st.columns(2)
        c1.write(f"**多方腳 {stock_a['name']} 可用工具**")
        for t in sel["long_leg"].long_tools:
            c1.write(f"- {t}")
        c2.write(f"**空方腳 {stock_b['name']} 可用工具**")
        for t in (sel["short_leg"].short_tools or ["（無法放空）"]):
            c2.write(f"- {t}")

        if not sel["short_leg_executable"]:
            st.error(f"放空腳 {stock_b['name']} 無任何可執行工具，此配對無法建立空單。")
            st.stop()

        # --- 部位配置（金額中性，取現價）---
        st.markdown("---")
        with st.spinner("抓現價、計算部位..."):
            prices = fetch_pair(stock_a["id"], stock_b["id"], data_cfg["dataset"],
                                data_cfg["start_date"], data_cfg["end_date"])
            price_a = float(prices[stock_a["id"]].iloc[-1])
            price_b = float(prices[stock_b["id"]].iloc[-1])

        st.caption(data_date_caption(prices) + "；下單前請以券商即時報價為準")

        # 交易單位：整股(兩腳都整張)或混合(空方整張融券、多方現股零股配平)。
        # 台股放空(融券/借券)都只能整股，零股無法放空；只有多方現股能零股。
        # 高價股用整股顆粒度太粗會配到 0 張，自動建議混合模式。
        round_lot = dollar_neutral(risk_cfg["capital"], price_a, price_b, lot_size=1000)
        default_odd = (round_lot.shares_long == 0 or round_lot.shares_short == 0)
        if default_odd:
            st.warning(
                f"⚠️ 以整股（1000股/張）配置，"
                f"{'多方' if round_lot.shares_long==0 else '空方'}腳會配到 0 張"
                f"（高價股單張金額 > 單腳資金 {risk_cfg['capital']/2:,.0f}）。"
                f"已預設改用**混合模式**（空方整張融券、多方現股零股配平）。"
            )
        unit_mode = st.radio(
            "交易單位",
            ["混合（空方整張融券＋多方現股零股配平）", "整股（兩腳都 1000股/張）"],
            index=0 if default_odd else 1, key="unit_mode", horizontal=True,
        )
        odd_lot = unit_mode.startswith("混合")
        if odd_lot:
            # 空方腳整股融券當基準，多方腳現股零股配平其市值
            plan = short_anchored_neutral(risk_cfg["capital"], price_a, price_b, short_lot=1000)
        else:
            plan = dollar_neutral(risk_cfg["capital"], price_a, price_b, lot_size=1000)

        days = cost_cfg["hold_days"]
        financing_pct = cost_cfg["financing_pct"]
        short_margin_pct = risk_cfg["margin_monitor"]["short_margin_pct"]

        def unit_desc(shares):
            """把股數描述成『N 張』或『N 股（零股）』。"""
            if shares % 1000 == 0 and shares > 0:
                return f"{shares // 1000} 張（{shares:,} 股）"
            return f"{shares:,} 股（零股）"

        if odd_lot:
            # 零股不可信用交易：多方用現股（全額自備、無融資），空方融券不支援零股
            long_cost = stock_cash_long_cost(
                plan.notional_long, cost_cfg["fee_rate"], cost_cfg["stock_tax_rate"],
                cost_cfg["fee_discount"],
            )
            long_self_pay = plan.notional_long  # 現股全額自備
            long_action = f"{stock_a['name']}現股（零股）買進"
        else:
            long_cost = stock_long_cost(
                plan.notional_long, days, cost_cfg["fee_rate"], cost_cfg["stock_tax_rate"],
                cost_cfg["margin_interest_rate"], financing_pct, cost_cfg["fee_discount"],
            )
            long_self_pay = plan.notional_long * (1 - financing_pct)
            long_action = f"{stock_a['name']}融資買進"

        short_cost = stock_short_cost(
            plan.notional_short, days, cost_cfg["fee_rate"], cost_cfg["stock_tax_rate"],
            cost_cfg["borrow_rate"], cost_cfg["fee_discount"],
        )
        short_margin_required = plan.notional_short * short_margin_pct
        gross = (plan.notional_long + plan.notional_short) / 2
        result_a = pair_breakeven(long_cost, short_cost, gross)

        st.write(f"### 方案 A：{long_action} + {stock_b['name']}放空")
        st.markdown(
            f"1. **{long_action} {unit_desc(plan.shares_long)}** @ {price_a:.2f} 元，"
            f"市值 {plan.notional_long:,.0f} 元，"
            + (f"**全額自備約 {long_self_pay:,.0f} 元**（零股不可融資）"
               if odd_lot else
               f"融資成數 {financing_pct:.0%}，**自備資金約 {long_self_pay:,.0f} 元**") + "\n"
            f"2. **{stock_b['name']}融券放空 {unit_desc(plan.shares_short)}** @ {price_b:.2f} 元，"
            f"市值 {plan.notional_short:,.0f} 元，融券保證金成數 {short_margin_pct:.0%}，"
            f"**所需保證金約 {short_margin_required:,.0f} 元**"
        )
        if odd_lot:
            st.caption(
                "混合模式說明：台股放空（融券／借券）**都只能整股，零股一律無法放空**。"
                "因此空方腳以整張融券為基準，多方腳再用現股零股（盤中零股交易）精準配平其市值，"
                "達到金額中性。若空方腳單張金額仍過大導致資金超標，"
                "唯一能再細分的放空工具是個股期貨（小型 100 股／口），但仍非任意股數。"
            )
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric(f"多方成本（{long_cost.tool}）", f"{long_cost.total_cost:,.0f} 元",
                   f"含利息 {long_cost.holding_cost:,.0f}")
        ac2.metric(f"空方成本（{short_cost.tool}）", f"{short_cost.total_cost:,.0f} 元",
                   f"含借券費 {short_cost.holding_cost:,.0f}")
        ac3.metric("損益兩平點", f"{result_a.breakeven_move_pct:.2%}",
                   f"總成本 {result_a.total_cost:,.0f}")
        st.caption(f"淨曝險（多方市值-空方市值）：{plan.net_exposure:,.0f} 元")

        # --- 方案B：認購權證做多（delta-matched）+ 融券放空 ---
        if sel["long_leg"].has_call_warrant:
            from execution.warrant_selector import WarrantScreenConfig, screen_warrants
            ws = config["execution"]["warrant_screen"]
            progress_bar = st.progress(0.0, text="準備篩選權證候選...")
            def _update_progress(done, total):
                progress_bar.progress(done / total, text=f"查詢權證報價中... {done}/{total}")
            try:
                wdf, rate_limited = screen_warrants(
                    stock_a["name"], stock_a["id"], prices["date"].max(),
                    WarrantScreenConfig(
                        direction="long", min_avg_volume=ws["min_avg_volume"],
                        min_days_to_expiry=ws["min_days_to_expiry"],
                        moneyness_band=(ws["moneyness_low"], ws["moneyness_high"]),
                        max_implied_vol=ws["max_implied_vol"],
                    ),
                    progress_callback=_update_progress,
                )
            finally:
                progress_bar.empty()
            st.markdown("---")
            st.write(f"**{stock_a['name']} 認購權證選券結果（依成本/風險排序）**")
            if rate_limited:
                st.warning("FinMind API 額度已用盡，選券結果可能不完整（僅涵蓋額度耗盡前查到的權證），"
                          "請稍後再試。")
            if wdf.empty:
                st.info("無權證通過篩選")
            else:
                st.dataframe(
                    wdf[["warrant_id", "name", "price", "moneyness", "implied_vol",
                         "days_to_expiry", "avg_volume", "score"]].head(15),
                    height=300,
                )

                best = wdf.iloc[0]
                delta = bs_delta(
                    spot=price_a, strike=best["strike"],
                    t=int(best["days_to_expiry"]) / 365.0, r=0.015,
                    sigma=best["implied_vol"], is_call=True,
                )
                target_shares = plan.shares_long   # delta-matched 對齊方案A多方股數
                n_units = int(target_shares / (best["exercise_ratio"] * delta))
                lots_warrant = n_units / 1000.0    # 權證以「張」計，1張=1000單位（可為小數）
                wcost = warrant_leg_cost(
                    warrant_price=best["price"], n_units=n_units, days=days,
                    spot=price_a, strike=best["strike"], implied_vol=best["implied_vol"],
                    exercise_ratio=best["exercise_ratio"],
                    days_to_expiry=int(best["days_to_expiry"]), is_call=True,
                    fee_rate=cost_cfg["fee_rate"], fee_discount=cost_cfg["fee_discount"],
                    warrant_tax_rate=cost_cfg["warrant_tax_rate"],
                    spread_pct=cost_cfg["warrant_spread_pct"],
                )
                result_b = pair_breakeven(wcost, short_cost, gross)
                premium = best["price"] * n_units

                st.write(f"### 方案 B：{stock_a['name']}認購權證（{best['name']}）買進 + "
                        f"{stock_b['name']}融券放空")
                st.markdown(
                    f"1. **{best['name']}（{best['warrant_id']}）買進 {lots_warrant:.2f} 張**"
                    f"（{n_units:,} 單位 @ {best['price']:.2f} 元，delta {delta:.3f}，"
                    f"equiv. {target_shares:,} 股{stock_a['name']}曝險），"
                    f"**權利金（全額自備）約 {premium:,.0f} 元**\n"
                    f"2. **{stock_b['name']}融券放空 {unit_desc(plan.shares_short)}** @ {price_b:.2f} 元，"
                    f"市值 {plan.notional_short:,.0f} 元，融券保證金成數 {short_margin_pct:.0%}，"
                    f"**所需保證金約 {short_margin_required:,.0f} 元**"
                )
                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("權證權利金（自備資金）", f"{premium:,.0f} 元",
                           f"{lots_warrant:.2f} 張")
                bc2.metric("權證成本", f"{wcost.total_cost:,.0f} 元",
                           f"theta 衰減 {wcost.theta_cost:,.0f}")
                bc3.metric("損益兩平點", f"{result_b.breakeven_move_pct:.2%}",
                           f"總成本 {result_b.total_cost:,.0f}")

                st.caption(
                    f"方案A自備資金約 {long_self_pay + short_margin_required:,.0f} 元"
                    f"（多方自備 {long_self_pay:,.0f} + 空方保證金 {short_margin_required:,.0f}），"
                    f"損益兩平 {result_a.breakeven_move_pct:.2%}；"
                    f"方案B自備資金約 {premium + short_margin_required:,.0f} 元"
                    f"（權證權利金 {premium:,.0f} + 空方保證金 {short_margin_required:,.0f}），"
                    f"損益兩平 {result_b.breakeven_move_pct:.2%}。"
                    f"權證方案成本較高（theta 衰減）但最大損失鎖定在權利金，無追繳風險；"
                    f"融資/融券自備資金較低但有保證金追繳風險。"
                )
      except Exception as e:
        show_data_error(e)
