"""工具選擇引擎：依每檔標的實際可用工具，判斷多空可執行的組合。

四種衍生工具的資料源與方向對應（這是本引擎的核心規則，四者不可混為一談）：
- 個股期貨（TAIFEX）：有期貨 -> 多可買期貨、空可賣期貨（雙向）
- 股票選擇權（TAIFEX，僅 34 檔）：有選擇權 -> 多可買買權/空可買賣權（雙向，此處以「選擇權」概稱）
- 認購權證（券商發行，證交所掛牌）：只做「多」方——看漲工具
- 認售權證（券商發行）：只做「空」方——看跌工具；台股認售遠比認購稀少

「達明只能融券、亞光多選項」的正確規則化：
達明有 23 檔認購權證但 0 檔認售、無期貨無選擇權 ->
  多方可用現股/融資/認購權證；空方只剩融券一途（認售不存在）。
亞光有期貨、108 檔認購但 0 認售 ->
  多方可用現股/融資/期貨/認購權證；空方可用融券/期貨空單。
"""
from dataclasses import dataclass, field

from data.fetchers.taifex_client import get_tradable_map
from data.fetchers.twse_client import get_warrant_map


@dataclass
class InstrumentOptions:
    stock_id: str
    name: str
    has_futures: bool
    has_options: bool
    has_call_warrant: bool
    has_put_warrant: bool
    long_tools: list = field(default_factory=list)
    short_tools: list = field(default_factory=list)


# 融券可放空與否需另查（處置股、平盤下不得放空等），此處先假設一般股票可融券。
# 若標的暫停融券，之後可從 config 或即時資料覆寫。
def select_instruments(
    stock_id: str,
    name: str,
    tradable: dict,
    warrant_map: dict,
    margin_shortable: bool = True,
) -> InstrumentOptions:
    info = tradable.get(stock_id, {})
    has_futures = bool(info.get("has_futures", False))
    has_options = bool(info.get("has_options", False))
    # 權證資料以標的「名稱」彙整（TWSE 權證表用名稱非代號）
    w = warrant_map.get(name, {})
    has_call = bool(w.get("has_call_warrant", False))
    has_put = bool(w.get("has_put_warrant", False))

    long_tools = ["現股買進", "融資買進"]
    short_tools = []
    if margin_shortable:
        short_tools.append("融券賣出")
    if has_futures:
        long_tools.append("個股期貨(多)")
        short_tools.append("個股期貨(空)")
    if has_options:
        long_tools.append("買進買權(選擇權)")
        short_tools.append("買進賣權(選擇權)")
    if has_call:
        long_tools.append(f"認購權證({w.get('n_call', 0)}檔)")
    if has_put:
        short_tools.append(f"認售權證({w.get('n_put', 0)}檔)")

    return InstrumentOptions(
        stock_id=stock_id,
        name=name,
        has_futures=has_futures,
        has_options=has_options,
        has_call_warrant=has_call,
        has_put_warrant=has_put,
        long_tools=long_tools,
        short_tools=short_tools,
    )


def select_pair_instruments(long_leg: dict, short_leg: dict) -> dict:
    """對一組配對（long_leg 做多、short_leg 放空）回傳雙腳可用工具。

    long_leg / short_leg 格式：{'id':..., 'name':..., 'margin_shortable': bool(可選)}
    """
    tradable = get_tradable_map()
    warrant_map = get_warrant_map()
    long_opts = select_instruments(
        long_leg["id"], long_leg["name"], tradable, warrant_map,
        margin_shortable=long_leg.get("margin_shortable", True),
    )
    short_opts = select_instruments(
        short_leg["id"], short_leg["name"], tradable, warrant_map,
        margin_shortable=short_leg.get("margin_shortable", True),
    )
    return {
        "long_leg": long_opts,
        "short_leg": short_opts,
        "short_leg_executable": bool(short_opts.short_tools),  # 放空腳是否有任何可執行工具
    }
