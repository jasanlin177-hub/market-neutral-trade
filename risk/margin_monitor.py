"""保證金維持率監控：模擬融資融券部位的整戶擔保維持率。

台股一般規則（實際成數與門檻以券商與主管機關公告為準，可在 config 調整）：
- 融資：上市股自備 40%、融資 60%
- 融券：保證金成數 90%，擔保品 = 保證金 + 賣出價金
- 整戶擔保維持率 = 擔保品總值 / 債務總值，低於 130% 追繳
  其中 擔保品總值 = 融資股票市值 + 融券保證金 + 融券賣出價金
       債務總值   = 融資金額 + 融券股票現值
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MarginState:
    maintenance_ratio: float       # 整戶擔保維持率
    margin_call: bool              # 是否低於追繳門檻
    long_move_to_call: float       # 多方腳單獨下跌多少比例會觸及追繳（空方腳價格不變）
    short_move_to_call: float      # 空方腳單獨上漲多少比例會觸及追繳（多方腳價格不變）


def _ratio(
    long_value: float,
    financing_amount: float,
    short_collateral: float,
    short_value: float,
) -> float:
    debt = financing_amount + short_value
    return (long_value + short_collateral) / debt if debt > 0 else np.inf


def pair_maintenance_ratio(
    entry_price_long: float,
    entry_price_short: float,
    current_price_long: float,
    current_price_short: float,
    shares_long: int,
    shares_short: int,
    financing_pct: float = 0.6,
    short_margin_pct: float = 0.9,
    call_threshold: float = 1.3,
) -> MarginState:
    """計算「融資買多方腳 + 融券放空空方腳」的整戶擔保維持率。

    long_move_to_call / short_move_to_call 用數值搜尋估算單腳不利變動
    多少比例會觸及追繳門檻，讓使用者知道離斷頭還有多遠。
    """
    financing_amount = entry_price_long * shares_long * financing_pct
    short_proceeds = entry_price_short * shares_short           # 融券賣出價金
    short_deposit = short_proceeds * short_margin_pct           # 融券保證金
    short_collateral = short_proceeds + short_deposit

    def ratio_at(pl: float, ps: float) -> float:
        return _ratio(pl * shares_long, financing_amount, short_collateral, ps * shares_short)

    current = ratio_at(current_price_long, current_price_short)

    # 多方腳單獨下跌：搜尋跌幅
    long_move = np.nan
    for pct in np.arange(0.0, 1.0, 0.005):
        if ratio_at(current_price_long * (1 - pct), current_price_short) < call_threshold:
            long_move = pct
            break

    # 空方腳單獨上漲：搜尋漲幅（上限 3 倍，軋空可能遠超過）
    short_move = np.nan
    for pct in np.arange(0.0, 3.0, 0.005):
        if ratio_at(current_price_long, current_price_short * (1 + pct)) < call_threshold:
            short_move = pct
            break

    return MarginState(
        maintenance_ratio=current,
        margin_call=current < call_threshold,
        long_move_to_call=long_move,
        short_move_to_call=short_move,
    )
