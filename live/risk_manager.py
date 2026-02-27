"""
リスク管理・ポジションサイジング
==================================
固定リスク方式 (ATRベース) でポジションサイズを計算。

計算式:
  risk_amount = balance × risk_pct          (例: $10,000,000 × 5% = $500,000)
  sl_distance = sl_atr × ATR               (例: 1.5 × $30 = $45)
  units       = risk_amount / sl_distance   (例: $500,000 / $45 = 11,111)

OANDA XAU_USD/XAG_USD:
  1 unit = 1 troy ounce
  P&L = units × 価格変動 (USD)
  → SL距離 (USD) = sl_atr × ATR (USD単位) ← 直接使える
"""

import logging

log = logging.getLogger('sena3fx')


def calc_position_size(balance: float, risk_pct: float,
                        sl_distance: float, max_units: int = None) -> int:
    """
    ATRベースのポジションサイズを計算。

    Args:
        balance:     口座残高 (USD)
        risk_pct:    1トレードあたりのリスク率 (例: 0.05 = 5%)
        sl_distance: SL距離 (USD) = sl_atr × ATR
        max_units:   最大ユニット数 (上限キャップ、None=無制限)

    Returns:
        int: 発注ユニット数 (0以上)
    """
    if balance <= 0 or risk_pct <= 0 or sl_distance <= 0:
        return 0

    risk_amount = balance * risk_pct
    units = int(risk_amount / sl_distance)

    if units <= 0:
        log.warning(
            f"計算されたユニット数が0以下: "
            f"balance={balance:.2f}, risk_pct={risk_pct:.3f}, "
            f"sl_distance={sl_distance:.4f}"
        )
        return 0

    if max_units is not None and units > max_units:
        log.info(f"ユニット数をmax_units={max_units}でキャップ (計算値: {units})")
        units = max_units

    return units


def check_drawdown_limit(current_balance: float, peak_balance: float,
                          max_dd_pct: float = 0.15) -> bool:
    """
    最大ドローダウン超過チェック。

    Args:
        current_balance: 現在の口座残高
        peak_balance:    ピーク時残高
        max_dd_pct:      最大許容ドローダウン率 (例: 0.15 = 15%)

    Returns:
        bool: True = 取引継続可, False = 上限超過につき停止
    """
    if peak_balance <= 0:
        return True

    drawdown = (peak_balance - current_balance) / peak_balance

    if drawdown >= max_dd_pct:
        log.warning(
            f"最大ドローダウン超過: {drawdown:.1%} >= {max_dd_pct:.1%} "
            f"(残高: ${current_balance:,.2f}, ピーク: ${peak_balance:,.2f})"
        )
        return False

    return True


def calc_sl_tp_prices(entry_price: float, atr: float,
                       sl_atr: float, tp_atr: float,
                       side: str = 'long') -> tuple:
    """
    SL / TP 価格を計算。

    Args:
        entry_price: エントリー価格
        atr:         ATR(14)値
        sl_atr:      SLのATR倍率 (例: 1.5)
        tp_atr:      TPのATR倍率 (例: 4.5)
        side:        'long' または 'short'

    Returns:
        (sl_price: float, tp_price: float)
    """
    sl_dist = sl_atr * atr
    tp_dist = tp_atr * atr

    if side == 'long':
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:  # short
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    return sl_price, tp_price
