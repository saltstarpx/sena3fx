"""
Exit Manager — ロットサイズ計算
================================
教材準拠: 「損切り距離でロット調整。遠ければロット減、近ければロット増。固定ロット禁止」

Rの定義:
  1R = 口座残高 × リスク% = 5,000,000 × 3% = 150,000円（固定）
  Rは「そのトレードのSL距離」ではなく「口座ベースの許容損失」として固定する。
  理由: SL距離がトレードごとに変わってもRの比較が一貫する。
        Kill SwitchやGiveback Stopの判定がシンプルになる。

換算例:
  +1R = +150,000円
  +2R = +300,000円
  -1R = -150,000円
  -2R = -300,000円（日次Kill Switch）
  -4R = -600,000円（週次Kill Switch）
"""


def calculate_position_size(
    symbol: str,
    entry_price: float,
    invalidation_price: float,
    config: dict,
) -> dict:
    """
    JPYベースでポジションサイズを計算する。

    Args:
        symbol:             'XAU_USD' または 'XAG_USD'
        entry_price:        エントリー価格（USD）
        invalidation_price: 構造SL価格（USD）= 最後の押し安値/戻り高値
        config:             config.yaml を読み込んだ dict

    Returns:
        dict:
            rejected (bool):        True の場合はエントリー不可
            reason (str):           拒否理由（rejected=True の場合のみ）
            units (int):            発注ユニット数
            risk_jpy (float):       想定リスク金額（JPY）≒ 1R
            sl_distance_usd (float): SL幅（USD）
            rr_at_tp1 (float):      TP1でのRR比（常に 1.0）
    """
    instr_cfg = config['instruments'][symbol]
    factor = instr_cfg['jpy_per_dollar_per_unit']
    max_loss_jpy = config['account']['max_loss_jpy']

    sl_distance = abs(entry_price - invalidation_price)
    if sl_distance <= 0:
        raise ValueError(f"SL距離が0以下: entry={entry_price}, invalidation={invalidation_price}")

    # SL距離の上限チェック
    max_sl = instr_cfg.get('max_sl_distance')
    if max_sl and sl_distance > max_sl:
        return {
            'rejected': True,
            'reason': (
                f"SL幅 ${sl_distance:.2f} が上限 ${max_sl:.2f} を超えています。"
                f"エントリーポイントを見直すか、トレードをスキップしてください。"
            ),
        }

    # units = max_loss_jpy / (jpy_per_dollar_per_unit × sl_distance_usd)
    units = int(max_loss_jpy / (factor * sl_distance))
    risk_jpy = units * factor * sl_distance

    min_u = instr_cfg['min_units']
    max_u = instr_cfg['max_units']

    if units < min_u:
        return {
            'rejected': True,
            'reason': (
                f"必要ロット {units} units が最小単位 {min_u} 未満。"
                f"SL幅が広すぎます（${sl_distance:.2f}）。"
                f"SLを近づけるかトレードをスキップしてください。"
            ),
        }

    # 上限キャップ（risk_jpy が 1R 未満になる）
    if units > max_u:
        units = max_u
        risk_jpy = units * factor * sl_distance

    # 最終リスク確認
    if risk_jpy > max_loss_jpy * 1.01:  # 1%の誤差マージン
        return {
            'rejected': True,
            'reason': (
                f"リスク金額 ¥{risk_jpy:,.0f} が上限 ¥{max_loss_jpy:,.0f} を超えています。"
            ),
        }

    return {
        'rejected': False,
        'units': units,
        'risk_jpy': round(risk_jpy, 0),
        'sl_distance_usd': round(sl_distance, 4),
        'rr_at_tp1': 1.0,
    }


def validate_minimum_size(units: int, symbol: str, config: dict) -> bool:
    """
    ユニット数が最小単位を満たすか確認する。

    Args:
        units:  計算されたユニット数
        symbol: 'XAU_USD' または 'XAG_USD'
        config: config dict

    Returns:
        bool: True = OK, False = 最小単位未満
    """
    min_u = config['instruments'][symbol]['min_units']
    return units >= min_u


def calc_tp1_price(entry_price: float, invalidation_price: float,
                   direction: str, r_multiple: float = 1.0) -> float:
    """
    TP1価格を計算する。

    TP1 = エントリー ± SL幅 × r_multiple
    デフォルト: r_multiple = 1.0 (1R)

    Args:
        entry_price:        エントリー価格
        invalidation_price: 構造SL価格
        direction:          'LONG' または 'SHORT'
        r_multiple:         RR倍率（デフォルト 1.0）

    Returns:
        float: TP1価格
    """
    sl_distance = abs(entry_price - invalidation_price)
    if direction == 'LONG':
        return entry_price + sl_distance * r_multiple
    else:
        return entry_price - sl_distance * r_multiple


def calc_unrealized_r(trade_entry: float, trade_sl: float,
                       current_price: float, direction: str) -> float:
    """
    現在の含み益をR倍率で計算する。

    R = (current_price - entry) / sl_distance  (LONG)
    R = (entry - current_price) / sl_distance  (SHORT)
    sl_distance = |entry - initial_sl|

    Args:
        trade_entry:   エントリー価格
        trade_sl:      初期SL価格
        current_price: 現在価格
        direction:     'LONG' または 'SHORT'

    Returns:
        float: R倍率 (正=利益, 負=損失)
    """
    sl_distance = abs(trade_entry - trade_sl)
    if sl_distance <= 0:
        return 0.0

    if direction == 'LONG':
        return (current_price - trade_entry) / sl_distance
    else:
        return (trade_entry - current_price) / sl_distance


def calc_pnl_jpy(entry_price: float, exit_price: float,
                  units: int, direction: str, symbol: str, config: dict) -> float:
    """
    損益をJPYで計算する。

    Args:
        entry_price: エントリー価格（USD）
        exit_price:  決済価格（USD）
        units:       ユニット数
        direction:   'LONG' または 'SHORT'
        symbol:      'XAU_USD' または 'XAG_USD'
        config:      config dict

    Returns:
        float: 損益（JPY）
    """
    factor = config['instruments'][symbol]['jpy_per_dollar_per_unit']
    if direction == 'LONG':
        pnl_usd = (exit_price - entry_price) * units
    else:
        pnl_usd = (entry_price - exit_price) * units
    return pnl_usd * factor
