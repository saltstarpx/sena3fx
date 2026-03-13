"""
Exit Manager — 出口ルールエンジン
==================================
全ルールは独立した純粋関数（または状態参照最小限の関数）。
OANDA API呼び出しは行わない → 完全にユニットテスト可能。

優先順位（衝突時は優先度が高いルールが常に勝つ）:
  Priority 1:  Kill Switch    → 新規エントリー停止（既存ポジ管理は継続）
  Priority 3:  Max Loss Guard → -1R強制撤退（Hard SLが滑った場合の保険）
  ロックアウト: 60分ロックアウト（short_term）/ 銘柄別時間フィルター（time_filter）
  Priority 4:  TP1            → +1Rで50%利確 + SL建値移動
  Priority 5:  Giveback Stop  → +2R到達後に+1Rまで戻り→全決済
  Priority 6:  Trailing Stop  → 4H足スイングトレール
  Priority 7:  Reversal Exit  → 4H足反転で全決済
  Priority 8:  Time Filter    → 銀24h超+1R未達→撤退（non_textbook）
  Priority 9:  Anti-patterns  → SL拡大/ナンピン/重複利確を拒否（veto層）

例外マトリクス（仕様書 Section 2.6 準拠）:
  - ロックアウト中にSLヒット → SL優先（サーバー側SLは常に実行）
  - Kill Switch中も既存ポジの出口管理は全て継続（新規のみブロック）
  - 60分ロックアウト中に-1R到達 → -1R優先（Priority 3）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from exit_manager.position_manager import TradeState, TradePhase


# ------------------------------------------------------------------ #
#  Action データクラス                                                #
# ------------------------------------------------------------------ #

@dataclass
class Action:
    """
    出口ルールが返すアクション。
    main.py の execute 層がこれを受け取って OANDA API を叩く。

    action_type:
      "CLOSE_ALL"     → 全量決済
      "PARTIAL_CLOSE" → 部分決済（units フィールドで指定）
      "MODIFY_SL"     → SL変更（new_sl フィールドで指定）
      "BLOCK_ENTRY"   → 新規エントリーブロック（Kill Switch）
    """
    action_type:  str
    reason:       str = ""
    units:        Optional[int] = None
    new_sl:       Optional[float] = None
    new_sl_reason: Optional[str] = None
    priority:     int = 9
    non_textbook: bool = False


# ------------------------------------------------------------------ #
#  Priority 1: Kill Switch                                            #
# ------------------------------------------------------------------ #

def check_kill_switch(
    realized_pnl_jpy: float,
    unrealized_pnl_jpy: float,
    config: dict,
    period: str = 'daily',
) -> Optional[Action]:
    """
    Priority 1: 日次/週次の損失上限超過で新規エントリーをブロック。

    計測: realized + unrealized（口座評価ベース）
    既存ポジションの出口管理（SL/TP/Giveback等）は継続。

    Args:
        realized_pnl_jpy:   本日/今週の実現損益（JPY）
        unrealized_pnl_jpy: 現在の含み損益合計（JPY）
        config:             config dict
        period:             'daily' または 'weekly'

    Returns:
        Action(action_type="BLOCK_ENTRY") if triggered, None otherwise.
    """
    ks_cfg = config['exit_rules']['kill_switch']
    if not ks_cfg.get('enabled', True):
        return None

    total_pnl = realized_pnl_jpy + unrealized_pnl_jpy

    if period == 'daily':
        threshold = -abs(float(ks_cfg['daily']['max_loss_jpy']))
    else:
        threshold = -abs(float(ks_cfg['weekly']['max_loss_jpy']))

    if total_pnl <= threshold:
        return Action(
            action_type='BLOCK_ENTRY',
            reason=f'kill_switch_{period}: total_pnl={total_pnl:.0f}jpy <= {threshold:.0f}jpy',
            priority=1,
        )
    return None


# ------------------------------------------------------------------ #
#  Priority 3: Max Loss Guard                                         #
# ------------------------------------------------------------------ #

def check_max_loss_guard(
    trade: TradeState,
    current_price: float,
    config: dict,
    jpy_per_usd: float,
) -> Optional[Action]:
    """
    Priority 3: 含み損が max_loss_jpy (-1R) を超えたら即全決済。

    Hard SL が滑った場合・SL未設定時の保険。
    ティックベース（毎ポーリングサイクルで確認）。

    Args:
        trade:         TradeState
        current_price: 現在の価格（USD）
        config:        config dict
        jpy_per_usd:   jpy_per_dollar_per_unit（instrument設定値）

    Returns:
        Action(action_type="CLOSE_ALL") if triggered, None otherwise.
    """
    cfg = config['exit_rules']['max_loss_guard']
    if not cfg.get('enabled', True):
        return None

    unrealized_jpy = trade.unrealized_pnl_jpy(current_price, jpy_per_usd)
    threshold = -abs(float(cfg['threshold_jpy']))

    if unrealized_jpy <= threshold:
        return Action(
            action_type='CLOSE_ALL',
            reason=(
                f'max_loss_guard: unrealized={unrealized_jpy:.0f}jpy '
                f'<= {threshold:.0f}jpy'
            ),
            priority=3,
        )
    return None


# ------------------------------------------------------------------ #
#  ロックアウト（二段構え）                                           #
# ------------------------------------------------------------------ #

def check_lockout_short_term(trade: TradeState, now: datetime) -> bool:
    """
    第一段ロックアウト: 建玉から60分はSL以外の決済を禁止。

    non_textbook=True でログ記録。
    SLヒット（サーバー側）は常に許可（ロックアウト関係なし）。

    Args:
        trade: TradeState
        now:   現在UTC時刻（TZなし）

    Returns:
        bool: True = ブロック, False = 許可
    """
    cfg = config_get_lockout_short_term_minutes(trade)
    elapsed_min = (now - trade.entry_time).total_seconds() / 60.0
    return elapsed_min < cfg


def config_get_lockout_short_term_minutes(trade: TradeState) -> float:
    """
    ロックアウト時間を返す（デフォルト60分）。
    将来的にconfigから動的に取得する場合のフック。
    """
    return 60.0


def check_lockout_time_filter(
    trade: TradeState,
    now: datetime,
    config: dict,
) -> bool:
    """
    第二段ロックアウト: 銘柄別の勝ちゾーン強制（non_textbook）。

    XAU_USD: min_hold_hours（デフォルト8h）未満は利確/撤退ブロック
    XAG_USD: 短期ロックアウトのみ（24h撤退は check_silver_time_stop が担当）

    non_textbook=True でログ記録。

    Returns:
        bool: True = ブロック, False = 許可
    """
    tf_cfg = config['exit_rules']['lockout']['time_filter']
    if not tf_cfg.get('enabled', True):
        return False

    if trade.instrument == 'XAU_USD':
        min_hours = float(
            tf_cfg.get('XAU_USD', {}).get('min_hold_hours', 8)
        )
        elapsed_h = (now - trade.entry_time).total_seconds() / 3600.0
        if elapsed_h < min_hours:
            return True  # ブロック（non_textbook）

    return False


# ------------------------------------------------------------------ #
#  Priority 4: TP1                                                    #
# ------------------------------------------------------------------ #

def check_tp1(
    trade: TradeState,
    current_price: float,
    config: dict,
) -> Optional[Action]:
    """
    Priority 4: TP1到達で50%利確 + SL建値+α移動。

    条件:
      - phase が OPEN
      - tp1_executed が False（1回のみ）
      - unrealized_r >= tp1.r_multiple（デフォルト1.0）

    Returns:
        Action(action_type="PARTIAL_CLOSE") if triggered, None otherwise.
        ※ SL建値移動は main.py が続けて実行する。
    """
    if trade.tp1_executed:
        return None
    if trade.phase == TradePhase.CLOSED:
        return None

    tp1_cfg = config['exit_rules']['tp1']
    r_multiple = float(tp1_cfg.get('r_multiple', 1.0))
    partial_pct = float(tp1_cfg.get('partial_close_pct', 50)) / 100.0

    current_r = trade.unrealized_r(current_price)
    if current_r < r_multiple:
        return None

    units_to_close = max(1, int(trade.current_units * partial_pct))
    return Action(
        action_type='PARTIAL_CLOSE',
        reason=(
            f'tp1_hit: price={current_price:.3f}, '
            f'r={current_r:.2f} >= {r_multiple:.1f}R'
        ),
        units=units_to_close,
        priority=4,
    )


def calc_breakeven_sl(trade: TradeState, config: dict) -> float:
    """
    建値+バッファのSL価格を計算する。
    TP1後に呼ばれる。

    buffer = sl_distance × breakeven_buffer_pct
    LONG:  sl = entry_price + buffer
    SHORT: sl = entry_price - buffer
    """
    buffer_pct = float(
        config['exit_rules']['tp1'].get('breakeven_buffer_pct', 0.1)
    )
    buffer = trade.sl_distance_usd * buffer_pct
    if trade.side == 'long':
        return trade.entry_price + buffer
    else:
        return trade.entry_price - buffer


# ------------------------------------------------------------------ #
#  Priority 5: Giveback Stop                                          #
# ------------------------------------------------------------------ #

def check_giveback_stop(
    trade: TradeState,
    current_unrealized_r: float,
    config: dict,
) -> Optional[Action]:
    """
    Priority 5: +2R到達後に+1Rまで戻り→全決済。

    ロジック:
      1. current_unrealized_r >= trigger_r(2.0) → peak_unrealized_r 更新
      2. peak_unrealized_r >= trigger_r AND current_unrealized_r <= exit_r(1.0)
         → 全決済

    「伸びたのに戻してゼロ」を防ぐ。

    Returns:
        Action(action_type="CLOSE_ALL") if triggered, None otherwise.
    """
    gs_cfg = config['exit_rules']['giveback_stop']
    if not gs_cfg.get('enabled', True):
        return None

    trigger_r = float(gs_cfg.get('trigger_r', 2.0))
    exit_r = float(gs_cfg.get('exit_r', 1.0))

    # ピーク更新（状態変更は呼び出し元 main.py が registry.update() で行う）
    # ここでは判定のみ
    armed = trade.peak_unrealized_r >= trigger_r

    if armed and current_unrealized_r <= exit_r:
        return Action(
            action_type='CLOSE_ALL',
            reason=(
                f'giveback_stop: peak={trade.peak_unrealized_r:.2f}R '
                f'gave back to {current_unrealized_r:.2f}R '
                f'(<= exit_r={exit_r}R)'
            ),
            priority=5,
        )
    return None


# ------------------------------------------------------------------ #
#  Priority 6: Trailing Stop（4H足確定時のみ）                       #
# ------------------------------------------------------------------ #

def check_trailing_stop(
    trade: TradeState,
    candles_4h: pd.DataFrame,
    config: dict,
    is_4h_close: bool = False,
) -> Optional[Action]:
    """
    Priority 6: 4H足確定時のスイングベーストレール。

    TP1後（phase=TP1_HIT または TRAILING）にのみ有効。
    「足確定時のみ更新」（ティック更新禁止）。

    ロジック:
      LONG:  最近5本の 4H 足ローの最小値 → 現在SLより上ならSL更新
      SHORT: 最近5本の 4H 足ハイの最大値 → 現在SLより下ならSL更新

    Returns:
        Action(action_type="MODIFY_SL") if SL should be updated, None otherwise.
    """
    if not is_4h_close:
        return None
    if trade.phase not in (TradePhase.TP1_HIT, TradePhase.TRAILING, TradePhase.OPEN):
        return None
    if candles_4h is None or len(candles_4h) < 3:
        return None

    lookback = min(5, len(candles_4h))
    recent = candles_4h.iloc[-lookback:]

    if trade.side == 'long':
        swing_low = float(recent['low'].min())
        if swing_low > trade.sl_price:
            return Action(
                action_type='MODIFY_SL',
                reason=(
                    f'trailing_stop: swing_low={swing_low:.3f} '
                    f'> current_sl={trade.sl_price:.3f}'
                ),
                new_sl=swing_low,
                new_sl_reason='4H足確定: 新スイングロー',
                priority=6,
            )
    else:  # short
        swing_high = float(recent['high'].max())
        if swing_high < trade.sl_price:
            return Action(
                action_type='MODIFY_SL',
                reason=(
                    f'trailing_stop: swing_high={swing_high:.3f} '
                    f'< current_sl={trade.sl_price:.3f}'
                ),
                new_sl=swing_high,
                new_sl_reason='4H足確定: 新スイングハイ',
                priority=6,
            )
    return None


# ------------------------------------------------------------------ #
#  Priority 7: Reversal Exit（4H足確定時のみ）                       #
# ------------------------------------------------------------------ #

def check_reversal_exit(
    trade: TradeState,
    candles_4h: pd.DataFrame,
    config: dict,
    is_4h_close: bool = False,
) -> Optional[Action]:
    """
    Priority 7: 4H足終値が直近構造ラインを逆方向に抜けたら全決済。

    教材準拠: 「日足レベルでローソクの色を戻したら撤退」

    実装:
      LONG:  最新確定バーの終値が、直前10本の安値最小値を割り込んだら撤退
      SHORT: 最新確定バーの終値が、直前10本の高値最大値を超えたら撤退

    Returns:
        Action(action_type="CLOSE_ALL") if reversal confirmed, None otherwise.
    """
    if not is_4h_close:
        return None
    if not config['exit_rules']['reversal_exit'].get('enabled', True):
        return None
    if candles_4h is None or len(candles_4h) < 11:
        return None

    prev_bars = candles_4h.iloc[-11:-1]   # 最新バーの直前10本
    latest_close = float(candles_4h.iloc[-1]['close'])

    if trade.side == 'long':
        structural_low = float(prev_bars['low'].min())
        if latest_close < structural_low:
            return Action(
                action_type='CLOSE_ALL',
                reason=(
                    f'reversal_exit: close={latest_close:.3f} '
                    f'< structural_low={structural_low:.3f}'
                ),
                priority=7,
            )
    else:  # short
        structural_high = float(prev_bars['high'].max())
        if latest_close > structural_high:
            return Action(
                action_type='CLOSE_ALL',
                reason=(
                    f'reversal_exit: close={latest_close:.3f} '
                    f'> structural_high={structural_high:.3f}'
                ),
                priority=7,
            )
    return None


# ------------------------------------------------------------------ #
#  Priority 8: Silver Time Stop（non_textbook）                      #
# ------------------------------------------------------------------ #

def check_silver_time_stop(
    trade: TradeState,
    now: datetime,
    current_r: float,
    config: dict,
) -> Optional[Action]:
    """
    Priority 8: XAG_USD専用 — 24h超えて+1R未達なら撤退。

    実績根拠: >24hで損失が重くなりやすい（non_textbook）。

    Returns:
        Action(action_type="CLOSE_ALL") if triggered, None otherwise.
    """
    if trade.instrument != 'XAG_USD':
        return None

    tf_cfg = config['exit_rules']['lockout']['time_filter'].get('XAG_USD', {})
    max_hours = float(tf_cfg.get('max_hold_hours_if_flat', 24))
    flat_threshold_r = float(tf_cfg.get('flat_threshold_r', 1.0))

    elapsed_h = (now - trade.entry_time).total_seconds() / 3600.0

    if elapsed_h >= max_hours and current_r < flat_threshold_r:
        return Action(
            action_type='CLOSE_ALL',
            reason=(
                f'silver_time_stop: {elapsed_h:.1f}h elapsed, '
                f'r={current_r:.2f} < {flat_threshold_r}'
            ),
            non_textbook=True,
            priority=8,
        )
    return None


# ------------------------------------------------------------------ #
#  Priority 9: Anti-patterns（veto層）                               #
# ------------------------------------------------------------------ #

def check_anti_patterns(
    action: Action,
    trade: TradeState,
    config: dict,
) -> tuple[bool, str]:
    """
    禁止パターン検知: 提案されたアクションをブロックするか判定する。

    ブロック条件:
      1. SL拡大（SLを不利な方向へ移動）
      2. 計画外ナンピン（ADD_UNITS タイプ）
      3. TP1後の追加半利（tp1_executed=True での PARTIAL_CLOSE）

    Returns:
        (blocked: bool, reason: str)
        blocked=True → main.py は VALIDATION_WARNING をログして実行をスキップ
    """
    ap_cfg = config['exit_rules']['anti_patterns']

    # 1. SL拡大禁止
    if (ap_cfg.get('no_manual_sl_widen', True)
            and action.action_type == 'MODIFY_SL'
            and action.new_sl is not None):
        if trade.side == 'long' and action.new_sl < trade.sl_price:
            return True, f'SL拡大禁止: new_sl={action.new_sl:.3f} < current_sl={trade.sl_price:.3f}'
        if trade.side == 'short' and action.new_sl > trade.sl_price:
            return True, f'SL拡大禁止: new_sl={action.new_sl:.3f} > current_sl={trade.sl_price:.3f}'

    # 2. 計画外ナンピン禁止
    if ap_cfg.get('no_unplanned_averaging', True) and action.action_type == 'ADD_UNITS':
        return True, '計画外ナンピン禁止'

    # 3. TP1後の追加半利禁止
    if (ap_cfg.get('no_repeated_partial', True)
            and action.action_type == 'PARTIAL_CLOSE'
            and trade.tp1_executed):
        return True, f'TP1後の重複部分利確禁止 (tp1_executed=True)'

    return False, ''


# ------------------------------------------------------------------ #
#  ボラティリティ・レジーム判定（non_textbook）                       #
# ------------------------------------------------------------------ #

def check_volatility_regime(
    candles_daily: pd.DataFrame,
    config: dict,
) -> dict:
    """
    日足ATR(14)が20日平均ATRの1.5倍以上なら「高ボラレジーム」と判定する。

    市況教材: 「戦争相場では柔軟に対応できるようにする」
              「必ずストップを置いて急な変動で無駄な損失を出さないようにする」

    Args:
        candles_daily: 日足OHLCデータ（30本以上推奨）。
                       列: open, high, low, close
        config:        config dict

    Returns:
        dict:
            is_high_vol (bool):         高ボラレジームか
            current_atr (float):        現在ATR(D,14)
            avg_atr (float):            20日平均ATR
            ratio (float):              current_atr / avg_atr
            effective_risk_jpy (float): 高ボラ時の調整リスク（JPY）
            effective_tp1_r (float):    高ボラ時のTP1 R倍率
            max_concurrent (int):       高ボラ時の同時建玉上限
            lockout_minutes (int):      高ボラ時のロックアウト分数
    """
    vr_cfg = config.get('exit_rules', {}).get('volatility_regime', {})
    base_risk = float(config['account']['max_loss_jpy'])
    base_tp1_r = float(config['exit_rules']['tp1']['r_multiple'])
    base_concurrent = int(config['account']['max_concurrent_trades'])
    base_lockout = int(
        config['exit_rules']['lockout']['short_term'].get('duration_minutes', 60)
    )

    if not vr_cfg.get('enabled', True) or candles_daily is None or len(candles_daily) < 20:
        return {
            'is_high_vol': False,
            'current_atr': 0.0,
            'avg_atr': 0.0,
            'ratio': 1.0,
            'effective_risk_jpy': base_risk,
            'effective_tp1_r': base_tp1_r,
            'max_concurrent': base_concurrent,
            'lockout_minutes': base_lockout,
        }

    # ATR(14) = 14日間の True Range の単純移動平均
    high = candles_daily['high']
    low = candles_daily['low']
    close = candles_daily['close']
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr14 = tr.rolling(14).mean()
    current_atr = float(atr14.iloc[-1])

    # 20日平均ATR（高ボラ判定のベースライン）
    avg_atr = float(atr14.iloc[-20:].mean())

    if avg_atr <= 0:
        return {
            'is_high_vol': False,
            'current_atr': current_atr,
            'avg_atr': avg_atr,
            'ratio': 1.0,
            'effective_risk_jpy': base_risk,
            'effective_tp1_r': base_tp1_r,
            'max_concurrent': base_concurrent,
            'lockout_minutes': base_lockout,
        }

    ratio = current_atr / avg_atr
    threshold = float(vr_cfg.get('atr_threshold_multiplier', 1.5))
    is_high_vol = ratio >= threshold

    if is_high_vol:
        adj = vr_cfg.get('high_vol_adjustments', {})
        risk_reduction = float(adj.get('risk_reduction_pct', 50)) / 100.0
        effective_risk = base_risk * (1.0 - risk_reduction)
        effective_tp1_r = float(adj.get('tp1_r_multiple', 0.7))
        max_concurrent = int(adj.get('max_concurrent_trades', 1))
        lockout_minutes = int(adj.get('lockout_short_term_minutes', 30))
    else:
        effective_risk = base_risk
        effective_tp1_r = base_tp1_r
        max_concurrent = base_concurrent
        lockout_minutes = base_lockout

    return {
        'is_high_vol': is_high_vol,
        'current_atr': round(current_atr, 4),
        'avg_atr': round(avg_atr, 4),
        'ratio': round(ratio, 3),
        'effective_risk_jpy': effective_risk,
        'effective_tp1_r': effective_tp1_r,
        'max_concurrent': max_concurrent,
        'lockout_minutes': lockout_minutes,
    }


# ------------------------------------------------------------------ #
#  初期SL検証（登録時のみ呼ばれる）                                   #
# ------------------------------------------------------------------ #

def check_initial_sl(trade: TradeState, config: dict) -> tuple[bool, str]:
    """
    SLの有効性を検証する（トレード登録時に一度だけ呼ばれる）。

    チェック:
      - sl_price が 0 以下 → 無効
      - sl_distance が min_sl_atr_ratio未満 → 「近すぎ」警告（拒否はしない）
      - sl_distance が max_sl_atr_ratio超 → 「遠すぎ」警告

    Returns:
        (valid: bool, warning_message: str)
    """
    if not trade.sl_price or trade.sl_price <= 0:
        return False, 'SL価格が設定されていません'

    if trade.sl_distance_usd <= 0:
        return False, 'SL距離が0以下です'

    # ATR比チェック（ATRが取得できない場合はスキップ）
    sl_cfg = config['exit_rules']['initial_sl']
    warning = ''

    min_ratio = float(sl_cfg.get('min_sl_atr_ratio', 0.5))
    max_ratio = float(sl_cfg.get('max_sl_atr_ratio', 3.0))

    # 警告のみ（拒否はしない）。ATR値は外部から渡す必要がある
    # → チェックは CLI の entry コマンドで ATR取得後に行う
    _ = min_ratio, max_ratio  # 将来の拡張用

    return True, warning
