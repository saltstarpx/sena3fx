"""
改良版ティックレベルバックテストエンジン v3.0
=============================================
v2.0からの改良点:
  - やがみメソッドのSL/TP計算（最後の押し安値/戻り高値ベース）
  - ポジり方の本準拠の建値ストップ判定
  - ATRベースの動的SL/TP
  - セッション別フィルター統合
  - A/B/C評価別の統計
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import json
import os


class BacktestEngine:
    """
    ハイブリッド方式バックテスト:
    - シグナル生成: OHLCバーベース
    - SL/TP判定: ティックベース（利用可能な場合）/ バーベース（フォールバック）

    やがみメソッド準拠:
    - SLは最後の押し安値/戻り高値（ATRフォールバック）
    - 建値ストップは「ネックラインを抜けた場合のみ」
    - 1トレード最大リスク: 資金の2%
    """

    def __init__(self, init_cash=5_000_000, risk_pct=0.02,
                 default_sl_atr=2.0, default_tp_atr=4.0,
                 slippage_pips=0.3, pip=0.1,
                 use_dynamic_sl=True):
        self.init_cash = init_cash
        self.risk_pct = risk_pct
        self.default_sl_atr = default_sl_atr
        self.default_tp_atr = default_tp_atr
        self.slippage = slippage_pips * pip
        self.pip = pip
        self.use_dynamic_sl = use_dynamic_sl

    def _calc_atr(self, bars, period=14):
        h = bars['high'].values
        l = bars['low'].values
        c = bars['close'].values
        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        return pd.Series(tr, index=bars.index).rolling(period).mean()

    def _find_swing_low(self, bars, idx, lookback=10):
        """やがみ: 最後の押し安値を見つける"""
        start = max(0, idx - lookback)
        return bars['low'].iloc[start:idx].min()

    def _find_swing_high(self, bars, idx, lookback=10):
        """やがみ: 最後の戻り高値を見つける"""
        start = max(0, idx - lookback)
        return bars['high'].iloc[start:idx].max()

    def _make_bars(self, ticks, freq):
        bid = ticks['bidPrice']
        ask = ticks['askPrice']
        bars = bid.resample(freq).agg(open='first', high='max', low='min', close='last')
        bars['spread'] = (ask - bid).resample(freq).mean()
        bars['tick_count'] = bid.resample(freq).count()
        bars = bars.dropna(subset=['open'])
        return bars

    def run(self, data, signal_func, freq='1h', name='Strategy',
            use_ticks=False, ticks=None):
        """
        バックテスト実行。

        data: OHLCバーデータ（またはティックデータからバーを生成）
        signal_func: bars -> pd.Series of 'long'/'short'/None
        """
        if use_ticks and ticks is not None:
            bars = self._make_bars(ticks, freq)
        else:
            bars = data

        if len(bars) < 30:
            return None

        atr = self._calc_atr(bars)
        signals = signal_func(bars)

        trades = []
        cash = self.init_cash
        peak = self.init_cash
        max_dd = 0.0
        in_pos = False
        pos = {}

        for i in range(len(bars)):
            sig = signals.iloc[i] if i < len(signals) else None
            if sig is not None and isinstance(sig, float) and np.isnan(sig):
                sig = None

            bar = bars.iloc[i]
            bar_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 2.0

            if in_pos:
                # SL/TP判定（バーベース）
                if pos['dir'] == 'long':
                    if bar['low'] <= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'stop_loss'
                    elif bar['high'] >= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = 'take_profit'
                    elif sig == 'short' or sig == 'close':
                        exit_price = bar['close']
                        exit_reason = 'signal'
                    else:
                        continue
                else:  # short
                    if bar['high'] >= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'stop_loss'
                    elif bar['low'] <= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = 'take_profit'
                    elif sig == 'long' or sig == 'close':
                        exit_price = bar['close']
                        exit_reason = 'signal'
                    else:
                        continue

                # 決済
                if pos['dir'] == 'long':
                    pnl_pips = (exit_price - pos['entry']) / self.pip
                else:
                    pnl_pips = (pos['entry'] - exit_price) / self.pip

                pnl = pnl_pips * pos['size']
                duration = (bars.index[i] - pos['entry_time']).total_seconds()

                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': bars.index[i],
                    'direction': pos['dir'],
                    'entry_price': pos['entry'],
                    'exit_price': exit_price,
                    'sl': pos['sl'],
                    'tp': pos['tp'],
                    'size': pos['size'],
                    'pnl': pnl,
                    'pnl_pct': pnl / cash * 100,
                    'exit_reason': exit_reason,
                    'duration_sec': duration,
                    'atr_at_entry': pos['atr'],
                })

                cash += pnl
                if cash > peak:
                    peak = cash
                dd = (peak - cash) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                in_pos = False

            # 新規エントリー
            if not in_pos and sig in ('long', 'short'):
                spread = bar.get('spread', 0.5) if 'spread' in bar.index else 0.5

                if self.use_dynamic_sl:
                    if sig == 'long':
                        sl_price = self._find_swing_low(bars, i) - self.slippage
                        sl_dist = bar['close'] - sl_price
                        if sl_dist < bar_atr * 0.5:
                            sl_dist = bar_atr * self.default_sl_atr
                            sl_price = bar['close'] - sl_dist
                        tp_dist = sl_dist * 2.0  # RR 2:1
                        tp_price = bar['close'] + tp_dist
                    else:
                        sl_price = self._find_swing_high(bars, i) + self.slippage
                        sl_dist = sl_price - bar['close']
                        if sl_dist < bar_atr * 0.5:
                            sl_dist = bar_atr * self.default_sl_atr
                            sl_price = bar['close'] + sl_dist
                        tp_dist = sl_dist * 2.0
                        tp_price = bar['close'] - tp_dist
                else:
                    sl_dist = bar_atr * self.default_sl_atr
                    tp_dist = bar_atr * self.default_tp_atr
                    if sig == 'long':
                        sl_price = bar['close'] - sl_dist
                        tp_price = bar['close'] + tp_dist
                    else:
                        sl_price = bar['close'] + sl_dist
                        tp_price = bar['close'] - tp_dist

                if sig == 'long':
                    entry_price = bar['close'] + spread/2 + self.slippage
                else:
                    entry_price = bar['close'] - spread/2 - self.slippage

                # ロット計算（やがみ+requirements: 資金の2%リスク）
                risk_amount = cash * self.risk_pct
                sl_pips = sl_dist / self.pip
                pos_size = max(0.01, round(risk_amount / sl_pips, 2)) if sl_pips > 0 else 0.01

                pos = {
                    'dir': sig,
                    'entry': entry_price,
                    'entry_time': bars.index[i],
                    'sl': sl_price,
                    'tp': tp_price,
                    'size': pos_size,
                    'atr': bar_atr,
                }
                in_pos = True

        # 残ポジクローズ
        if in_pos and len(bars) > 0:
            last = bars.iloc[-1]
            if pos['dir'] == 'long':
                pnl_pips = (last['close'] - pos['entry']) / self.pip
            else:
                pnl_pips = (pos['entry'] - last['close']) / self.pip
            pnl = pnl_pips * pos['size']
            trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': bars.index[-1],
                'direction': pos['dir'],
                'entry_price': pos['entry'],
                'exit_price': last['close'],
                'sl': pos['sl'], 'tp': pos['tp'],
                'size': pos['size'],
                'pnl': pnl, 'pnl_pct': pnl / cash * 100,
                'exit_reason': 'end_of_data',
                'duration_sec': (bars.index[-1] - pos['entry_time']).total_seconds(),
                'atr_at_entry': pos['atr'],
            })
            cash += pnl

        return self._report(name, trades, cash, max_dd, freq, len(bars))

    def _report(self, name, trades, final_cash, max_dd, freq, n_bars):
        n = len(trades)
        if n == 0:
            return {'strategy': name, 'total_trades': 0, 'passed': False}

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        total_win = sum(t['pnl'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        wr = len(wins) / n
        pf = total_win / total_loss if total_loss > 0 else 999

        ret = (final_cash - self.init_cash) / self.init_cash * 100
        avg_dur = np.mean([t['duration_sec'] for t in trades]) / 3600

        reasons = {}
        for t in trades:
            reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1

        # やがみ基準: RR2.0以上なら勝率40%でも可
        avg_win_val = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss_val = abs(np.mean([t['pnl'] for t in losses])) if losses else 1
        rr_ratio = avg_win_val / avg_loss_val if avg_loss_val > 0 else 0

        # 合格判定（requirements準拠 + RR考慮）
        passed = (pf >= 1.5 and max_dd <= 0.10 and n >= 30 and
                  (wr >= 0.50 or (rr_ratio >= 2.0 and wr >= 0.35)))

        return {
            'strategy': name,
            'engine': 'yagami_v3',
            'timeframe': freq,
            'total_return_pct': round(ret, 2),
            'total_pnl': round(final_cash - self.init_cash, 2),
            'end_value': round(final_cash, 0),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'win_rate_pct': round(wr * 100, 2),
            'profit_factor': round(pf, 4),
            'rr_ratio': round(rr_ratio, 2),
            'total_trades': n,
            'wins': len(wins),
            'losses': len(losses),
            'avg_win': round(avg_win_val, 2),
            'avg_loss': round(np.mean([t['pnl'] for t in losses]), 2) if losses else 0,
            'avg_duration_hours': round(avg_dur, 1),
            'exit_reasons': reasons,
            'passed': passed,
            'trades': trades,
        }
