"""
改良版ティックレベルバックテストエンジン v4.0
=============================================
v3.0からの追加:
  - ピラミッティング（含み益時に追加エントリー）
  - シーズナリティフィルター（allowed_months）
  - 月次PnL統計
  - 複利対応（資金比率リスク管理で自動スケール）
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import json
import os


class BacktestEngine:
    """
    ハイブリッド方式バックテスト v4.0:
    - シグナル生成: OHLCバーベース
    - SL/TP判定: バーベース
    - ピラミッティング: 含み益 pyramid_atr ATR ごとに追加エントリー
    - シーズナリティ: allowed_months で取引月を限定

    やがみメソッド準拠:
    - SLは最後の押し安値/戻り高値（ATRフォールバック）
    - ピラミッドごとにSLを前の建値に移動（建値ストップ）
    - 1トレード最大リスク: 資金の risk_pct（デフォルト5%）
    """

    def __init__(self, init_cash=5_000_000, risk_pct=0.05,
                 default_sl_atr=2.0, default_tp_atr=4.0,
                 slippage_pips=0.3, pip=0.1,
                 use_dynamic_sl=True,
                 pyramid_entries=2,
                 pyramid_atr=1.0):
        """
        Args:
            risk_pct: 初期エントリーのリスク比率（デフォルト5%、複利で資産に追随）
            pyramid_entries: ピラミッド追加回数（0=なし、2=2回追加）
            pyramid_atr: 追加エントリーのトリガー（含み益が pyramid_atr * ATR に達したら）
        """
        self.init_cash = init_cash
        self.risk_pct = risk_pct
        self.default_sl_atr = default_sl_atr
        self.default_tp_atr = default_tp_atr
        self.slippage = slippage_pips * pip
        self.pip = pip
        self.use_dynamic_sl = use_dynamic_sl
        self.pyramid_entries = pyramid_entries
        self.pyramid_atr = pyramid_atr

    def _calc_atr(self, bars, period=14):
        h = bars['high'].values
        l = bars['low'].values
        c = bars['close'].values
        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        return pd.Series(tr, index=bars.index).rolling(period).mean()

    def _find_swing_low(self, bars, idx, htf_bars=None, n_confirm=2):
        """
        真のスウィングロー: 両側 n_confirm 本より安値が高い局所最小値。
        やがみ: 「SLは1H/4Hの直近安値に置く」

        htf_bars: 1H/4H上位足バー（指定があれば優先使用）
        n_confirm: 左右の確認本数（デフォルト2）
        """
        search_bars = htf_bars if htf_bars is not None else bars
        current_time = bars.index[idx]

        # current_time 以前のバーのみ対象
        mask = search_bars.index <= current_time
        past = search_bars.loc[mask]

        if len(past) < n_confirm * 2 + 3:
            # データ不足: 直近20本の最安値にフォールバック
            start = max(0, idx - 20)
            return bars['low'].iloc[start:idx + 1].min()

        lows = past['low'].values
        n = len(lows)

        # 最も直近のスウィングローを後ろから探索
        for i in range(n - n_confirm - 1, n_confirm - 1, -1):
            lo = lows[i]
            if (all(lows[i - j] > lo for j in range(1, n_confirm + 1)) and
                    all(lows[i + j] > lo for j in range(1, n_confirm + 1))):
                return lo

        # ピボットが見つからない場合: 直近20本の最安値
        return past['low'].iloc[-20:].min()

    def _find_swing_high(self, bars, idx, htf_bars=None, n_confirm=2):
        """
        真のスウィングハイ: 両側 n_confirm 本より高値が低い局所最大値。
        やがみ: 「SLは1H/4Hの直近高値に置く」
        """
        search_bars = htf_bars if htf_bars is not None else bars
        current_time = bars.index[idx]

        mask = search_bars.index <= current_time
        past = search_bars.loc[mask]

        if len(past) < n_confirm * 2 + 3:
            start = max(0, idx - 20)
            return bars['high'].iloc[start:idx + 1].max()

        highs = past['high'].values
        n = len(highs)

        for i in range(n - n_confirm - 1, n_confirm - 1, -1):
            hi = highs[i]
            if (all(highs[i - j] < hi for j in range(1, n_confirm + 1)) and
                    all(highs[i + j] < hi for j in range(1, n_confirm + 1))):
                return hi

        return past['high'].iloc[-20:].max()

    def _make_bars(self, ticks, freq):
        bid = ticks['bidPrice']
        ask = ticks['askPrice']
        bars = bid.resample(freq).agg(open='first', high='max', low='min', close='last')
        bars['spread'] = (ask - bid).resample(freq).mean()
        bars['tick_count'] = bid.resample(freq).count()
        bars = bars.dropna(subset=['open'])
        return bars

    def run(self, data, signal_func, freq='1h', name='Strategy',
            use_ticks=False, ticks=None,
            allowed_months=None,
            htf_bars=None):
        """
        バックテスト実行。

        data: OHLCバーデータ（またはティックデータからバーを生成）
        signal_func: bars -> pd.Series of 'long'/'short'/None
        allowed_months: 取引を許可する月のリスト（例: [1,2,3,10,11,12]）。
                        None の場合は全月でトレード。
        htf_bars: 1H/4H 上位足バー。SLのスウィングピボット検出に使用。
                  指定なしの場合は同一時間足のバーで検出。
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
        pos = {}  # {'dir', 'sl', 'tp', 'atr', 'layers': [{'entry','size','entry_time'}], 'pyramid_count'}

        for i in range(len(bars)):
            sig = signals.iloc[i] if i < len(signals) else None
            if sig is not None and isinstance(sig, float) and np.isnan(sig):
                sig = None

            bar = bars.iloc[i]
            bar_time = bars.index[i]
            bar_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 2.0

            if in_pos:
                # ===== ピラミッティング判定 =====
                if self.pyramid_entries > 0 and pos['pyramid_count'] < self.pyramid_entries:
                    trigger_dist = pos['atr'] * self.pyramid_atr * (pos['pyramid_count'] + 1)
                    first_entry = pos['layers'][0]['entry']

                    pyramid_triggered = (
                        (pos['dir'] == 'long' and bar['close'] >= first_entry + trigger_dist)
                        or (pos['dir'] == 'short' and bar['close'] <= first_entry - trigger_dist)
                    )

                    if pyramid_triggered:
                        # 追加エントリー（前の建値がSLになる → ブレークイーブン管理）
                        prev_entry = pos['layers'][-1]['entry']
                        if pos['dir'] == 'long':
                            new_entry = bar['close'] + self.slippage
                            # SLを前の建値に移動
                            pos['sl'] = max(pos['sl'], prev_entry)
                        else:
                            new_entry = bar['close'] - self.slippage
                            pos['sl'] = min(pos['sl'], prev_entry)

                        # 追加サイズ = 初回の50%（スケールイン）
                        base_size = pos['layers'][0]['size']
                        add_size = max(0.01, round(base_size * 0.5, 2))

                        pos['layers'].append({
                            'entry': new_entry,
                            'size': add_size,
                            'entry_time': bar_time,
                        })
                        pos['pyramid_count'] += 1

                # ===== SL/TP判定（共有ライン） =====
                exit_price = None
                exit_reason = None

                if pos['dir'] == 'long':
                    if bar['low'] <= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'stop_loss'
                    elif bar['high'] >= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = 'take_profit'
                    elif sig in ('short', 'close'):
                        exit_price = bar['close']
                        exit_reason = 'signal'
                else:
                    if bar['high'] >= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'stop_loss'
                    elif bar['low'] <= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = 'take_profit'
                    elif sig in ('long', 'close'):
                        exit_price = bar['close']
                        exit_reason = 'signal'

                if exit_price is None:
                    continue

                # ===== 全レイヤー決済 =====
                total_pnl = 0.0
                total_size = sum(l['size'] for l in pos['layers'])
                weighted_entry = sum(l['entry'] * l['size'] for l in pos['layers']) / total_size

                for layer in pos['layers']:
                    if pos['dir'] == 'long':
                        pnl_pips = (exit_price - layer['entry']) / self.pip
                    else:
                        pnl_pips = (layer['entry'] - exit_price) / self.pip
                    total_pnl += pnl_pips * layer['size']

                duration = (bar_time - pos['layers'][0]['entry_time']).total_seconds()

                trades.append({
                    'entry_time': pos['layers'][0]['entry_time'],
                    'exit_time': bar_time,
                    'direction': pos['dir'],
                    'entry_price': round(weighted_entry, 5),
                    'exit_price': exit_price,
                    'sl': pos['sl'],
                    'tp': pos['tp'],
                    'size': total_size,
                    'pyramid_layers': pos['pyramid_count'] + 1,
                    'pnl': total_pnl,
                    'pnl_pct': total_pnl / cash * 100,
                    'exit_reason': exit_reason,
                    'duration_sec': duration,
                    'atr_at_entry': pos['atr'],
                })

                cash += total_pnl
                if cash > peak:
                    peak = cash
                dd = (peak - cash) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                in_pos = False

            # ===== 新規エントリー =====
            if in_pos:
                continue
            if sig not in ('long', 'short'):
                continue

            # シーズナリティフィルター
            if allowed_months is not None and bar_time.month not in allowed_months:
                continue

            spread = bar.get('spread', 0.5) if 'spread' in bar.index else 0.5

            if self.use_dynamic_sl:
                if sig == 'long':
                    sl_price = self._find_swing_low(bars, i, htf_bars) - self.slippage
                    sl_dist = bar['close'] - sl_price
                    if sl_dist < bar_atr * 0.5:
                        sl_dist = bar_atr * self.default_sl_atr
                        sl_price = bar['close'] - sl_dist
                    tp_dist = sl_dist * 2.0
                    tp_price = bar['close'] + tp_dist
                else:
                    sl_price = self._find_swing_high(bars, i, htf_bars) + self.slippage
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
                entry_price = bar['close'] + spread / 2 + self.slippage
            else:
                entry_price = bar['close'] - spread / 2 - self.slippage

            # 複利ロット計算（現在の資金 × risk_pct がリスク額）
            risk_amount = cash * self.risk_pct
            sl_pips = sl_dist / self.pip
            pos_size = max(0.01, round(risk_amount / sl_pips, 2)) if sl_pips > 0 else 0.01

            pos = {
                'dir': sig,
                'sl': sl_price,
                'tp': tp_price,
                'atr': bar_atr,
                'layers': [{'entry': entry_price, 'size': pos_size, 'entry_time': bar_time}],
                'pyramid_count': 0,
            }
            in_pos = True

        # 残ポジクローズ（全レイヤー）
        if in_pos and len(bars) > 0:
            last = bars.iloc[-1]
            total_pnl = 0.0
            total_size = sum(l['size'] for l in pos['layers'])
            weighted_entry = sum(l['entry'] * l['size'] for l in pos['layers']) / total_size

            for layer in pos['layers']:
                if pos['dir'] == 'long':
                    pnl_pips = (last['close'] - layer['entry']) / self.pip
                else:
                    pnl_pips = (layer['entry'] - last['close']) / self.pip
                total_pnl += pnl_pips * layer['size']

            trades.append({
                'entry_time': pos['layers'][0]['entry_time'],
                'exit_time': bars.index[-1],
                'direction': pos['dir'],
                'entry_price': round(weighted_entry, 5),
                'exit_price': last['close'],
                'sl': pos['sl'], 'tp': pos['tp'],
                'size': total_size,
                'pyramid_layers': pos['pyramid_count'] + 1,
                'pnl': total_pnl, 'pnl_pct': total_pnl / cash * 100,
                'exit_reason': 'end_of_data',
                'duration_sec': (bars.index[-1] - pos['layers'][0]['entry_time']).total_seconds(),
                'atr_at_entry': pos['atr'],
            })
            cash += total_pnl

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

        avg_win_val = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss_val = abs(np.mean([t['pnl'] for t in losses])) if losses else 1
        rr_ratio = avg_win_val / avg_loss_val if avg_loss_val > 0 else 0

        # 月次統計
        monthly_pnl = {}
        for t in trades:
            et = t['exit_time']
            key = f"{et.year}-{et.month:02d}" if hasattr(et, 'year') else str(et)[:7]
            monthly_pnl[key] = monthly_pnl.get(key, 0) + t['pnl']

        # ピラミッド統計
        pyramid_trades = [t for t in trades if t.get('pyramid_layers', 1) > 1]
        pyramid_rate = len(pyramid_trades) / n * 100 if n > 0 else 0

        # 合格判定（やがみ基準 + RR考慮）
        passed = (pf >= 1.5 and max_dd <= 0.15 and n >= 30 and
                  (wr >= 0.50 or (rr_ratio >= 2.0 and wr >= 0.35)))

        return {
            'strategy': name,
            'engine': 'yagami_v4',
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
            'monthly_pnl': monthly_pnl,
            'pyramid_trade_rate_pct': round(pyramid_rate, 1),
            'passed': passed,
            'trades': trades,
        }
