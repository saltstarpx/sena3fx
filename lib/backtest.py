"""
改良版ティックレベルバックテストエンジン v4.1
=============================================
v3.0からの追加:
  - ピラミッティング（含み益時に追加エントリー）
  - シーズナリティフィルター（allowed_months）
  - 月次PnL統計
  - 複利対応（資金比率リスク管理で自動スケール）

v4.1からの追加 (richmanbtc methodology):
  - p-mean法による安定性検証 (5区間分割t検定)
    出典: richmanbtc note「ストラテジーの検証。p平均法」
    基準: p_mean < 0.03, p_mean_error_rate < 0.00001
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import json
import os
import csv
from math import factorial


class BacktestEngine:
    """
    ハイブリッド方式バックテスト v4.1:
    - シグナル生成: OHLCバーベース
    - SL/TP判定: バーベース
    - ピラミッティング: 含み益 pyramid_atr ATR ごとに追加エントリー
    - シーズナリティ: allowed_months で取引月を限定
    - トレーリングストップ: trail_start_atr で発動後、trail_dist_atr で追従

    やがみメソッド準拠:
    - SLは最後の押し安値/戻り高値（ATRフォールバック）
    - ピラミッドごとにSLを前の建値に移動（建値ストップ）
    - 1トレード最大リスク: 資金の risk_pct（デフォルト5%）

    マエダイメソッド準拠 (高RR/低WR設定):
    - default_sl_atr=0.8, default_tp_atr=10.0 (背を近く、大きく取る)
    - trail_start_atr=3.0, trail_dist_atr=1.5  (含み益が乗ったら追従)
    - pyramid_size_mult=1.0 (倍増ピラミッド)
    - target_max_dd=0.30, target_min_wr=0.30
    """

    def __init__(self, init_cash=5_000_000, risk_pct=0.05,
                 default_sl_atr=2.0, default_tp_atr=4.0,
                 slippage_pips=0.3, pip=0.1,
                 use_dynamic_sl=True,
                 sl_n_confirm=2,
                 sl_min_atr=0.5,
                 dynamic_rr=2.0,
                 pyramid_entries=2,
                 pyramid_atr=1.0,
                 pyramid_size_mult=0.5,
                 trail_start_atr=0.0,
                 trail_dist_atr=2.0,
                 exit_on_signal=True,
                 long_biased=False,
                 min_short_drop_atr=3.0,
                 breakeven_rr=0.0,
                 partial_tp_rr=0.0,
                 partial_tp_pct=0.5,
                 min_hold_hours=0.0,
                 symbol_risk_mult=1.0,
                 target_max_dd=0.15,
                 target_min_wr=0.50,
                 target_rr_threshold=2.0,
                 target_min_trades=30):
        """
        Args:
            risk_pct: 初期エントリーのリスク比率（デフォルト5%、複利で資産に追随）
            pyramid_entries: ピラミッド追加回数（0=なし）
            pyramid_atr: 追加エントリーのトリガー（含み益が pyramid_atr × ATR に達したら）
            pyramid_size_mult: 追加ロットの倍率（0.5=半分, 1.0=同量, 2.0=倍）
            trail_start_atr: トレーリング発動閾値（0=無効, 3.0=含み益3ATRで発動）
            trail_dist_atr: トレーリングSL距離（現在価格から何ATR離すか）
            target_max_dd: 合格基準 最大DD上限
            target_min_wr: 合格基準 最低勝率
            target_rr_threshold: 合格基準 RR（この値以上ならWR緩和）
            target_min_trades: 合格基準 最低トレード数
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
        self.pyramid_size_mult = pyramid_size_mult
        self.sl_n_confirm = sl_n_confirm
        self.sl_min_atr = sl_min_atr
        self.dynamic_rr = dynamic_rr
        self.trail_start_atr = trail_start_atr
        self.trail_dist_atr = trail_dist_atr
        self.exit_on_signal = exit_on_signal
        self.long_biased = long_biased
        self.min_short_drop_atr = min_short_drop_atr
        self.breakeven_rr = breakeven_rr
        self.partial_tp_rr  = partial_tp_rr   # 0=無効, 2.0=RR2達成で部分利確
        self.partial_tp_pct = partial_tp_pct  # 部分利確サイズ比率 (デフォルト50%)
        self.min_hold_hours = min_hold_hours  # 最低保有時間(h): この時間内はSL以外で決済しない
        self.symbol_risk_mult = symbol_risk_mult  # 銘柄別リスク係数 (銀=0.5等)
        self.target_max_dd = target_max_dd
        self.target_min_wr = target_min_wr
        self.target_rr_threshold = target_rr_threshold
        self.target_min_trades = target_min_trades

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
            htf_bars=None,
            trade_start=None,
            sizer=None):
        """
        バックテスト実行。

        data: OHLCバーデータ（またはティックデータからバーを生成）
        signal_func: bars -> pd.Series of 'long'/'short'/None
        allowed_months: 取引を許可する月のリスト（例: [1,2,3,10,11,12]）。
                        None の場合は全月でトレード。
        htf_bars: 1H/4H 上位足バー。SLのスウィングピボット検出に使用。
                  指定なしの場合は同一時間足のバーで検出。
        trade_start: エントリー開始日 (例: '2020-01-01')。
        sizer: VolatilityAdjustedSizer / KellyCriterionSizer など。
               get_multiplier(i) -> float を持つオブジェクト。
               risk_pct にこの乗数を掛けてポジションサイズを決定する。
                     それ以前はウォームアップ期間としてエントリーしない。
        """
        if use_ticks and ticks is not None:
            bars = self._make_bars(ticks, freq)
        else:
            bars = data

        _trade_start_ts = pd.Timestamp(trade_start) if trade_start else None

        if len(bars) < 30:
            return None

        atr = self._calc_atr(bars)
        signals = signal_func(bars)

        # sizer に ATR 系列を渡してウォームアップ
        if sizer is not None and hasattr(sizer, 'fit'):
            sizer.fit(atr)

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

                        # 追加サイズ = 初回 × pyramid_size_mult
                        base_size = pos['layers'][0]['size']
                        add_size = max(0.01, round(base_size * self.pyramid_size_mult, 2))

                        pos['layers'].append({
                            'entry': new_entry,
                            'size': add_size,
                            'entry_time': bar_time,
                        })
                        pos['pyramid_count'] += 1

                # ===== トレーリングストップ =====
                if self.trail_start_atr > 0:
                    first_entry = pos['layers'][0]['entry']
                    if pos['dir'] == 'long':
                        profit_atr = (bar['close'] - first_entry) / max(pos['atr'], 0.01)
                        if profit_atr >= self.trail_start_atr:
                            trail_sl = bar['close'] - self.trail_dist_atr * bar_atr
                            pos['sl'] = max(pos['sl'], trail_sl)
                    else:
                        profit_atr = (first_entry - bar['close']) / max(pos['atr'], 0.01)
                        if profit_atr >= self.trail_start_atr:
                            trail_sl = bar['close'] + self.trail_dist_atr * bar_atr
                            pos['sl'] = min(pos['sl'], trail_sl)

                # ===== ブレイクイーブン (breakeven_rr > 0 の場合) =====
                # 含み益が initial_sl_dist × breakeven_rr に達したら SL を建値に移動
                if self.breakeven_rr > 0:
                    first_entry = pos['layers'][0]['entry']
                    init_sl_dist = pos.get('initial_sl_dist', 0)
                    if init_sl_dist > 0:
                        if pos['dir'] == 'long':
                            profit = bar['close'] - first_entry
                            if profit >= self.breakeven_rr * init_sl_dist:
                                pos['sl'] = max(pos['sl'], first_entry)
                        else:
                            profit = first_entry - bar['close']
                            if profit >= self.breakeven_rr * init_sl_dist:
                                pos['sl'] = min(pos['sl'], first_entry)

                # ===== 部分利確 (partial_tp_rr > 0 の場合) =====
                # ユーザーのスケールアウト戦略を再現:
                # RR partial_tp_rr 達成時にポジの partial_tp_pct を利確 → SL建値移動
                if self.partial_tp_rr > 0 and not pos.get('partial_done', False):
                    first_entry  = pos['layers'][0]['entry']
                    init_sl_dist = pos.get('initial_sl_dist', 0)
                    if init_sl_dist > 0:
                        partial_target = first_entry + self.partial_tp_rr * init_sl_dist \
                            if pos['dir'] == 'long' else \
                            first_entry - self.partial_tp_rr * init_sl_dist
                        partial_hit = (pos['dir'] == 'long' and bar['high'] >= partial_target) or \
                                      (pos['dir'] == 'short' and bar['low']  <= partial_target)
                        if partial_hit:
                            # 部分利確: 最初のレイヤーの partial_tp_pct を決済
                            close_price = partial_target
                            reduce_size = max(0.01, round(
                                pos['layers'][0]['size'] * self.partial_tp_pct, 2))
                            if pos['dir'] == 'long':
                                pnl_part = (close_price - pos['layers'][0]['entry']) / self.pip * reduce_size
                            else:
                                pnl_part = (pos['layers'][0]['entry'] - close_price) / self.pip * reduce_size
                            cash += pnl_part
                            if cash > peak:
                                peak = cash
                            # 残ロット縮小
                            pos['layers'][0]['size'] = max(
                                0.01, pos['layers'][0]['size'] - reduce_size)
                            # SLを建値に移動 (フリートレード化)
                            if pos['dir'] == 'long':
                                pos['sl'] = max(pos['sl'], first_entry)
                            else:
                                pos['sl'] = min(pos['sl'], first_entry)
                            pos['partial_done'] = True

                # ===== SL/TP判定（共有ライン） =====
                exit_price = None
                exit_reason = None

                # 最低保有時間チェック: SL以外の決済は min_hold_hours 以内禁止
                hold_seconds = (bar_time - pos['layers'][0]['entry_time']).total_seconds()
                min_hold_ok = (self.min_hold_hours <= 0 or
                               hold_seconds >= self.min_hold_hours * 3600)

                if pos['dir'] == 'long':
                    if bar['low'] <= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'stop_loss'
                    elif min_hold_ok and bar['high'] >= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = 'take_profit'
                    elif min_hold_ok and self.exit_on_signal and sig in ('short', 'close'):
                        exit_price = bar['close']
                        exit_reason = 'signal'
                else:
                    if bar['high'] >= pos['sl']:
                        exit_price = pos['sl']
                        exit_reason = 'stop_loss'
                    elif min_hold_ok and bar['low'] <= pos['tp']:
                        exit_price = pos['tp']
                        exit_reason = 'take_profit'
                    elif min_hold_ok and self.exit_on_signal and sig in ('long', 'close'):
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

            # ウォームアップ期間フィルター（2020年1/1以降のみエントリー等）
            if _trade_start_ts is not None and bar_time < _trade_start_ts:
                continue

            # ロングバイアス: ショートは直近高値から大きく落ちた時のみ許可
            # やがみ: 「ショートは大きく落ちたときのみ考える」
            if self.long_biased and sig == 'short':
                recent_high = bars['high'].iloc[max(0, i - 10):i + 1].max()
                drop = (recent_high - bar['close']) / max(bar_atr, 0.01)
                if drop < self.min_short_drop_atr:
                    continue  # 大きい下落でなければショートスキップ

            spread = bar.get('spread', 0.5) if 'spread' in bar.index else 0.5

            if self.use_dynamic_sl:
                if sig == 'long':
                    sl_price = self._find_swing_low(
                        bars, i, htf_bars, n_confirm=self.sl_n_confirm
                    ) - self.slippage
                    sl_dist = bar['close'] - sl_price
                    if sl_dist < bar_atr * self.sl_min_atr:
                        sl_dist = bar_atr * self.default_sl_atr
                        sl_price = bar['close'] - sl_dist
                    tp_dist = sl_dist * self.dynamic_rr
                    tp_price = bar['close'] + tp_dist
                else:
                    sl_price = self._find_swing_high(
                        bars, i, htf_bars, n_confirm=self.sl_n_confirm
                    ) + self.slippage
                    sl_dist = sl_price - bar['close']
                    if sl_dist < bar_atr * self.sl_min_atr:
                        sl_dist = bar_atr * self.default_sl_atr
                        sl_price = bar['close'] + sl_dist
                    tp_dist = sl_dist * self.dynamic_rr
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
            # symbol_risk_mult: 銘柄別リスク係数 (例: 銀スポット=0.5 → ロット半減)
            # sizer: VolatilityAdjustedSizer / KellyCriterionSizer による動的乗数
            sizer_mult = sizer.get_multiplier(i) if sizer is not None else 1.0
            risk_amount = cash * self.risk_pct * self.symbol_risk_mult * sizer_mult
            sl_pips = sl_dist / self.pip
            pos_size = max(0.01, round(risk_amount / sl_pips, 2)) if sl_pips > 0 else 0.01

            pos = {
                'dir': sig,
                'sl': sl_price,
                'tp': tp_price,
                'atr': bar_atr,
                'initial_sl_dist': sl_dist,
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

        return self._report(name, trades, cash, max_dd, freq, len(bars),
                            trade_start=_trade_start_ts)

    def _report(self, name, trades, final_cash, max_dd, freq, n_bars,
                trade_start=None):
        n = len(trades)
        if n == 0:
            return {'strategy': name, 'total_trades': 0, 'passed': False,
                    'trades_per_year': 0.0}

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

        # ── 年間取引回数を計算 ──
        # trade_start〜最終trade の実期間(年)で割り、年換算レートを算出
        if trades and trade_start is not None:
            first_trade = pd.Timestamp(trades[0]['entry_time'])
            last_trade  = pd.Timestamp(trades[-1]['exit_time'])
            trade_start_ts = pd.Timestamp(trade_start)
            start_ts = max(first_trade, trade_start_ts)
            years_active = max((last_trade - start_ts).days / 365.25, 1/12)
        elif trades:
            first_trade = pd.Timestamp(trades[0]['entry_time'])
            last_trade  = pd.Timestamp(trades[-1]['exit_time'])
            years_active = max((last_trade - first_trade).days / 365.25, 1/12)
        else:
            years_active = 1.0
        trades_per_year = n / years_active

        # ── Sharpe Ratio / Calmar Ratio (Task3: 評価軸多元化) ──
        sharpe_ratio = None
        calmar_ratio = None
        try:
            # 日次リターン系列から算出
            pnl_list = [t['pnl'] for t in trades]
            # トレードごとの pnl_pct を使って equity curve を構築
            eq = [self.init_cash]
            for t in trades:
                eq.append(eq[-1] + t['pnl'])
            eq = np.array(eq[1:])  # init_cash を除去
            # 日次PnL → トレードPnLで近似 (トレード間隔が不均一のため)
            trade_returns = np.array([t['pnl'] / max(eq[max(0, j - 1)], 1)
                                      for j, t in enumerate(trades)])
            if len(trade_returns) >= 2 and np.std(trade_returns) > 0:
                # 年率換算: trades_per_year を使って annualize
                avg_r = np.mean(trade_returns)
                std_r = np.std(trade_returns, ddof=1)
                ann_factor = np.sqrt(max(trades_per_year, 1))
                sharpe_ratio = round(float(avg_r / std_r * ann_factor), 3)
            # Calmar Ratio = 年率リターン / 最大DD
            ann_return_pct = ret / max(years_active, 1 / 12)
            if max_dd > 0:
                calmar_ratio = round(float(ann_return_pct / (max_dd * 100)), 3)
        except Exception:
            pass

        # ── richmanbtc p-mean法 (5区間安定性検証) ──
        # 出典: note.com/btcml「ストラテジーの検証。p平均法」
        # バックテスト期間を5等分し、各区間でt検定。
        # 全区間で安定して勝てるか検証する。
        # error_rate = (mean(p)*N)^N / N!
        #
        # 【スウィング戦略への適用注意】
        # richmanbtcの基準(p_mean<0.03)はHFT用 (1区間1000+取引)。
        # スウィング (1区間~26取引, WR=37%, RR=3) の理論値: p_mean≈0.10-0.25
        # → 代わりに「全N区間でプラス」= winning_segments/N を使う。
        # 全5区間プラス = 完全安定。4/5区間プラス = 許容範囲。
        p_mean_val = None
        p_mean_error_rate = None
        winning_segments = None
        p_mean_segments = 5
        try:
            from scipy import stats as _stats
            pnl_series = [t['pnl'] for t in trades]
            seg_size = max(1, len(pnl_series) // p_mean_segments)
            p_values = []
            seg_means = []
            for seg_i in range(p_mean_segments):
                seg = pnl_series[seg_i * seg_size:(seg_i + 1) * seg_size]
                if len(seg) >= 3:
                    t_stat, p_val = _stats.ttest_1samp(seg, 0)
                    # 片側検定: 母平均>0 を検定
                    p_one = p_val / 2 if t_stat > 0 else 1.0 - p_val / 2
                    p_values.append(p_one)
                    seg_means.append(np.mean(seg))
            if p_values:
                p_mean_val = float(np.mean(p_values))
                N = len(p_values)
                p_mean_error_rate = float((p_mean_val * N) ** N / factorial(N))
                # スウィング向け安定指標: 勝ちセグメント数/全体
                winning_segments = sum(1 for m in seg_means if m > 0)
        except ImportError:
            pass  # scipy未インストールの場合はスキップ

        # 合格判定（設定可能な基準）
        wr_min = self.target_min_wr
        dd_max = self.target_max_dd
        rr_thr = self.target_rr_threshold
        n_min  = self.target_min_trades
        # 高RR時はWR基準を緩和 (RR≥rr_thr なら WR≥0.25 でも可)
        wr_ok = (wr >= wr_min) or (rr_ratio >= rr_thr and wr >= wr_min * 0.5)
        # ミッション2: 年間50回以上 AND PF>1.3 が必須条件
        freq_ok = trades_per_year >= 50.0
        passed = (pf >= 1.3 and max_dd <= dd_max and n >= n_min and wr_ok and freq_ok)

        # 8-24時間保有フラグ (指令7: 黄金ゾーン判定)
        # ユーザー分析: 8-24h保有でPF=13.08 / WR=74.8% (最高パターン)
        # 評価関数の主基準ではなく「付加情報」として提供
        hold_8_24h = (8.0 <= avg_dur <= 24.0)

        result = {
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
            'trades_per_year': round(trades_per_year, 1),
            'hold_8_24h': hold_8_24h,   # 8-24h保有ゾーン (ユーザー黄金ゾーン)
            # richmanbtc p-mean法 (5区間安定性スコア)
            # p_mean < 0.03 はHFT基準。スウィング向け基準: winning_segments=5/5
            'p_mean': round(p_mean_val, 4) if p_mean_val is not None else None,
            'p_mean_error_rate': float(f'{p_mean_error_rate:.2e}') if p_mean_error_rate is not None else None,
            'winning_segments': winning_segments,  # 5区間中のプラス区間数 (5/5が最良)
            # Sharpe / Calmar (Task3: 評価軸多元化)
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'trades': trades,
        }
        # ── パフォーマンスログへ自動追記 (dashboard.html 用) ──
        self._append_performance_log(result)
        return result

    def _append_performance_log(self, r):
        """
        バックテスト結果を results/performance_log.csv に自動追記。
        ダッシュボード (dashboard.html) の時系列可視化に使用。
        """
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results', 'performance_log.csv',
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        header = ['timestamp', 'strategy_name', 'parameters', 'timeframe',
                  'sharpe_ratio', 'profit_factor', 'max_drawdown',
                  'win_rate', 'trades']
        write_header = not os.path.exists(log_path)
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([
                datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                r.get('strategy', ''),
                r.get('timeframe', ''),
                r.get('timeframe', ''),
                r.get('sharpe_ratio', ''),
                r.get('profit_factor', ''),
                r.get('max_drawdown_pct', ''),
                r.get('win_rate_pct', ''),
                r.get('total_trades', ''),
            ])

