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
import csv


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
                 target_min_trades=30,
                 # ── Safety Valve (Proposal A) ──
                 min_sl_atr_mult=0.3,
                 min_sl_price_pct=0.001,
                 max_pos_size=10000.0,
                 max_notional_pct=5.0,
                 skip_log_path=None):
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
        # Safety Valve
        self.min_sl_atr_mult = min_sl_atr_mult
        self.min_sl_price_pct = min_sl_price_pct
        self.max_pos_size = max_pos_size
        self.max_notional_pct = max_notional_pct
        self.skip_log_path = skip_log_path

    def _log_skip(self, bar_time, direction, reason):
        """
        Safety Valve でスキップされたエントリーを JSONL に記録。
        skip_log_path が None の場合は results/skip_log.jsonl に書き込む。
        """
        path = self.skip_log_path
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'results', 'skip_log.jsonl',
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        record = {
            'ts': str(bar_time),
            'dir': direction,
            'reason': reason,
        }
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

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
            sizer=None,
            disable_atr_sl=False):
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
        disable_atr_sl: Trueの場合、ダイナミックATR-SLを無効化し、
                        固定SL (default_sl_atr×ATR) を使用する。
                        VolatilityAdjustedSizerとの重複解消に使用。
                     それ以前はウォームアップ期間としてエントリーしない。
        """
        if use_ticks and ticks is not None:
            bars = self._make_bars(ticks, freq)
        else:
            bars = data

        # disable_atr_sl: ATR-SLを無効化 → 固定SL(default_sl_atr×ATR)を使用
        # VolatilityAdjustedSizer使用時にスウィングSLとの重複を解消する
        _use_dynamic_sl_orig = self.use_dynamic_sl
        if disable_atr_sl:
            self.use_dynamic_sl = False

        _trade_start_ts = pd.Timestamp(trade_start) if trade_start else None

        if len(bars) < 30:
            return None

        atr = self._calc_atr(bars)
        signals = signal_func(bars)

        # sizer に ATR 系列を渡してウォームアップ
        if sizer is not None and hasattr(sizer, 'fit'):
            sizer.fit(atr)

        # LivermorePyramidingSizer 検出: on_bar() を持つ場合はリバモア式ピラミッド使用
        _use_livermore = sizer is not None and hasattr(sizer, 'on_bar')

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
                # ===== リバモア式ピラミッティング判定 (LivermorePyramidingSizer) =====
                if _use_livermore:
                    add_size = sizer.on_bar(pos['dir'], bar['close'])
                    if add_size > 0:
                        # 追加エントリー
                        pos['layers'].append({
                            'entry': bar['close'],
                            'size': add_size,
                            'entry_time': bar_time
                        })
                        # リバモア式ではSLを全建玉共通のブレイクイーブン等に動かす場合があるが、
                        # ここではエンジンの基本機能に任せる。

                # ===== SL/TP 判定 (全レイヤー一括) =====
                exit_price = None
                reason = ""
                if pos['dir'] == 'long':
                    if bar['low'] <= pos['sl']:
                        exit_price = pos['sl']
                        reason = "SL"
                    elif bar['high'] >= pos['tp']:
                        exit_price = pos['tp']
                        reason = "TP"
                    elif self.exit_on_signal and sig == 'short':
                        exit_price = bar['close']
                        reason = "Signal"
                else:
                    if bar['high'] >= pos['sl']:
                        exit_price = pos['sl']
                        reason = "SL"
                    elif bar['low'] <= pos['tp']:
                        exit_price = pos['tp']
                        reason = "TP"
                    elif self.exit_on_signal and sig == 'long':
                        exit_price = bar['close']
                        reason = "Signal"

                # 部分利確判定
                if not exit_price and self.partial_tp_rr > 0 and not pos.get('partial_done'):
                    entry_avg = np.mean([l['entry'] for l in pos['layers']])
                    if pos['dir'] == 'long':
                        if bar['high'] >= entry_avg + (entry_avg - pos['sl']) * self.partial_tp_rr:
                            # 半分決済
                            for l in pos['layers']:
                                pnl = (bar['close'] - l['entry']) * l['size'] * self.partial_tp_pct
                                cash += pnl
                                l['size'] *= (1 - self.partial_tp_pct)
                            pos['partial_done'] = True

                if exit_price:
                    # 全決済
                    total_pnl = 0
                    entry_avg = np.mean([l['entry'] for l in pos['layers']])
                    for l in pos['layers']:
                        pnl = (exit_price - l['entry']) * l['size'] if pos['dir'] == 'long' else (l['entry'] - exit_price) * l['size']
                        total_pnl += pnl
                    
                    cash += total_pnl
                    trades.append({
                        'dir': pos['dir'],
                        'entry': entry_avg,
                        'exit': exit_price,
                        'entry_time': pos['layers'][0]['entry_time'],
                        'exit_time': bar_time,
                        'pnl': total_pnl,
                        'reason': reason,
                        'pyramids': len(pos['layers']) - 1
                    })
                    in_pos = False
                    pos = {}
                else:
                    # トレーリングストップ更新
                    if self.trail_start_atr > 0:
                        entry_avg = np.mean([l['entry'] for l in pos['layers']])
                        if pos['dir'] == 'long':
                            if bar['close'] > entry_avg + bar_atr * self.trail_start_atr:
                                new_sl = bar['close'] - bar_atr * self.trail_dist_atr
                                pos['sl'] = max(pos['sl'], new_sl)
                        else:
                            if bar['close'] < entry_avg - bar_atr * self.trail_start_atr:
                                new_sl = bar['close'] + bar_atr * self.trail_dist_atr
                                pos['sl'] = min(pos['sl'], new_sl)

                    # 標準ピラミッティング判定
                    if not _use_livermore and pos['pyramid_count'] < self.pyramid_entries:
                        last_entry = pos['layers'][-1]['entry']
                        if pos['dir'] == 'long':
                            if bar['close'] >= last_entry + bar_atr * self.pyramid_atr:
                                # 追加
                                risk_amt = cash * self.risk_pct * self.symbol_risk_mult
                                sl_dist = abs(bar['close'] - pos['sl'])
                                if sl_dist > 0:
                                    size = (risk_amt / sl_dist) * self.pyramid_size_mult
                                    pos['layers'].append({'entry': bar['close'], 'size': size, 'entry_time': bar_time})
                                    pos['pyramid_count'] += 1
                                    # SLを建値に移動
                                    pos['sl'] = last_entry
                        else:
                            if bar['close'] <= last_entry - bar_atr * self.pyramid_atr:
                                risk_amt = cash * self.risk_pct * self.symbol_risk_mult
                                sl_dist = abs(bar['close'] - pos['sl'])
                                if sl_dist > 0:
                                    size = (risk_amt / sl_dist) * self.pyramid_size_mult
                                    pos['layers'].append({'entry': bar['close'], 'size': size, 'entry_time': bar_time})
                                    pos['pyramid_count'] += 1
                                    pos['sl'] = last_entry

            else:
                # エントリー判定
                if _trade_start_ts and bar_time < _trade_start_ts:
                    continue
                
                if allowed_months and bar_time.month not in allowed_months:
                    continue

                if sig in ['long', 'short']:
                    # ── Safety Valve (Proposal A) ──
                    # SL幅が狭すぎる、または出来高（Tick数）が極端に少ない場合は見送り
                    reason_skip = ""
                    if bar['tick_count'] < 5:
                        reason_skip = "Low Tick Count"
                    
                    if reason_skip:
                        self._log_skip(bar_time, sig, reason_skip)
                        continue

                    # SL決定
                    sl = 0
                    if sig == 'long':
                        if self.use_dynamic_sl:
                            sl = self._find_swing_low(bars, i, htf_bars, self.sl_n_confirm)
                            # ATRによる下限保証
                            sl = min(sl, bar['close'] - bar_atr * self.sl_min_atr)
                        else:
                            sl = bar['close'] - bar_atr * self.default_sl_atr
                    else:
                        if self.use_dynamic_sl:
                            sl = self._find_swing_high(bars, i, htf_bars, self.sl_n_confirm)
                            sl = max(sl, bar['close'] + bar_atr * self.sl_min_atr)
                        else:
                            sl = bar['close'] + bar_atr * self.default_sl_atr

                    # ── Safety Valve: SL距離チェック ──
                    sl_dist = abs(bar['close'] - sl)
                    if sl_dist < bar_atr * self.min_sl_atr_mult:
                        self._log_skip(bar_time, sig, f"SL too tight ({sl_dist:.4f} < {bar_atr*self.min_sl_atr_mult:.4f})")
                        continue
                    if sl_dist < bar['close'] * self.min_sl_price_pct:
                        self._log_skip(bar_time, sig, "SL too tight (price %)")
                        continue

                    # サイズ決定
                    risk_mult = 1.0
                    if sizer is not None:
                        risk_mult = sizer.get_multiplier(i)
                    
                    risk_amt = cash * self.risk_pct * self.symbol_risk_mult * risk_mult
                    size = risk_amt / sl_dist
                    
                    # ── Safety Valve: 最大ロット制限 ──
                    if size > self.max_pos_size:
                        size = self.max_pos_size
                    notional = size * bar['close']
                    if notional > cash * self.max_notional_pct:
                        size = (cash * self.max_notional_pct) / bar['close']

                    # TP決定
                    tp = 0
                    if sig == 'long':
                        tp = bar['close'] + (bar['close'] - sl) * self.dynamic_rr
                    else:
                        tp = bar['close'] - (sl - bar['close']) * self.dynamic_rr

                    in_pos = True
                    pos = {
                        'dir': sig,
                        'sl': sl,
                        'tp': tp,
                        'atr': bar_atr,
                        'layers': [{'entry': bar['close'], 'size': size, 'entry_time': bar_time}],
                        'pyramid_count': 0
                    }

            # DD更新
            peak = max(peak, cash)
            dd = (peak - cash) / peak
            max_dd = max(max_dd, dd)

        # 結果集計
        n = len(trades)
        if n == 0:
            return None
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        wr = len(wins) / n
        
        total_win = sum(t['pnl'] for t in wins)
        total_loss = abs(sum(t['pnl'] for t in losses))
        pf = total_win / total_loss if total_loss > 0 else 99.9
        
        avg_win_val = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss_val = np.mean([abs(t['pnl']) for t in losses]) if losses else 1
        rr_ratio = avg_win_val / avg_loss_val if avg_loss_val > 0 else 0
        
        final_cash = cash
        ret = (final_cash / self.init_cash - 1) * 100
        
        # 月次集計
        df_t = pd.DataFrame(trades)
        df_t['exit_time'] = pd.to_datetime(df_t['exit_time'])
        monthly_pnl = df_t.set_index('exit_time')['pnl'].resample('M').sum().to_dict()
        
        # 決済理由
        reasons = df_t['reason'].value_counts().to_dict()
        
        # ピラミッド成功率
        pyramid_trades = [t for t in trades if t['pyramids'] > 0]
        pyramid_rate = len(pyramid_trades) / n * 100

        # 合格判定
        passed = True
        if max_dd > self.target_max_dd: passed = False
        if n < self.target_min_trades: passed = False
        # WR判定 (RRが高い場合はWR閾値を下げる)
        wr_threshold = self.target_min_wr
        if rr_ratio > self.target_rr_threshold:
            wr_threshold *= 0.7
        if wr < wr_threshold: passed = False

        # 年換算トレード数
        total_days = (bars.index[-1] - bars.index[0]).days
        trades_per_year = n / (total_days / 365) if total_days > 0 else 0

        # 黄金ゾーン (8-24h保有)
        df_t['duration'] = (df_t['exit_time'] - pd.to_datetime(df_t['entry_time'])).dt.total_seconds() / 3600
        hold_8_24h = len(df_t[(df_t['duration'] >= 8) & (df_t['duration'] <= 24)]) / n * 100

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
            'avg_duration_hours': round(avg_dur, 1) if 'avg_dur' in locals() else 0,
            'exit_reasons': reasons,
            'monthly_pnl': monthly_pnl,
            'pyramid_trade_rate_pct': round(pyramid_rate, 1),
            'passed': passed,
            'trades_per_year': round(trades_per_year, 1),
            'hold_8_24h': hold_8_24h,   # 8-24h保有ゾーン (ユーザー黄金ゾーン)
            'trades': trades,
            # ── PF二重計算検証 (Proposal B) ──
            'pf_verify': verify_pf(trades, reported_pf=round(pf, 4)),
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


def verify_pf(trades: list, reported_pf: float = None) -> dict:
    """
    トレード履歴から PF を独立計算してレポート値と照合する (Proposal B)。

    目的:
      バックテストエンジンが返す profit_factor と、トレードリストから
      算出した値が一致するか検証し、バグを早期発見する。
    """
    if not trades:
        return {'status': 'no_trades'}
    
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
    
    calc_pf = sum(wins) / sum(losses) if sum(losses) > 0 else 99.9
    calc_pf = round(calc_pf, 4)
    
    diff = abs(calc_pf - reported_pf) if reported_pf is not None else 0
    status = "OK" if diff < 0.0001 else f"MISMATCH (diff: {diff:.6f})"
    
    return {
        'status': status,
        'calculated_pf': calc_pf,
        'reported_pf': reported_pf,
        'trade_count': len(trades)
    }
