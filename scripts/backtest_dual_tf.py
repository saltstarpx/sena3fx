"""
デュアルTF バックテスト + 定量分析
=====================================
4H EMA21 トレンド × 15m DC ブレイクアウト
スイングSL + RRフィルター + 時間フィルター

実行: python scripts/backtest_dual_tf.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

from lib.swing   import build_swing_sl_series_long, build_swing_sl_series_short
from lib.dual_tf import align_4h_trend_to_15m, compute_15m_signals_vectorized
from live.time_filter import filter_signals_by_time

# ------------------------------------------------------------------ #
#  データ読み込み                                                     #
# ------------------------------------------------------------------ #
def load_ohlc(path):
    df = pd.read_csv(path)
    try:
        dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert(None)
    df['datetime'] = dt
    df = df.set_index('datetime').sort_index()
    return df[['open','high','low','close','volume']].astype(float)

# ------------------------------------------------------------------ #
#  バックテストエンジン                                               #
# ------------------------------------------------------------------ #
def run_backtest(bars_4h, bars_15m, instrument='XAU_USD',
                 init_cash=10_000_000, risk_pct=0.02,
                 ema_days=21, dc_lookback=20,
                 swing_window=3, swing_lookback=10,
                 rr_target=2.5, min_rr=2.0, long_only=True,
                 start_date=None, end_date=None):
    """
    バー・バー・シミュレーション (ロングオンリー)。

    Returns:
        trades:        list of dict  (全トレード記録)
        equity_curve:  pd.Series    (15m 足レベルのエクイティ)
    """
    if start_date:
        b4 = bars_4h[ bars_4h.index  >= start_date]
        b15= bars_15m[bars_15m.index >= start_date]
    else:
        b4, b15 = bars_4h, bars_15m
    if end_date:
        b4 = b4[b4.index   <= end_date]
        b15= b15[b15.index <= end_date]

    print(f"\n[{instrument}] 4H: {len(b4)} 本  15m: {len(b15)} 本")

    # --- シグナル生成 ---
    print("  4H トレンド計算...")
    trend_15m = align_4h_trend_to_15m(b4, b15, ema_days)

    print("  15m エントリーシグナル計算...")
    signals = compute_15m_signals_vectorized(b15, trend_15m, dc_lookback)
    print(f"  時間フィルター前: long={( signals=='long').sum()}, short={(signals=='short').sum()}")

    signals = filter_signals_by_time(signals)
    print(f"  時間フィルター後: long={( signals=='long').sum()}, short={(signals=='short').sum()}")

    # --- スイングSL ---
    print("  スイングSL計算...")
    sl_long_4h = build_swing_sl_series_long( b4, swing_window, swing_lookback)
    if not long_only:
        sl_short_4h = build_swing_sl_series_short(b4, swing_window, swing_lookback)

    comb_idx     = b4.index.union(b15.index)
    sl_long_15m  = sl_long_4h.reindex(comb_idx).ffill().reindex(b15.index)
    if not long_only:
        sl_short_15m = sl_short_4h.reindex(comb_idx).ffill().reindex(b15.index)

    # --- シミュレーション ---
    print("  バー・バー・シミュレーション...")
    cash      = float(init_cash)
    peak_cash = float(init_cash)
    position  = None
    trades    = []
    equity    = []
    skipped_no_sl   = 0
    skipped_rr      = 0
    skipped_size0   = 0

    close_arr = b15['close'].values
    high_arr  = b15['high'].values
    low_arr   = b15['low'].values
    idx_arr   = b15.index

    n = len(b15)
    warmup = dc_lookback + swing_lookback + swing_window * 2 + 5

    for i in range(warmup, n):
        dt    = idx_arr[i]
        close = close_arr[i]
        high  = high_arr[i]
        low   = low_arr[i]

        # === ポジションあり: SL/TP チェック ===
        if position is not None:
            ep   = position['entry']
            sl_p = position['sl']
            tp_p = position['tp']
            side = position['side']

            exit_price  = None
            exit_reason = None

            if side == 'long':
                if low <= sl_p:
                    exit_price  = sl_p
                    exit_reason = 'SL'
                elif high >= tp_p:
                    exit_price  = tp_p
                    exit_reason = 'TP'
            else:
                if high >= sl_p:
                    exit_price  = sl_p
                    exit_reason = 'SL'
                elif low <= tp_p:
                    exit_price  = tp_p
                    exit_reason = 'TP'

            if exit_price is not None:
                units = position['units']
                pnl   = (exit_price - ep) * units if side == 'long' else (ep - exit_price) * units
                cash += pnl
                peak_cash = max(peak_cash, cash)

                sl_dist   = abs(ep - sl_p)
                tp_dist   = abs(tp_p - ep)
                actual_rr = tp_dist / sl_dist if sl_dist > 0 else 0
                won = pnl > 0

                trades.append({
                    'entry_time':  position['entry_time'],
                    'exit_time':   dt,
                    'instrument':  instrument,
                    'side':        side,
                    'entry':       ep,
                    'sl':          sl_p,
                    'tp':          tp_p,
                    'exit':        exit_price,
                    'units':       units,
                    'pnl':         pnl,
                    'pnl_pct':     pnl / position['cash_at_entry'] * 100,
                    'reason':      exit_reason,
                    'rr_set':      actual_rr,
                    'won':         won,
                    'sl_dist':     sl_dist,
                    'hold_bars':   i - position['entry_idx'],
                })
                position = None

        # === ポジションなし: エントリー判断 ===
        if position is None:
            sig = signals.iloc[i]

            if sig == 'long' or (sig == 'short' and not long_only):
                sl_price = sl_long_15m.iloc[i] if sig == 'long' else sl_short_15m.iloc[i]

                ok = True
                if pd.isna(sl_price):
                    skipped_no_sl += 1; ok = False

                if ok:
                    ep      = close
                    sl_dist = abs(ep - sl_price)
                    if sl_dist <= 1e-6:
                        skipped_no_sl += 1; ok = False

                if ok:
                    if sig == 'long'  and sl_price >= ep:
                        skipped_no_sl += 1; ok = False
                    if sig == 'short' and sl_price <= ep:
                        skipped_no_sl += 1; ok = False

                if ok:
                    tp_price  = (ep + rr_target * sl_dist) if sig == 'long' else (ep - rr_target * sl_dist)
                    actual_rr = abs(tp_price - ep) / sl_dist
                    if actual_rr < min_rr:
                        skipped_rr += 1; ok = False

                if ok:
                    units = int((cash * risk_pct) / sl_dist)
                    if units <= 0:
                        skipped_size0 += 1; ok = False

                if ok:
                    position = {
                        'entry_time':    dt,
                        'entry_idx':     i,
                        'side':          sig,
                        'entry':         ep,
                        'sl':            sl_price,
                        'tp':            tp_price,
                        'units':         units,
                        'cash_at_entry': cash,
                    }

        # === エクイティ記録 ===
        if position is not None:
            ep    = position['entry']
            units = position['units']
            side  = position['side']
            unreal = (close - ep) * units if side == 'long' else (ep - close) * units
            nav = cash + unreal
        else:
            nav = cash
        equity.append(nav)

    if position is not None:
        # 未決済ポジションを最終バーで強制クローズ
        last_close = close_arr[-1]
        ep    = position['entry']
        units = position['units']
        side  = position['side']
        pnl   = (last_close - ep) * units if side == 'long' else (ep - last_close) * units
        cash += pnl
        trades.append({
            'entry_time':  position['entry_time'],
            'exit_time':   idx_arr[-1],
            'instrument':  instrument,
            'side':        side,
            'entry':       ep,
            'sl':          position['sl'],
            'tp':          position['tp'],
            'exit':        last_close,
            'units':       units,
            'pnl':         pnl,
            'pnl_pct':     pnl / position['cash_at_entry'] * 100,
            'reason':      'FORCE_CLOSE',
            'rr_set':      rr_target,
            'won':         pnl > 0,
            'sl_dist':     abs(ep - position['sl']),
            'hold_bars':   n - position['entry_idx'],
        })

    eq_series = pd.Series(equity, index=idx_arr[warmup:])
    print(f"  完了: {len(trades)} トレード "
          f"(SL無効スキップ={skipped_no_sl}, RRスキップ={skipped_rr})")
    return trades, eq_series, float(cash)

# ------------------------------------------------------------------ #
#  定量分析                                                           #
# ------------------------------------------------------------------ #
def quantitative_analysis(trades, equity_curve, init_cash, instrument):
    """全統計指標の計算"""
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time']  = pd.to_datetime(df['exit_time'])
    df['month']      = df['exit_time'].dt.to_period('M')
    df['hour_entry'] = df['entry_time'].dt.hour
    df['hold_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600

    n         = len(df)
    wins      = df[df['won']]
    losses    = df[~df['won']]
    n_win     = len(wins)
    n_loss    = len(losses)
    win_rate  = n_win / n * 100 if n > 0 else 0

    gross_profit = wins['pnl'].sum()   if n_win  > 0 else 0
    gross_loss   = abs(losses['pnl'].sum()) if n_loss > 0 else 0
    pf           = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    total_pnl = df['pnl'].sum()
    roi        = total_pnl / init_cash * 100

    # ドローダウン
    eq_arr = equity_curve.values
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (peak - eq_arr) / peak * 100
    max_dd = dd.max()

    # シャープ・ソルティノ・カルマー (15m足単位)
    eq_s    = equity_curve
    returns = eq_s.pct_change().dropna()
    annual_factor = 252 * 96  # 15m足 → 年率換算 (1年=252日×96本/日)
    mean_r  = returns.mean()
    std_r   = returns.std()
    neg_r   = returns[returns < 0]

    sharpe  = (mean_r / std_r * np.sqrt(annual_factor)) if std_r > 0 else 0
    sortino = (mean_r / neg_r.std() * np.sqrt(annual_factor)) if len(neg_r) > 0 and neg_r.std() > 0 else 0

    # 月次PnL
    monthly_pnl = df.groupby('month')['pnl'].sum()
    monthly_pnl_pct = monthly_pnl / init_cash * 100

    # 年率換算リターン (バックテスト期間で換算)
    bt_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    annual_roi = (roi / bt_days * 365) if bt_days > 0 else 0
    calmar = annual_roi / max_dd if max_dd > 0 else float('inf')

    # 連勝・連敗
    streaks = []
    cur_streak = 0
    cur_type   = None
    max_win_streak  = 0
    max_loss_streak = 0
    for won in df['won']:
        t = 'W' if won else 'L'
        if t == cur_type:
            cur_streak += 1
        else:
            cur_streak = 1
            cur_type   = t
        if t == 'W':
            max_win_streak  = max(max_win_streak, cur_streak)
        else:
            max_loss_streak = max(max_loss_streak, cur_streak)

    # 平均保有時間
    avg_hold = df['hold_hours'].mean()
    avg_win_hold  = wins['hold_hours'].mean()  if n_win  > 0 else 0
    avg_loss_hold = losses['hold_hours'].mean() if n_loss > 0 else 0

    # 期待値 (RR加重)
    avg_win_pnl  = wins['pnl'].mean()   if n_win  > 0 else 0
    avg_loss_pnl = losses['pnl'].mean() if n_loss > 0 else 0
    expectancy   = (win_rate/100 * avg_win_pnl) + ((1 - win_rate/100) * avg_loss_pnl)

    stats = dict(
        instrument       = instrument,
        n_trades         = n,
        n_win            = n_win,
        n_loss           = n_loss,
        win_rate         = win_rate,
        profit_factor    = pf,
        roi              = roi,
        annual_roi       = annual_roi,
        total_pnl        = total_pnl,
        gross_profit     = gross_profit,
        gross_loss       = gross_loss,
        max_dd           = max_dd,
        sharpe           = sharpe,
        sortino          = sortino,
        calmar           = calmar,
        avg_win_pnl      = avg_win_pnl,
        avg_loss_pnl     = avg_loss_pnl,
        avg_rr_set       = df['rr_set'].mean(),
        expectancy       = expectancy,
        avg_hold_h       = avg_hold,
        avg_win_hold_h   = avg_win_hold,
        avg_loss_hold_h  = avg_loss_hold,
        max_win_streak   = max_win_streak,
        max_loss_streak  = max_loss_streak,
        monthly_pnl      = monthly_pnl,
        monthly_pnl_pct  = monthly_pnl_pct,
        df               = df,
    )
    return stats

# ------------------------------------------------------------------ #
#  チャート生成                                                       #
# ------------------------------------------------------------------ #
def plot_analysis(stats_list, equity_curves, init_cash, outpath):
    """金・銀の定量分析チャートを生成"""

    n_inst = len(stats_list)
    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor('#0d1117')

    gs = gridspec.GridSpec(
        5, 2,
        figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.93, bottom=0.04, left=0.07, right=0.97
    )

    GOLD  = '#FFD700'
    SILVER= '#C0C0C0'
    GREEN = '#26a641'
    RED   = '#f85149'
    BLUE  = '#58a6ff'

    colors = [GOLD, SILVER]
    inst_colors = {
        'XAU_USD': GOLD,
        'XAG_USD': SILVER,
    }

    # タイトル
    fig.text(
        0.5, 0.965,
        'デュアルTFバックテスト 定量分析レポート\n'
        '4H EMA21 トレンド × 15m DC ブレイクアウト + スイングSL\n'
        '期間: 2025-09 〜 2026-02  /  初期資金: ¥10,000,000',
        ha='center', va='top', color='white',
        fontsize=13, fontweight='bold', linespacing=1.6
    )

    # --- [0,0] + [0,1] エクイティカーブ ---
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.set_facecolor('#161b22')
    ax_eq.set_title('エクイティカーブ (複利)', color='white', fontsize=11, pad=6)

    for stats, eq, col in zip(stats_list, equity_curves, colors):
        inst = stats['instrument']
        label = f"{inst}  ROI:{stats['roi']:.1f}%  MaxDD:{stats['max_dd']:.1f}%"
        ax_eq.plot(eq.index, eq.values / 1e6, color=col, lw=1.6, label=label, alpha=0.9)

    ax_eq.axhline(init_cash / 1e6, color='white', lw=0.8, ls='--', alpha=0.4)
    ax_eq.set_ylabel('資産 (百万円)', color='white', fontsize=9)
    ax_eq.tick_params(colors='white', labelsize=8)
    ax_eq.spines[:].set_color('#30363d')
    ax_eq.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d',
                 labelcolor='white', loc='upper left')
    ax_eq.yaxis.label.set_color('white')

    # --- [1,0] 月次PnL% ヒートマップ (Gold) ---
    for col_idx, (stats, col) in enumerate(zip(stats_list, colors)):
        ax_hm = fig.add_subplot(gs[1, col_idx])
        ax_hm.set_facecolor('#161b22')
        mp = stats['monthly_pnl_pct']
        inst = stats['instrument']

        months = mp.index
        vals   = mp.values
        colors_hm = [GREEN if v >= 0 else RED for v in vals]

        bars = ax_hm.bar(
            [str(m) for m in months], vals,
            color=colors_hm, alpha=0.85, width=0.7
        )
        ax_hm.axhline(0, color='white', lw=0.6, alpha=0.5)
        ax_hm.set_title(f'{inst} 月次PnL%', color=col, fontsize=10, pad=5)
        ax_hm.tick_params(colors='white', labelsize=7.5, axis='both')
        ax_hm.tick_params(axis='x', rotation=30)
        ax_hm.set_ylabel('PnL (%)', color='white', fontsize=8)
        ax_hm.spines[:].set_color('#30363d')

        for bar, v in zip(bars, vals):
            ax_hm.text(
                bar.get_x() + bar.get_width()/2,
                v + (0.3 if v >= 0 else -0.8),
                f'{v:+.1f}%', ha='center', va='bottom' if v >= 0 else 'top',
                color='white', fontsize=7, fontweight='bold'
            )

    # --- [2,0] 勝率・PF・Calmar サマリーバー ---
    ax_sum = fig.add_subplot(gs[2, 0])
    ax_sum.set_facecolor('#161b22')
    ax_sum.set_title('主要指標比較', color='white', fontsize=10, pad=5)

    metrics = ['勝率%', 'PF×10', 'Calmar']
    x = np.arange(len(metrics))
    width = 0.35

    for i_s, (stats, col) in enumerate(zip(stats_list, colors)):
        vals = [
            stats['win_rate'],
            min(stats['profit_factor'] * 10, 100),
            min(stats['calmar'], 100),
        ]
        offset = (i_s - 0.5) * width
        bars_s = ax_sum.bar(x + offset, vals, width, color=col, alpha=0.85,
                            label=stats['instrument'])
        for bar, v in zip(bars_s, vals):
            ax_sum.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.1f}', ha='center', va='bottom', color='white', fontsize=8
            )

    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(metrics, color='white', fontsize=9)
    ax_sum.tick_params(colors='white', labelsize=8)
    ax_sum.spines[:].set_color('#30363d')
    ax_sum.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # --- [2,1] 統計テーブル ---
    ax_tbl = fig.add_subplot(gs[2, 1])
    ax_tbl.set_facecolor('#161b22')
    ax_tbl.axis('off')
    ax_tbl.set_title('統計サマリー', color='white', fontsize=10, pad=5)

    headers = ['指標', 'Gold (XAU)', 'Silver (XAG)']
    rows_data = []
    if len(stats_list) >= 2:
        s0, s1 = stats_list[0], stats_list[1]
        rows_data = [
            ['トレード数',     f"{s0['n_trades']}",          f"{s1['n_trades']}"],
            ['勝率',          f"{s0['win_rate']:.1f}%",      f"{s1['win_rate']:.1f}%"],
            ['PF',            f"{s0['profit_factor']:.2f}",  f"{s1['profit_factor']:.2f}"],
            ['ROI',           f"{s0['roi']:.1f}%",           f"{s1['roi']:.1f}%"],
            ['年率ROI',       f"{s0['annual_roi']:.1f}%",    f"{s1['annual_roi']:.1f}%"],
            ['MaxDD',         f"{s0['max_dd']:.1f}%",        f"{s1['max_dd']:.1f}%"],
            ['Sharpe',        f"{s0['sharpe']:.2f}",         f"{s1['sharpe']:.2f}"],
            ['Sortino',       f"{s0['sortino']:.2f}",        f"{s1['sortino']:.2f}"],
            ['Calmar',        f"{s0['calmar']:.2f}",         f"{s1['calmar']:.2f}"],
            ['期待値',        f"${s0['expectancy']:,.0f}",   f"${s1['expectancy']:,.0f}"],
            ['平均保有',      f"{s0['avg_hold_h']:.1f}h",   f"{s1['avg_hold_h']:.1f}h"],
            ['最大連勝',      f"{s0['max_win_streak']}",     f"{s1['max_win_streak']}"],
            ['最大連敗',      f"{s0['max_loss_streak']}",    f"{s1['max_loss_streak']}"],
        ]
    elif len(stats_list) == 1:
        s0 = stats_list[0]
        rows_data = [
            ['トレード数', f"{s0['n_trades']}", '—'],
            ['勝率',      f"{s0['win_rate']:.1f}%", '—'],
            ['PF',        f"{s0['profit_factor']:.2f}", '—'],
            ['ROI',       f"{s0['roi']:.1f}%", '—'],
            ['MaxDD',     f"{s0['max_dd']:.1f}%", '—'],
            ['Sharpe',    f"{s0['sharpe']:.2f}", '—'],
            ['Calmar',    f"{s0['calmar']:.2f}", '—'],
        ]

    tbl = ax_tbl.table(
        cellText=rows_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1c2128' if r % 2 == 0 else '#161b22')
        cell.set_edgecolor('#30363d')
        cell.set_text_props(color='white')
        if r == 0:
            cell.set_facecolor('#0d1117')
            cell.set_text_props(color=GOLD if c == 1 else (SILVER if c == 2 else 'white'),
                                fontweight='bold')
    tbl.scale(1, 1.35)

    # --- [3,0] エントリー時刻分布 ---
    ax_hr = fig.add_subplot(gs[3, 0])
    ax_hr.set_facecolor('#161b22')
    ax_hr.set_title('エントリー時刻分布 (UTC)', color='white', fontsize=10, pad=5)

    all_hours = range(24)
    for stats, col in zip(stats_list, colors):
        df_t = stats['df']
        hour_cnt = df_t.groupby('hour_entry').size().reindex(all_hours, fill_value=0)
        ax_hr.bar(
            [h - 0.2 if col == GOLD else h + 0.2 for h in all_hours],
            hour_cnt.values, width=0.4, color=col, alpha=0.8,
            label=stats['instrument']
        )

    ax_hr.set_xticks(range(0, 24, 3))
    ax_hr.tick_params(colors='white', labelsize=8)
    ax_hr.set_xlabel('時刻 (UTC)', color='white', fontsize=8)
    ax_hr.set_ylabel('トレード数', color='white', fontsize=8)
    ax_hr.spines[:].set_color('#30363d')
    ax_hr.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # --- [3,1] PnL分布ヒストグラム ---
    ax_pnl = fig.add_subplot(gs[3, 1])
    ax_pnl.set_facecolor('#161b22')
    ax_pnl.set_title('PnL 分布 (USD/トレード)', color='white', fontsize=10, pad=5)

    for stats, col in zip(stats_list, colors):
        df_t = stats['df']
        ax_pnl.hist(df_t['pnl'] / 1000, bins=20, color=col, alpha=0.5,
                    label=stats['instrument'], edgecolor='none')

    ax_pnl.axvline(0, color='white', lw=1, ls='--', alpha=0.6)
    ax_pnl.set_xlabel('PnL (千USD)', color='white', fontsize=8)
    ax_pnl.set_ylabel('件数', color='white', fontsize=8)
    ax_pnl.tick_params(colors='white', labelsize=8)
    ax_pnl.spines[:].set_color('#30363d')
    ax_pnl.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # --- [4,0] 連勝/連敗 ---
    ax_str = fig.add_subplot(gs[4, 0])
    ax_str.set_facecolor('#161b22')
    ax_str.set_title('個別トレード損益 (順番)', color='white', fontsize=10, pad=5)

    for stats, col in zip(stats_list, colors):
        df_t = stats['df']
        pnls = df_t['pnl'].values
        bar_colors = [GREEN if p > 0 else RED for p in pnls]
        ax_str.bar(range(len(pnls)), pnls / 1000, color=bar_colors, alpha=0.75, width=0.8)

    ax_str.axhline(0, color='white', lw=0.8, alpha=0.5)
    ax_str.set_xlabel('トレード番号', color='white', fontsize=8)
    ax_str.set_ylabel('PnL (千USD)', color='white', fontsize=8)
    ax_str.tick_params(colors='white', labelsize=8)
    ax_str.spines[:].set_color('#30363d')

    # --- [4,1] 保有時間分布 ---
    ax_hld = fig.add_subplot(gs[4, 1])
    ax_hld.set_facecolor('#161b22')
    ax_hld.set_title('保有時間分布 (時間)', color='white', fontsize=10, pad=5)

    for stats, col in zip(stats_list, colors):
        df_t = stats['df']
        ax_hld.hist(df_t['hold_hours'], bins=20, color=col, alpha=0.5,
                    label=stats['instrument'], edgecolor='none')

    ax_hld.set_xlabel('保有時間 (h)', color='white', fontsize=8)
    ax_hld.set_ylabel('件数', color='white', fontsize=8)
    ax_hld.tick_params(colors='white', labelsize=8)
    ax_hld.spines[:].set_color('#30363d')
    ax_hld.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"\nチャート保存: {outpath}")

# ------------------------------------------------------------------ #
#  メイン                                                             #
# ------------------------------------------------------------------ #
def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    START = '2025-09-01'
    END   = '2026-02-28'
    INIT  = 10_000_000

    params = dict(
        init_cash    = INIT,
        risk_pct     = 0.02,       # 資金の2%リスク
        ema_days     = 21,
        dc_lookback  = 20,
        swing_window = 3,
        swing_lookback = 10,
        rr_target    = 2.5,        # TP = 2.5 × SL距離
        min_rr       = 2.0,
        long_only    = True,
        start_date   = START,
        end_date     = END,
    )

    stats_list    = []
    equity_curves = []
    final_cash    = []

    # ---- Gold ----
    b4_xau  = load_ohlc(os.path.join(root, 'data/ohlc/XAUUSD_2025_4h.csv'))
    b15_xau = load_ohlc(os.path.join(root, 'data/ohlc/XAUUSD_2025_15m.csv'))
    trades_xau, eq_xau, cash_xau = run_backtest(b4_xau, b15_xau, instrument='XAU_USD', **params)
    stats_xau = quantitative_analysis(trades_xau, eq_xau, INIT, 'XAU_USD')
    stats_list.append(stats_xau)
    equity_curves.append(eq_xau)
    final_cash.append(cash_xau)

    # ---- Silver ----
    b4_xag  = load_ohlc(os.path.join(root, 'data/ohlc/XAGUSD_2025_4h.csv'))
    b15_xag = load_ohlc(os.path.join(root, 'data/ohlc/XAGUSD_2025_15m.csv'))
    trades_xag, eq_xag, cash_xag = run_backtest(b4_xag, b15_xag, instrument='XAG_USD', **params)
    stats_xag = quantitative_analysis(trades_xag, eq_xag, INIT, 'XAG_USD')
    stats_list.append(stats_xag)
    equity_curves.append(eq_xag)
    final_cash.append(cash_xag)

    # ---- 結果表示 ----
    print("\n" + "=" * 65)
    print("  デュアルTF バックテスト 定量分析結果")
    print(f"  期間: {START} 〜 {END}   初期資金: ¥{INIT:,}")
    print("=" * 65)

    for stats, fc in zip(stats_list, final_cash):
        if not stats:
            continue
        inst = stats['instrument']
        label = '金 (XAU_USD)' if 'XAU' in inst else '銀 (XAG_USD)'
        print(f"\n【{label}】")
        print(f"  トレード数:    {stats['n_trades']} "
              f"(勝:{stats['n_win']} 負:{stats['n_loss']})")
        print(f"  勝率:          {stats['win_rate']:.1f}%")
        print(f"  PF:            {stats['profit_factor']:.3f}")
        print(f"  ROI:           {stats['roi']:+.1f}%")
        print(f"  年率ROI:       {stats['annual_roi']:+.1f}%")
        print(f"  最終資産:      ¥{fc:,.0f}")
        print(f"  最大DD:        {stats['max_dd']:.2f}%")
        print(f"  Sharpe:        {stats['sharpe']:.3f}")
        print(f"  Sortino:       {stats['sortino']:.3f}")
        print(f"  Calmar:        {stats['calmar']:.3f}")
        print(f"  期待値/T:      ${stats['expectancy']:,.0f}")
        print(f"  平均保有:      {stats['avg_hold_h']:.1f}h "
              f"(勝:{stats['avg_win_hold_h']:.1f}h / "
              f"負:{stats['avg_loss_hold_h']:.1f}h)")
        print(f"  最大連勝/連敗: {stats['max_win_streak']} / {stats['max_loss_streak']}")
        print(f"  月次PnL%:")
        for m, v in stats['monthly_pnl_pct'].items():
            bar = '█' * int(abs(v) / 2)
            sign = '+' if v >= 0 else ''
            print(f"    {m}: {sign}{v:.1f}%  {bar}")

    # ---- チャート出力 ----
    outpath = os.path.join(root, 'results', 'dual_tf_quant_analysis.png')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    valid_stats = [s for s in stats_list if s]
    valid_eq    = [eq for eq, s in zip(equity_curves, stats_list) if s]
    plot_analysis(valid_stats, valid_eq, INIT, outpath)
    print(f"\nチャート: results/dual_tf_quant_analysis.png")

if __name__ == '__main__':
    main()
