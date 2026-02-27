"""
戦略ポートフォリオ・バックテスト
(改善指令 v2.0 Mission3)
=================================================
3戦略を 33% ずつ資金配分して同時実行し、
ポートフォリオ全体のパフォーマンスを評価する。

ポートフォリオ構成:
  1. コア戦略   : NY_8H_RSI_PB_45
       EMA200 上昇トレンド中の RSI 押し目ロング (ROI +90%)
  2. サブ戦略1 : BearDiv_Short_8H  ★新設計
       弱気ダイバージェンス + 主要レジスタンス + MTF フィルター ショート
  3. サブ戦略2 : 12H_DC30d_Confirm2
       12H ドンチャン30日 ブレイクアウト (ROI +42%)

設計思想:
  - 3戦略は異なるロジック (押し目・反転・ブレイクアウト) → 相関が低い
  - ポートフォリオ化でドローダウンを分散させ、収益を安定化
  - 資金配分は均等 (33.3%×3) で始め、結果を見て調整可能

実行:
  python scripts/backtest_portfolio.py          # サンプルデータ
  python scripts/backtest_portfolio.py --dukascopy   # Dukascopy OHLC
  python scripts/backtest_portfolio.py --start 2022-01-01
"""
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
IMAGES_DIR  = os.path.join(BASE_DIR, 'reports', 'images')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR,  exist_ok=True)

# ── ポートフォリオ設定 ──
TOTAL_CAPITAL = 10_000_000   # ポートフォリオ総資本 (1000万円)
N_STRATEGIES  = 3
ALLOC_PCT     = 1.0 / N_STRATEGIES   # 各戦略 33.3%
ALLOC_CAP     = int(TOTAL_CAPITAL * ALLOC_PCT)   # ~3,333,333 円


# ==============================================================
# エンジン生成ユーティリティ
# ==============================================================

def _make_long_engine_8h(init_cash):
    """NY_8H_RSI_PB_45 専用エンジン (ロング寄り、RR5)"""
    from lib.backtest import BacktestEngine
    return BacktestEngine(
        init_cash           = init_cash,
        risk_pct            = 0.03,
        default_sl_atr      = 1.0,
        default_tp_atr      = 10.0,
        slippage_pips       = 0.3,
        pip                 = 0.1,
        use_dynamic_sl      = True,
        sl_n_confirm        = 3,
        sl_min_atr          = 0.8,
        dynamic_rr          = 5.0,
        pyramid_entries     = 3,
        pyramid_atr         = 2.0,
        pyramid_size_mult   = 1.0,
        trail_start_atr     = 4.0,
        trail_dist_atr      = 3.0,
        exit_on_signal      = False,
        long_biased         = True,
        min_short_drop_atr  = 3.0,
        breakeven_rr        = 2.0,
        partial_tp_rr       = 0.0,
        partial_tp_pct      = 0.5,
        min_hold_hours      = 1.0,
        target_max_dd       = 0.30,
        target_min_wr       = 0.30,
        target_rr_threshold = 5.0,
        target_min_trades   = 4,
    )


def _make_short_engine_8h(init_cash):
    """BearDiv_Short_8H 専用エンジン (ショートバイアスなし、RR2)"""
    from lib.backtest import BacktestEngine
    return BacktestEngine(
        init_cash           = init_cash,
        risk_pct            = 0.03,
        default_sl_atr      = 1.0,
        default_tp_atr      = 10.0,
        slippage_pips       = 0.3,
        pip                 = 0.1,
        use_dynamic_sl      = True,
        sl_n_confirm        = 2,
        sl_min_atr          = 0.8,
        dynamic_rr          = 2.0,       # ショートは短期利確
        pyramid_entries     = 3,
        pyramid_atr         = 2.0,
        pyramid_size_mult   = 1.0,
        trail_start_atr     = 2.0,
        trail_dist_atr      = 1.5,
        exit_on_signal      = False,
        long_biased         = False,     # ショートバイアスなし (シグナル通り)
        min_short_drop_atr  = 1.0,       # ショート条件を緩める
        breakeven_rr        = 1.0,
        partial_tp_rr       = 1.0,       # RR1 で 50% 利確
        partial_tp_pct      = 0.5,
        min_hold_hours      = 1.0,
        target_max_dd       = 0.30,
        target_min_wr       = 0.30,
        target_rr_threshold = 5.0,
        target_min_trades   = 3,
    )


def _make_breakout_engine_12h(init_cash):
    """12H_DC30d_Confirm2 専用エンジン (ロング寄り、RR5)"""
    from lib.backtest import BacktestEngine
    return BacktestEngine(
        init_cash           = init_cash,
        risk_pct            = 0.03,
        default_sl_atr      = 1.0,
        default_tp_atr      = 10.0,
        slippage_pips       = 0.3,
        pip                 = 0.1,
        use_dynamic_sl      = True,
        sl_n_confirm        = 2,
        sl_min_atr          = 0.8,
        dynamic_rr          = 5.0,
        pyramid_entries     = 3,
        pyramid_atr         = 2.0,
        pyramid_size_mult   = 1.0,
        trail_start_atr     = 4.0,
        trail_dist_atr      = 3.0,
        exit_on_signal      = False,
        long_biased         = True,
        min_short_drop_atr  = 3.0,
        breakeven_rr        = 2.0,
        partial_tp_rr       = 0.0,
        partial_tp_pct      = 0.5,
        min_hold_hours      = 1.0,
        target_max_dd       = 0.30,
        target_min_wr       = 0.30,
        target_rr_threshold = 5.0,
        target_min_trades   = 3,
    )


# ==============================================================
# 戦略実行
# ==============================================================

def run_strategy(engine, bars, sig_fn, freq, name, htf_bars=None,
                 trade_start='2020-01-01'):
    """
    1戦略を実行し、trade_start 以降のトレードを strategy_name タグ付きで返す。
    """
    result = engine.run(bars, sig_fn, freq=freq, name=name, htf_bars=htf_bars)
    if result is None:
        return []

    ts_start = pd.Timestamp(trade_start)
    trades = []
    for t in result.get('trades', []):
        if pd.Timestamp(t['entry_time']) >= ts_start:
            t['strategy_name'] = name
            trades.append(t)
    return trades


# ==============================================================
# ポートフォリオ指標計算
# ==============================================================

def _calc_max_dd_pct(trade_list, init_capital):
    """トレードリスト (時系列順) から最大ドローダウン (%) を計算"""
    if not trade_list:
        return 0.0
    pnl_vals = [t['pnl'] for t in sorted(trade_list, key=lambda x: x['exit_time'])]
    equity = init_capital + pd.Series(pnl_vals).cumsum()
    running_max = equity.cummax()
    dd_pct = (equity - running_max) / running_max * 100
    return float(abs(dd_pct.min()))


def portfolio_metrics(all_trades):
    """全戦略の合算トレードからポートフォリオ指標を計算"""
    if not all_trades:
        return {}

    sorted_trades = sorted(all_trades, key=lambda t: t['entry_time'])
    wins   = [t for t in sorted_trades if t['pnl'] > 0]
    losses = [t for t in sorted_trades if t['pnl'] <= 0]

    total_pnl    = sum(t['pnl'] for t in sorted_trades)
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss   = abs(sum(t['pnl'] for t in losses))
    pf  = gross_profit / max(gross_loss, 1e-9)
    wr  = len(wins) / len(sorted_trades)

    # 累積 PnL 曲線
    times    = [pd.Timestamp(t['exit_time']) for t in sorted_trades]
    cum_pnl  = pd.Series([t['pnl'] for t in sorted_trades], index=times).cumsum()

    # 最大 DD (ポートフォリオ資本ベース)
    equity     = TOTAL_CAPITAL + cum_pnl
    run_max    = equity.cummax()
    dd_series  = (equity - run_max) / run_max * 100
    max_dd_pct = float(abs(dd_series.min()))

    roi = total_pnl / TOTAL_CAPITAL * 100

    # 戦略別内訳
    by_strategy = {}
    for t in sorted_trades:
        sn = t.get('strategy_name', 'unknown')
        if sn not in by_strategy:
            by_strategy[sn] = {'pnl': 0.0, 'trades': 0, 'wins': 0}
        by_strategy[sn]['pnl']    += t['pnl']
        by_strategy[sn]['trades'] += 1
        by_strategy[sn]['wins']   += 1 if t['pnl'] > 0 else 0

    return {
        'total_trades':  len(sorted_trades),
        'wins':          len(wins),
        'losses':        len(losses),
        'win_rate':      round(wr * 100, 1),
        'profit_factor': round(pf, 3),
        'total_pnl':     round(total_pnl, 0),
        'roi_pct':       round(roi, 2),
        'max_dd_pct':    round(max_dd_pct, 2),
        'cum_pnl':       cum_pnl,
        'by_strategy':   by_strategy,
    }


# ==============================================================
# 表示・保存
# ==============================================================

def print_portfolio_report(strategy_results, metrics):
    """ポートフォリオ結果を表示"""
    print(f"\n{'='*75}")
    print(f"  ポートフォリオ・バックテスト結果 (改善指令 v2.0)")
    print(f"  総資本: {TOTAL_CAPITAL:,} 円  |  配分: {ALLOC_CAP:,} 円 × {N_STRATEGIES} 戦略")
    print(f"{'='*75}")

    # ── 個別戦略サマリー ──
    print(f"\n{'─'*75}")
    print(f"{'戦略名':<32} {'TF':<5} {'取引':>5} {'勝率':>7} {'PF':>7} "
          f"{'ROI':>8} {'MaxDD':>7}")
    print(f"{'─'*75}")

    for name, res in strategy_results.items():
        trades = res['trades']
        freq   = res['freq']
        if not trades:
            print(f"  {name:<30} {freq:<5} {'(0)'}")
            continue
        wins_s = [t for t in trades if t['pnl'] > 0]
        loss_s = [t for t in trades if t['pnl'] <= 0]
        pnl_s  = sum(t['pnl'] for t in trades)
        pf_s   = (sum(t['pnl'] for t in wins_s)
                  / max(abs(sum(t['pnl'] for t in loss_s)), 1e-9))
        wr_s   = len(wins_s) / len(trades) * 100
        roi_s  = pnl_s / ALLOC_CAP * 100
        dd_s   = _calc_max_dd_pct(trades, ALLOC_CAP)
        mark   = ' ★' if pf_s >= 1.5 else ''
        print(f"  {name:<30} {freq:<5} {len(trades):>5} "
              f"{wr_s:>6.1f}% {pf_s:>7.3f} {roi_s:>7.1f}% {dd_s:>6.1f}%{mark}")

    # ── ポートフォリオ合計 ──
    if not metrics:
        print("\n[結果] 有効なトレードなし")
        return

    print(f"\n{'='*75}")
    print(f"  ポートフォリオ合計")
    print(f"{'─'*75}")
    print(f"  取引数               : {metrics['total_trades']} "
          f"(勝: {metrics['wins']} / 負: {metrics['losses']})")
    print(f"  勝率                 : {metrics['win_rate']:.1f}%")
    pf_mark = "★ 合格" if metrics['profit_factor'] >= 1.5 else "× 未達"
    print(f"  プロフィットファクター: {metrics['profit_factor']:.3f}  [{pf_mark}]")
    print(f"  合計 PnL             : {metrics['total_pnl']:+,.0f} 円")
    roi_mark = "★ 黒字" if metrics['roi_pct'] > 0 else "× 赤字"
    print(f"  ROI                  : {metrics['roi_pct']:+.2f}%  [{roi_mark}]")
    dd_mark = "★ 合格" if metrics['max_dd_pct'] <= 10 else ("△ 注意" if metrics['max_dd_pct'] <= 20 else "× 超過")
    print(f"  最大ドローダウン     : {metrics['max_dd_pct']:.2f}%  [{dd_mark}]")

    # 合格基準チェック
    print(f"\n  {'─'*50}")
    print(f"  合格基準チェック:")
    checks = [
        ("PF ≥ 1.5",           metrics['profit_factor'] >= 1.5),
        ("MaxDD ≤ 10%",        metrics['max_dd_pct'] <= 10),
        ("WinRate ≥ 35%",      metrics['win_rate'] >= 35),
        ("取引数 ≥ 30",        metrics['total_trades'] >= 30),
    ]
    passed = sum(1 for _, ok in checks if ok)
    for label, ok in checks:
        icon = "✓" if ok else "✗"
        print(f"    {icon}  {label}")
    print(f"    → {passed}/{len(checks)} 項目合格")

    # 戦略別 PnL 寄与
    print(f"\n  戦略別 PnL 寄与:")
    total_abs_pnl = max(abs(metrics['total_pnl']), 1)
    for strat, s in metrics['by_strategy'].items():
        contrib = s['pnl'] / total_abs_pnl * 100
        wr_s    = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
        pnl_str = f"{s['pnl']:+,.0f}"
        print(f"    {strat:<32}: {pnl_str:>12} 円  ({contrib:+5.1f}%)  "
              f"{s['trades']} 取引  WR={wr_s:.0f}%")

    print(f"{'='*75}")


def save_results(all_trades, metrics, strategy_results):
    """CSV に結果を保存"""
    if not all_trades:
        return

    # 全トレード CSV
    rows = []
    for t in sorted(all_trades, key=lambda x: x['entry_time']):
        rows.append({
            'strategy':      t.get('strategy_name', ''),
            'entry_time':    t['entry_time'],
            'exit_time':     t['exit_time'],
            'direction':     t['direction'],
            'entry_price':   t['entry_price'],
            'exit_price':    t['exit_price'],
            'sl':            t['sl'],
            'tp':            t['tp'],
            'size':          t['size'],
            'pnl':           round(t['pnl'], 2),
            'pnl_pct':       round(t['pnl_pct'], 4),
            'exit_reason':   t['exit_reason'],
            'duration_h':    round(t['duration_sec'] / 3600, 1),
        })
    trades_path = os.path.join(RESULTS_DIR, 'portfolio_trades.csv')
    pd.DataFrame(rows).to_csv(trades_path, index=False)
    print(f"\n[保存] 全トレード: {trades_path}")

    # サマリー CSV
    summary_rows = []
    for name, res in strategy_results.items():
        trades = res['trades']
        if not trades:
            continue
        wins_s = [t for t in trades if t['pnl'] > 0]
        loss_s = [t for t in trades if t['pnl'] <= 0]
        pnl_s  = sum(t['pnl'] for t in trades)
        pf_s   = (sum(t['pnl'] for t in wins_s)
                  / max(abs(sum(t['pnl'] for t in loss_s)), 1e-9))
        summary_rows.append({
            'strategy':      name,
            'freq':          res['freq'],
            'trades':        len(trades),
            'wins':          len(wins_s),
            'win_rate_pct':  round(len(wins_s) / len(trades) * 100, 1),
            'profit_factor': round(pf_s, 3),
            'total_pnl':     round(pnl_s, 0),
            'roi_pct':       round(pnl_s / ALLOC_CAP * 100, 2),
            'max_dd_pct':    round(_calc_max_dd_pct(trades, ALLOC_CAP), 2),
        })

    # ポートフォリオ合計行
    if metrics:
        summary_rows.append({
            'strategy':      '★ PORTFOLIO_TOTAL',
            'freq':          'mixed',
            'trades':        metrics['total_trades'],
            'wins':          metrics['wins'],
            'win_rate_pct':  metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'total_pnl':     metrics['total_pnl'],
            'roi_pct':       metrics['roi_pct'],
            'max_dd_pct':    metrics['max_dd_pct'],
        })

    summary_path = os.path.join(RESULTS_DIR, 'portfolio_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[保存] サマリー  : {summary_path}")


def _generate_portfolio_chart(cum_pnl_total, strategy_results, data_source):
    """ポートフォリオ累積 PnL チャートを生成"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        n_strats = len([r for r in strategy_results.values() if r['trades']])
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor='#1a1a2e')
        fig.suptitle(f'Portfolio Backtest  (source: {data_source})',
                     color='white', fontsize=13, y=1.01)

        colors = ['#00ff88', '#ff6644', '#44aaff']
        strat_colors = {}

        # ── 左: 戦略別 + 合計 累積 PnL ──
        ax1 = axes[0]
        ax1.set_facecolor('#16213e')
        ax1.set_title('Cumulative PnL by Strategy', color='white', fontsize=11)

        for ci, (name, res) in enumerate(strategy_results.items()):
            trades = res['trades']
            if not trades:
                continue
            sorted_t = sorted(trades, key=lambda x: x['exit_time'])
            times = [pd.Timestamp(t['exit_time']) for t in sorted_t]
            pnls  = [t['pnl'] for t in sorted_t]
            cum   = pd.Series(pnls, index=times).cumsum()
            col   = colors[ci % len(colors)]
            strat_colors[name] = col
            ax1.plot(cum.index, cum.values, color=col, lw=1.5,
                     label=name.replace('_', ' '), alpha=0.8)

        if cum_pnl_total is not None and len(cum_pnl_total) > 0:
            ax1.plot(cum_pnl_total.index, cum_pnl_total.values,
                     color='white', lw=2.5, label='TOTAL', alpha=1.0)
        ax1.axhline(0, color='gray', lw=0.5, linestyle='--')
        ax1.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
                   framealpha=0.7)
        ax1.tick_params(colors='#aaaaaa', labelsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
        for spine in ax1.spines.values():
            spine.set_color('#333355')

        # ── 右: ポートフォリオ DD チャート ──
        ax2 = axes[1]
        ax2.set_facecolor('#16213e')
        ax2.set_title('Portfolio Drawdown', color='white', fontsize=11)
        if cum_pnl_total is not None and len(cum_pnl_total) > 0:
            equity    = TOTAL_CAPITAL + cum_pnl_total
            run_max   = equity.cummax()
            dd_pct    = (equity - run_max) / run_max * 100
            ax2.fill_between(dd_pct.index, 0, dd_pct.values,
                             color='#ff4444', alpha=0.6)
            ax2.axhline(-10, color='yellow', lw=0.8, linestyle='--', label='10% DD')
            ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax2.tick_params(colors='#aaaaaa', labelsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
        for spine in ax2.spines.values():
            spine.set_color('#333355')

        plt.tight_layout()
        fpath = os.path.join(IMAGES_DIR, 'portfolio_backtest.png')
        plt.savefig(fpath, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        print(f"[画像] {fpath}")
    except Exception as e:
        print(f"[画像] 生成スキップ: {e}")


# ==============================================================
# メイン
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description='戦略ポートフォリオ・バックテスト')
    parser.add_argument('--dukascopy', action='store_true',
                        help='Dukascopy OHLC CSV を使用 (data/XAUUSD_*.csv)')
    parser.add_argument('--start', default='2020-01-01',
                        help='バックテスト開始日 (デフォルト: 2020-01-01)')
    parser.add_argument('--no-image', action='store_true',
                        help='グラフ生成をスキップ')
    args = parser.parse_args()

    print(f"{'='*75}")
    print(f"  sena3fx ポートフォリオ・バックテスト (改善指令 v2.0 Mission3)")
    print(f"  総資本: {TOTAL_CAPITAL:,} 円  |  配分: 33% × 3 戦略")
    print(f"{'='*75}\n")

    # ── データ読み込み ──
    from scripts.backtest_maedai import load_data
    data = load_data(
        use_dukascopy=args.dukascopy,
        start_warmup='2019-01-01',   # EMA200 等のウォームアップ含む
    )
    data_source = data['source']
    print(f"データソース: {data_source}\n")

    bars_8h  = data.get('8h')
    bars_12h = data.get('12h')
    bars_d1  = data.get('1d')

    if bars_8h is None or bars_12h is None:
        print("[エラー] 8H/12H データが取得できませんでした。")
        return

    # ── 戦略シグナル関数 ──
    from lib.yagami import (
        sig_rsi_pullback_filtered,
        sig_dc_filtered,
        sig_bearish_divergence_short,
    )

    portfolio_config = [
        # 1. コア: NY_8H_RSI_PB_45
        {
            'name':    'NY_8H_RSI_PB_45',
            'freq':    '8h',
            'bars':    bars_8h,
            'htf':     bars_d1,
            'engine':  _make_long_engine_8h(ALLOC_CAP),
            'sig_fn':  sig_rsi_pullback_filtered(
                           '8h', rsi_oversold=45,
                           ny_session_only=False,
                           block_noon_jst=True,
                           block_saturday=True,
                       ),
            'desc': 'EMA200上昇中のRSI押し目ロング (コア: ROI+90%)',
        },
        # 2. サブ1: 弱気ダイバージェンス ショート (Mission1)
        {
            'name':    'BearDiv_Short_8H',
            'freq':    '8h',
            'bars':    bars_8h,
            'htf':     bars_d1,
            'engine':  _make_short_engine_8h(ALLOC_CAP),
            'sig_fn':  sig_bearish_divergence_short(
                           freq='8h',
                           div_lookback=25,
                           div_pivot_bars=3,
                           res_lookback=100,
                           res_atr_mult=1.5,
                           rsi_period=14,
                           ema_days=200,
                           block_noon_jst=True,
                           block_saturday=True,
                       ),
            'desc': '弱気ダイバージェンス+主要レジ+MTFフィルター (新設計)',
        },
        # 3. サブ2: 12H_DC30d_Confirm2
        {
            'name':    '12H_DC30d_Confirm2',
            'freq':    '12h',
            'bars':    bars_12h,
            'htf':     bars_d1,
            'engine':  _make_breakout_engine_12h(ALLOC_CAP),
            'sig_fn':  sig_dc_filtered(
                           '12h', lookback_days=30, confirm_bars=2,
                           ny_session_only=False,
                           block_noon_jst=True,
                           block_saturday=True,
                       ),
            'desc': '12Hドンチャン30日ブレイクアウト (サブ2: ROI+42%)',
        },
    ]

    # ── 各戦略を実行 ──
    print("戦略実行中...")
    print(f"{'─'*55}")
    strategy_results = {}
    all_trades       = []

    for cfg in portfolio_config:
        name    = cfg['name']
        bars    = cfg['bars']
        htf     = cfg['htf']
        engine  = cfg['engine']
        sig_fn  = cfg['sig_fn']
        freq    = cfg['freq']
        desc    = cfg['desc']

        if bars is None or len(bars) < 50:
            print(f"  {name}: データ不足 → スキップ")
            strategy_results[name] = {'trades': [], 'freq': freq}
            continue

        print(f"  [{freq.upper()}] {name}")
        print(f"         {desc}")

        trades = run_strategy(
            engine=engine, bars=bars, sig_fn=sig_fn,
            freq=freq, name=name, htf_bars=htf,
            trade_start=args.start,
        )

        wins_n  = sum(1 for t in trades if t['pnl'] > 0)
        pnl_sum = sum(t['pnl'] for t in trades)
        pf_n    = (sum(t['pnl'] for t in trades if t['pnl'] > 0)
                   / max(abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0)), 1e-9))
        if trades:
            print(f"         → {len(trades)} 取引  "
                  f"WR={wins_n/len(trades)*100:.0f}%  "
                  f"PF={pf_n:.2f}  PnL={pnl_sum:+,.0f}")
        else:
            print(f"         → 0 取引")
        print()

        strategy_results[name] = {'trades': trades, 'freq': freq}
        all_trades.extend(trades)

    # ── ポートフォリオ指標 ──
    print("ポートフォリオ集計中...")
    metrics = portfolio_metrics(all_trades)

    # ── 結果表示 ──
    print_portfolio_report(strategy_results, metrics)

    # ── 保存 ──
    save_results(all_trades, metrics, strategy_results)

    # ── グラフ ──
    if not args.no_image and metrics:
        _generate_portfolio_chart(
            metrics.get('cum_pnl'),
            strategy_results,
            data_source,
        )

    # ── 最終サマリー ──
    print(f"\n結果ファイル:")
    print(f"  results/portfolio_trades.csv")
    print(f"  results/portfolio_summary.csv")
    if not args.no_image:
        print(f"  reports/images/portfolio_backtest.png")


if __name__ == '__main__':
    main()
