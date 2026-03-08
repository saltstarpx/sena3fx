"""
戦略ポートフォリオ バックテスト比較
====================================
3チーム (Yagami, Maedai, Risk Manager) の戦略を統合評価し、
以下を比較レポートする:

  Task 1: USD強弱フィルターの有無による PF/MDD/Sharpe 比較
  Task 2: 季節フィルター (7月+9月除外) の定量的評価
  Task 3: Maedai戦略の Sharpe/Calmar ランキング + DC パラメータ探索

実行方法:
  python scripts/backtest_portfolio.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.backtest import BacktestEngine
from lib.yagami import (
    sig_yagami_A, sig_yagami_B, sig_yagami_full_filter,
    sig_yagami_london_ny,
    sig_maedai_dc_ema_tf, sig_maedai_yagami_union, sig_maedai_best,
)
from strategies.market_filters import (
    make_usd_filtered_signal, seasonal_effectiveness,
    SEASON_SKIP_JUL_SEP, SEASON_ALL,
)

OHLC_1H = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_1h.csv')
OHLC_4H = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
RESULTS = os.path.join(ROOT, 'results')


def load_ohlc(path):
    df = pd.read_csv(path)
    try:
        dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    df['datetime'] = dt
    df = df.set_index('datetime').sort_index()
    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    return df[cols].astype(float)


def fmt(v, prec=2):
    if v is None:
        return 'N/A'
    return f'{v:.{prec}f}'


# ================================================================== #
#  Task 1: USD 強弱フィルター比較                                      #
# ================================================================== #
def task1_usd_filter(df_1h, df_4h):
    print('\n' + '=' * 70)
    print('  Task 1 [Teammate C]: USD 強弱フィルター比較')
    print('=' * 70)

    engine = BacktestEngine(
        init_cash=5_000_000, risk_pct=0.05,
        default_sl_atr=2.0, default_tp_atr=4.0,
        pyramid_entries=0, target_max_dd=0.20,
        target_min_wr=0.35, target_rr_threshold=2.0,
        target_min_trades=20,
    )

    # Yagami 戦略 + Maedai 戦略 (素 vs USD強弱フィルター付き)
    pairs = [
        # (名前, シグナル, データ, freq, htf)
        ('YagamiA_4H',      sig_yagami_A(freq='4h'),       df_4h, '4h', None),
        ('YagamiA_4H+USD',  make_usd_filtered_signal(sig_yagami_A, 75)(freq='4h'),
                                                             df_4h, '4h', None),
        ('YagamiB_4H',      sig_yagami_B(freq='4h'),       df_4h, '4h', None),
        ('YagamiB_4H+USD',  make_usd_filtered_signal(sig_yagami_B, 75)(freq='4h'),
                                                             df_4h, '4h', None),
        ('YagamiFull_1H',   sig_yagami_full_filter(freq='1h'), df_1h, '1h', df_4h),
        ('YagamiFull_1H+USD', make_usd_filtered_signal(sig_yagami_full_filter, 75)(freq='1h'),
                                                             df_1h, '1h', df_4h),
        ('MaedaiDC15_4H',   sig_maedai_dc_ema_tf(freq='4h', lookback_days=15), df_4h, '4h', None),
        ('MaedaiDC15_4H+USD', make_usd_filtered_signal(
            sig_maedai_dc_ema_tf, 75)(freq='4h', lookback_days=15), df_4h, '4h', None),
        ('MaedaiUnion_4H',  sig_maedai_yagami_union(freq='4h'), df_4h, '4h', None),
        ('MaedaiUnion_4H+USD', make_usd_filtered_signal(
            sig_maedai_yagami_union, 75)(freq='4h'), df_4h, '4h', None),
    ]

    results = []
    for name, sig_fn, data, freq, htf in pairs:
        r = engine.run(data=data, signal_func=sig_fn, freq=freq,
                       name=name, htf_bars=htf)
        if r is None or r.get('total_trades', 0) == 0:
            print(f'  {name:30s} → トレードなし')
            continue
        results.append(r)
        usd_tag = ' (+USD)' if '+USD' in name else ''
        print(f'  {name:30s}  N={r["total_trades"]:>4d}  '
              f'WR={r["win_rate_pct"]:5.1f}%  PF={r["profit_factor"]:6.3f}  '
              f'DD={r["max_drawdown_pct"]:5.1f}%  '
              f'Sharpe={fmt(r.get("sharpe_ratio"), 3):>7s}  '
              f'Calmar={fmt(r.get("calmar_ratio"), 3):>7s}{usd_tag}')

    # 比較サマリー
    if results:
        print('\n  ── フィルター効果サマリー ──')
        base = [r for r in results if '+USD' not in r['strategy']]
        filt = [r for r in results if '+USD' in r['strategy']]
        if base and filt:
            avg_pf_base = np.mean([r['profit_factor'] for r in base])
            avg_pf_filt = np.mean([r['profit_factor'] for r in filt])
            avg_dd_base = np.mean([r['max_drawdown_pct'] for r in base])
            avg_dd_filt = np.mean([r['max_drawdown_pct'] for r in filt])
            avg_sh_base = np.mean([r.get('sharpe_ratio') or 0 for r in base])
            avg_sh_filt = np.mean([r.get('sharpe_ratio') or 0 for r in filt])
            print(f'    平均PF:     素 {avg_pf_base:.3f}  →  +USD {avg_pf_filt:.3f}  '
                  f'(差 {avg_pf_filt - avg_pf_base:+.3f})')
            print(f'    平均DD:     素 {avg_dd_base:.1f}%  →  +USD {avg_dd_filt:.1f}%  '
                  f'(差 {avg_dd_filt - avg_dd_base:+.1f}%)')
            print(f'    平均Sharpe: 素 {avg_sh_base:.3f}  →  +USD {avg_sh_filt:.3f}  '
                  f'(差 {avg_sh_filt - avg_sh_base:+.3f})')

    return results


# ================================================================== #
#  Task 2: 季節フィルター定量評価                                      #
# ================================================================== #
def task2_seasonal(df_1h, df_4h):
    print('\n' + '=' * 70)
    print('  Task 2 [Teammate C]: 季節フィルター定量評価 (7月+9月除外)')
    print('=' * 70)

    engine = BacktestEngine(
        init_cash=5_000_000, risk_pct=0.05,
        default_sl_atr=2.0, default_tp_atr=4.0,
        pyramid_entries=0, target_max_dd=0.20,
        target_min_wr=0.35, target_rr_threshold=2.0,
        target_min_trades=15,
    )

    strategies = [
        ('YagamiA_4H',     sig_yagami_A(freq='4h'),      df_4h, '4h', None),
        ('YagamiB_4H',     sig_yagami_B(freq='4h'),      df_4h, '4h', None),
        ('YagamiFull_1H',  sig_yagami_full_filter(freq='1h'), df_1h, '1h', df_4h),
    ]

    results = []
    for name, sig_fn, data, freq, htf in strategies:
        # 全月
        r_all = engine.run(data=data, signal_func=sig_fn, freq=freq,
                           name=f'{name}_ALL', htf_bars=htf, allowed_months=SEASON_ALL)
        # 7月+9月除外
        r_skip = engine.run(data=data, signal_func=sig_fn, freq=freq,
                            name=f'{name}_SKIP79', htf_bars=htf,
                            allowed_months=SEASON_SKIP_JUL_SEP)

        if r_all is None or r_all.get('total_trades', 0) == 0:
            print(f'  {name} → トレードなし')
            continue

        trades_all = r_all.get('trades', [])
        se = seasonal_effectiveness(trades_all, skip_months=(7, 9))

        n_all  = r_all['total_trades']
        n_skip = r_skip['total_trades'] if r_skip else 0

        print(f'\n  [{name}]')
        print(f'    全月:       N={n_all:>3d}  WR={r_all["win_rate_pct"]:5.1f}%  '
              f'PF={r_all["profit_factor"]:6.3f}  DD={r_all["max_drawdown_pct"]:5.1f}%  '
              f'Sharpe={fmt(r_all.get("sharpe_ratio"), 3):>7s}')
        if r_skip and r_skip.get('total_trades', 0) > 0:
            print(f'    7+9月除外:  N={n_skip:>3d}  WR={r_skip["win_rate_pct"]:5.1f}%  '
                  f'PF={r_skip["profit_factor"]:6.3f}  DD={r_skip["max_drawdown_pct"]:5.1f}%  '
                  f'Sharpe={fmt(r_skip.get("sharpe_ratio"), 3):>7s}')

        # 月別ブレイクダウン
        print(f'    除外月PnL:  {se["excluded_months_pnl"]:>+12.0f}  '
              f'({se["excluded_months_count"]}回)')
        print(f'    残存月PnL:  {se["included_months_pnl"]:>+12.0f}  '
              f'({se["included_months_count"]}回)')

        mb = se['monthly_breakdown']
        if mb:
            print(f'    月別: ', end='')
            for m in sorted(mb.keys()):
                d = mb[m]
                wr = d['wins'] / d['count'] * 100 if d['count'] > 0 else 0
                flag = '✗' if m in (7, 9) else ' '
                print(f'{m:>2d}月({d["count"]:>2d}回 WR{wr:4.0f}% PnL{d["pnl"]:>+9.0f}){flag}  ', end='')
            print()

        results.append({
            'strategy': name,
            'all_n': n_all, 'all_pf': r_all['profit_factor'],
            'all_dd': r_all['max_drawdown_pct'],
            'all_sharpe': r_all.get('sharpe_ratio'),
            'skip_n': n_skip,
            'skip_pf': r_skip['profit_factor'] if r_skip and r_skip.get('total_trades', 0) > 0 else None,
            'skip_dd': r_skip['max_drawdown_pct'] if r_skip and r_skip.get('total_trades', 0) > 0 else None,
            'skip_sharpe': r_skip.get('sharpe_ratio') if r_skip else None,
            'excluded_pnl': se['excluded_months_pnl'],
        })

    return results


# ================================================================== #
#  Task 3: Maedai Sharpe/Calmar ランキング + DC パラメータ探索         #
# ================================================================== #
def task3_maedai_optimization(df_4h):
    print('\n' + '=' * 70)
    print('  Task 3 [Teammate B]: Maedai Sharpe/Calmar ランキング')
    print('=' * 70)

    # Maedai専用エンジン (低WR/高RR設定)
    engine = BacktestEngine(
        init_cash=5_000_000, risk_pct=0.03,
        default_sl_atr=1.5, default_tp_atr=6.0,
        use_dynamic_sl=True, sl_n_confirm=2, sl_min_atr=0.5,
        dynamic_rr=3.0,
        pyramid_entries=0,
        trail_start_atr=3.0, trail_dist_atr=1.5,
        target_max_dd=0.30, target_min_wr=0.25,
        target_rr_threshold=3.0, target_min_trades=15,
    )

    # Donchian パラメータグリッド
    dc_params = [
        {'lookback_days': 10, 'ema_days': 200, 'label': 'DC10_EMA200'},
        {'lookback_days': 15, 'ema_days': 200, 'label': 'DC15_EMA200'},
        {'lookback_days': 20, 'ema_days': 200, 'label': 'DC20_EMA200'},
        {'lookback_days': 30, 'ema_days': 200, 'label': 'DC30_EMA200'},
        {'lookback_days': 40, 'ema_days': 200, 'label': 'DC40_EMA200'},
        {'lookback_days': 15, 'ema_days': 100, 'label': 'DC15_EMA100'},
        {'lookback_days': 20, 'ema_days': 100, 'label': 'DC20_EMA100'},
        {'lookback_days': 30, 'ema_days': 100, 'label': 'DC30_EMA100'},
    ]

    results = []
    for p in dc_params:
        sig = sig_maedai_dc_ema_tf(
            freq='4h',
            lookback_days=p['lookback_days'],
            ema_days=p['ema_days'],
        )
        r = engine.run(data=df_4h, signal_func=sig, freq='4h', name=p['label'])
        if r is None or r.get('total_trades', 0) == 0:
            print(f'  {p["label"]:20s} → トレードなし')
            continue
        results.append(r)
        print(f'  {p["label"]:20s}  N={r["total_trades"]:>3d}  '
              f'WR={r["win_rate_pct"]:5.1f}%  PF={r["profit_factor"]:6.3f}  '
              f'RR={r["rr_ratio"]:5.2f}  DD={r["max_drawdown_pct"]:5.1f}%  '
              f'Sharpe={fmt(r.get("sharpe_ratio"), 3):>7s}  '
              f'Calmar={fmt(r.get("calmar_ratio"), 3):>7s}')

    # + Union 戦略
    union_sig = sig_maedai_yagami_union(freq='4h')
    r_union = engine.run(data=df_4h, signal_func=union_sig, freq='4h', name='Union')
    if r_union and r_union.get('total_trades', 0) > 0:
        results.append(r_union)
        print(f'  {"Union":20s}  N={r_union["total_trades"]:>3d}  '
              f'WR={r_union["win_rate_pct"]:5.1f}%  PF={r_union["profit_factor"]:6.3f}  '
              f'RR={r_union["rr_ratio"]:5.2f}  DD={r_union["max_drawdown_pct"]:5.1f}%  '
              f'Sharpe={fmt(r_union.get("sharpe_ratio"), 3):>7s}  '
              f'Calmar={fmt(r_union.get("calmar_ratio"), 3):>7s}')

    best_sig = sig_maedai_best(freq='4h')
    r_best = engine.run(data=df_4h, signal_func=best_sig, freq='4h', name='MaedaiBest')
    if r_best and r_best.get('total_trades', 0) > 0:
        results.append(r_best)
        print(f'  {"MaedaiBest":20s}  N={r_best["total_trades"]:>3d}  '
              f'WR={r_best["win_rate_pct"]:5.1f}%  PF={r_best["profit_factor"]:6.3f}  '
              f'RR={r_best["rr_ratio"]:5.2f}  DD={r_best["max_drawdown_pct"]:5.1f}%  '
              f'Sharpe={fmt(r_best.get("sharpe_ratio"), 3):>7s}  '
              f'Calmar={fmt(r_best.get("calmar_ratio"), 3):>7s}')

    # Sharpe ランキング
    if results:
        print('\n  ── Sharpe Ratio ランキング ──')
        ranked = sorted(results, key=lambda r: r.get('sharpe_ratio') or -999, reverse=True)
        for i, r in enumerate(ranked[:5]):
            print(f'    #{i+1}  {r["strategy"]:20s}  Sharpe={fmt(r.get("sharpe_ratio"), 3):>7s}  '
                  f'Calmar={fmt(r.get("calmar_ratio"), 3):>7s}  '
                  f'PF={r["profit_factor"]:6.3f}  DD={r["max_drawdown_pct"]:5.1f}%')

    return results


# ================================================================== #
#  可視化                                                              #
# ================================================================== #
def plot_portfolio_comparison(t1_results, t2_results, t3_results, outpath):
    """3タスクの比較結果を1枚のチャートに集約"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('戦略ポートフォリオ 統合評価レポート',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    bg = '#161b22'

    # Task 1: USD filter comparison
    ax = axes[0]
    ax.set_facecolor(bg)
    ax.set_title('Task1: USD強弱フィルター効果', color='white', fontsize=10)
    if t1_results:
        base = [r for r in t1_results if '+USD' not in r['strategy']]
        filt = [r for r in t1_results if '+USD' in r['strategy']]
        metrics = ['profit_factor', 'max_drawdown_pct']
        m_labels = ['PF', 'DD(%)']
        x = np.arange(len(m_labels))
        w = 0.35
        base_vals = [np.mean([r[m] for r in base]) if base else 0 for m in metrics]
        filt_vals = [np.mean([r[m] for r in filt]) if filt else 0 for m in metrics]
        ax.bar(x - w/2, base_vals, w, label='素', color='#58a6ff', alpha=0.85)
        ax.bar(x + w/2, filt_vals, w, label='+USD filter', color='#f85149', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(m_labels, color='white')
        for xi, (bv, fv) in enumerate(zip(base_vals, filt_vals)):
            ax.text(xi - w/2, bv, f'{bv:.2f}', ha='center', va='bottom', fontsize=8, color='white')
            ax.text(xi + w/2, fv, f'{fv:.2f}', ha='center', va='bottom', fontsize=8, color='white')
    ax.tick_params(colors='white', labelsize=8)
    ax.spines[:].set_color('#30363d')
    ax.legend(fontsize=8, facecolor=bg, edgecolor='#30363d', labelcolor='white')

    # Task 2: Seasonal
    ax = axes[1]
    ax.set_facecolor(bg)
    ax.set_title('Task2: 季節フィルター (7+9月除外)', color='white', fontsize=10)
    if t2_results:
        names = [r['strategy'] for r in t2_results]
        x = np.arange(len(names))
        w = 0.35
        all_pf = [r['all_pf'] for r in t2_results]
        skip_pf = [r['skip_pf'] or 0 for r in t2_results]
        ax.bar(x - w/2, all_pf, w, label='全月', color='#58a6ff', alpha=0.85)
        ax.bar(x + w/2, skip_pf, w, label='7+9月除外', color='#26a641', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=7, color='white')
        for xi, (av, sv) in enumerate(zip(all_pf, skip_pf)):
            ax.text(xi - w/2, av, f'{av:.2f}', ha='center', va='bottom', fontsize=7, color='white')
            ax.text(xi + w/2, sv, f'{sv:.2f}', ha='center', va='bottom', fontsize=7, color='white')
    ax.tick_params(colors='white', labelsize=8)
    ax.spines[:].set_color('#30363d')
    ax.legend(fontsize=8, facecolor=bg, edgecolor='#30363d', labelcolor='white')

    # Task 3: Maedai Sharpe ranking
    ax = axes[2]
    ax.set_facecolor(bg)
    ax.set_title('Task3: Maedai Sharpe Ranking', color='white', fontsize=10)
    if t3_results:
        ranked = sorted(t3_results, key=lambda r: r.get('sharpe_ratio') or -999, reverse=True)[:8]
        names = [r['strategy'] for r in ranked]
        sharpes = [r.get('sharpe_ratio') or 0 for r in ranked]
        colors = ['#26a641' if s > 0 else '#f85149' for s in sharpes]
        y = np.arange(len(names))
        ax.barh(y, sharpes, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7, color='white')
        for yi, s in zip(y, sharpes):
            ax.text(s, yi, f' {s:.2f}', va='center', fontsize=7, color='white')
        ax.axvline(0, color='#30363d', lw=0.8)
    ax.tick_params(colors='white', labelsize=8)
    ax.spines[:].set_color('#30363d')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'\n  チャート保存: {outpath}')


# ================================================================== #
#  メイン                                                              #
# ================================================================== #
def main():
    print('=' * 70)
    print('  戦略ポートフォリオ バックテスト比較')
    print('  (Yagami × Maedai × Risk Manager)')
    print('=' * 70)

    df_1h = load_ohlc(OHLC_1H)
    df_4h = load_ohlc(OHLC_4H)
    print(f'\n  1H: {len(df_1h)} 本  4H: {len(df_4h)} 本')

    t1 = task1_usd_filter(df_1h, df_4h)
    t2 = task2_seasonal(df_1h, df_4h)
    t3 = task3_maedai_optimization(df_4h)

    # 可視化
    img_path = os.path.join(RESULTS, 'portfolio_comparison.png')
    plot_portfolio_comparison(t1, t2, t3, img_path)

    # CSV保存
    all_rows = []
    for r in (t1 or []):
        all_rows.append({
            'task': 'T1_USD', 'strategy': r['strategy'],
            'trades': r['total_trades'], 'wr': r['win_rate_pct'],
            'pf': r['profit_factor'], 'rr': r['rr_ratio'],
            'dd': r['max_drawdown_pct'],
            'sharpe': r.get('sharpe_ratio'), 'calmar': r.get('calmar_ratio'),
        })
    for r in (t3 or []):
        all_rows.append({
            'task': 'T3_Maedai', 'strategy': r['strategy'],
            'trades': r['total_trades'], 'wr': r['win_rate_pct'],
            'pf': r['profit_factor'], 'rr': r['rr_ratio'],
            'dd': r['max_drawdown_pct'],
            'sharpe': r.get('sharpe_ratio'), 'calmar': r.get('calmar_ratio'),
        })
    if all_rows:
        csv_path = os.path.join(RESULTS, 'portfolio_comparison.csv')
        pd.DataFrame(all_rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'  CSV保存: {csv_path}')

    print(f'\n{"="*70}')
    print(f'  完了')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()
