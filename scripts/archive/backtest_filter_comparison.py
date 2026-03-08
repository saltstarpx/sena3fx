"""
フィルター比較バックテスト: USD強弱 × 季節性 × サイズ調整
============================================================
シルバー (XAGUSD) を対象に5つのシナリオを比較検証する:

  A: ベースライン        — 追加フィルターなし
  B: USDフィルター       — USD強すぎる時はロング回避
  C: 季節フィルター      — 7月・9月をスキップ
  D: 複合フィルター      — USD + 季節 を両方適用
  E: 複合 + サイズ調整   — D + USDが中程度に強い時は30%縮小

設定: risk_pct=3.5% (リスク最適化推奨値), 期間=2025-09〜2026-02
実行: python scripts/backtest_filter_comparison.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from scripts.backtest_dual_tf import load_ohlc, run_backtest, quantitative_analysis
from scripts.usd_strength_filter import (
    calc_usd_strength,
    get_usd_filter_mask,
    get_seasonal_filter_mask,
    get_usd_scale_series,
)

# ================================================================== #
#  設定                                                               #
# ================================================================== #
START     = '2025-09-01'
END       = '2026-02-28'
INIT_CASH = 10_000_000
RISK_PCT  = 0.035          # リスク最適化推奨値 (3.5%)

USD_BLOCK_THRESHOLD  = 1.0   # USD強弱スコアがこれ以上なら→ロングブロック
USD_SCALE_THRESHOLD  = 0.5   # これ以上→ポジションを 70% に縮小
SIZE_SCALE           = 0.7   # 縮小率 (30%減)
SKIP_MONTHS          = (7, 9) # 季節フィルター: 7月・9月をスキップ

BACKTEST_PARAMS = dict(
    instrument     = 'XAG_USD',
    init_cash      = INIT_CASH,
    risk_pct       = RISK_PCT,
    ema_days       = 21,
    dc_lookback    = 20,
    swing_window   = 3,
    swing_lookback = 10,
    rr_target      = 2.5,
    min_rr         = 2.0,
    long_only      = True,
    start_date     = START,
    end_date       = END,
)


# ================================================================== #
#  シナリオ定義                                                       #
# ================================================================== #
def build_scenarios(bars_15m, usd_strength):
    """
    各シナリオの (extra_filter, risk_scale_series) を返す。
    bars_15m が必要なのは target_index の取得のため。
    """
    idx = bars_15m.loc[START:END].index

    # --- USD フィルターマスク ---
    usd_mask  = get_usd_filter_mask(usd_strength, idx, USD_BLOCK_THRESHOLD)

    # --- 季節フィルターマスク ---
    seas_mask = get_seasonal_filter_mask(idx, SKIP_MONTHS)

    # --- USD スケールシリーズ ---
    usd_scale = get_usd_scale_series(
        usd_strength, idx, USD_SCALE_THRESHOLD, SIZE_SCALE
    )

    scenarios = [
        {
            'label':       'A: ベースライン',
            'tag':         'baseline',
            'extra_filter': None,
            'risk_scale':   None,
            'color':        '#58a6ff',
        },
        {
            'label':       'B: USDフィルター',
            'tag':         'usd_filter',
            'extra_filter': usd_mask,
            'risk_scale':   None,
            'color':        '#f0a500',
        },
        {
            'label':       'C: 季節フィルター (7・9月スキップ)',
            'tag':         'seasonal',
            'extra_filter': seas_mask,
            'risk_scale':   None,
            'color':        '#3fb950',
        },
        {
            'label':       'D: 複合 (USD + 季節)',
            'tag':         'combined',
            'extra_filter': usd_mask & seas_mask,
            'risk_scale':   None,
            'color':        '#bc8cff',
        },
        {
            'label':       'E: 複合 + サイズ調整',
            'tag':         'combined_scale',
            'extra_filter': usd_mask & seas_mask,
            'risk_scale':   usd_scale,
            'color':        '#ff7b72',
        },
    ]
    return scenarios


# ================================================================== #
#  全シナリオ実行                                                     #
# ================================================================== #
def run_all_scenarios(b4, b15, scenarios):
    results = []
    for sc in scenarios:
        print(f"\n{'='*60}")
        print(f"  シナリオ: {sc['label']}")
        print(f"{'='*60}")

        trades, equity, cash = run_backtest(
            b4, b15,
            extra_filter      = sc['extra_filter'],
            risk_scale_series = sc['risk_scale'],
            **BACKTEST_PARAMS,
        )
        stats = quantitative_analysis(trades, equity, INIT_CASH, 'XAG_USD')

        results.append({
            'scenario': sc['label'],
            'tag':      sc['tag'],
            'color':    sc['color'],
            'stats':    stats,
            'equity':   equity,
            'cash':     cash,
            'trades':   trades,
        })
    return results


# ================================================================== #
#  サマリー表示                                                       #
# ================================================================== #
def print_summary(results):
    print("\n" + "=" * 80)
    print(f"  フィルター比較結果サマリー — Silver (XAG_USD)")
    print(f"  期間: {START} 〜 {END}  /  初期資金: ¥{INIT_CASH:,}  /  risk_pct: {RISK_PCT*100:.1f}%")
    print("=" * 80)
    hdr = f"{'シナリオ':<30} {'取引数':>5} {'勝率':>7} {'PF':>6} "
    hdr += f"{'ROI':>8} {'MaxDD':>7} {'Calmar':>8} {'Sharpe':>8} {'最終資産':>14}"
    print(hdr)
    print("-" * 80)

    base_roi = None
    for r in results:
        s = r['stats']
        if not s:
            print(f"  {r['scenario']:<28} — トレードなし")
            continue
        if base_roi is None:
            base_roi = s['roi']
        diff = s['roi'] - base_roi
        diff_str = f"({diff:+.1f}%)" if diff != 0 else ""
        print(
            f"  {r['scenario']:<28} "
            f"{s['n_trades']:>5} "
            f"{s['win_rate']:>6.1f}% "
            f"{s['profit_factor']:>6.2f} "
            f"{s['roi']:>+7.1f}% {diff_str:<8} "
            f"{s['max_dd']:>6.1f}% "
            f"{s['calmar']:>8.2f} "
            f"{s['sharpe']:>8.2f} "
            f"¥{r['cash']:>13,.0f}"
        )
    print("=" * 80)


# ================================================================== #
#  チャート生成                                                       #
# ================================================================== #
def plot_comparison(results, usd_strength, outpath):
    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor('#0d1117')

    gs = gridspec.GridSpec(
        6, 2,
        figure=fig,
        hspace=0.50, wspace=0.35,
        top=0.93, bottom=0.04, left=0.07, right=0.97
    )

    GREEN = '#26a641'
    RED   = '#f85149'

    fig.text(
        0.5, 0.965,
        'フィルター比較バックテスト レポート\n'
        f'Silver (XAG_USD) — USD強弱フィルター × 季節フィルター × サイズ調整\n'
        f'期間: {START} 〜 {END}  /  初期資金: ¥{INIT_CASH:,}  /  risk_pct: {RISK_PCT*100:.1f}%  '
        f'USD閾値: {USD_BLOCK_THRESHOLD}  季節スキップ: {SKIP_MONTHS}',
        ha='center', va='top', color='white',
        fontsize=12, fontweight='bold', linespacing=1.6,
    )

    # --- [0, :] エクイティカーブ全シナリオ ---
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.set_facecolor('#161b22')
    ax_eq.set_title('エクイティカーブ比較', color='white', fontsize=11, pad=6)
    ax_eq.axhline(INIT_CASH / 1e6, color='white', lw=0.8, ls='--', alpha=0.4)

    for r in results:
        s = r['stats']
        if not s:
            continue
        lbl = f"{r['scenario']}  ROI:{s['roi']:+.1f}%  DD:{s['max_dd']:.1f}%"
        ax_eq.plot(r['equity'].index, r['equity'].values / 1e6,
                   color=r['color'], lw=1.8, label=lbl, alpha=0.9)

    ax_eq.set_ylabel('資産 (百万円)', color='white', fontsize=9)
    ax_eq.tick_params(colors='white', labelsize=8)
    ax_eq.spines[:].set_color('#30363d')
    ax_eq.legend(fontsize=8.5, facecolor='#161b22', edgecolor='#30363d',
                 labelcolor='white', loc='upper left')

    # --- [1, :] USD強弱プロキシ時系列 ---
    ax_usd = fig.add_subplot(gs[1, :])
    ax_usd.set_facecolor('#161b22')
    ax_usd.set_title(
        f'USD強弱プロキシ (XAUUSD 20日リターン逆Z / ブロック閾値={USD_BLOCK_THRESHOLD})',
        color='white', fontsize=10, pad=6,
    )
    usd_plot = usd_strength.loc[START:END].dropna()
    col_vals  = ['#f85149' if v >= USD_BLOCK_THRESHOLD
                 else '#f0a500' if v >= USD_SCALE_THRESHOLD
                 else '#58a6ff' for v in usd_plot.values]
    ax_usd.bar(usd_plot.index, usd_plot.values, color=col_vals, alpha=0.7, width=1)
    ax_usd.axhline(USD_BLOCK_THRESHOLD,  color='#f85149', lw=1.2, ls='--', alpha=0.9,
                   label=f'ブロック閾値 ({USD_BLOCK_THRESHOLD})')
    ax_usd.axhline(USD_SCALE_THRESHOLD, color='#f0a500', lw=1.0, ls=':', alpha=0.8,
                   label=f'縮小閾値 ({USD_SCALE_THRESHOLD})')
    ax_usd.axhline(0, color='white', lw=0.6, alpha=0.4)
    ax_usd.set_ylabel('USD強弱スコア', color='white', fontsize=9)
    ax_usd.tick_params(colors='white', labelsize=8)
    ax_usd.spines[:].set_color('#30363d')
    ax_usd.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # --- [2, 0] ROI棒グラフ ---
    ax_roi = fig.add_subplot(gs[2, 0])
    ax_roi.set_facecolor('#161b22')
    ax_roi.set_title('ROI 比較', color='white', fontsize=10, pad=5)

    labels = [r['scenario'].split(':')[0] for r in results if r['stats']]
    rois   = [r['stats']['roi'] for r in results if r['stats']]
    cols   = [r['color'] for r in results if r['stats']]
    bars   = ax_roi.bar(labels, rois, color=cols, alpha=0.85)
    for bar, v in zip(bars, rois):
        ax_roi.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{v:+.1f}%', ha='center', va='bottom',
                    color='white', fontsize=9, fontweight='bold')
    ax_roi.set_ylabel('ROI (%)', color='white', fontsize=9)
    ax_roi.tick_params(colors='white', labelsize=9)
    ax_roi.spines[:].set_color('#30363d')
    ax_roi.axhline(0, color='white', lw=0.6, alpha=0.4)

    # --- [2, 1] MaxDD棒グラフ ---
    ax_dd = fig.add_subplot(gs[2, 1])
    ax_dd.set_facecolor('#161b22')
    ax_dd.set_title('MaxDD 比較', color='white', fontsize=10, pad=5)

    dds  = [r['stats']['max_dd'] for r in results if r['stats']]
    brs  = ax_dd.bar(labels, dds, color=cols, alpha=0.85)
    for bar, v in zip(brs, dds):
        ax_dd.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.2,
                   f'{v:.1f}%', ha='center', va='bottom',
                   color='white', fontsize=9, fontweight='bold')
    ax_dd.set_ylabel('MaxDD (%)', color='white', fontsize=9)
    ax_dd.tick_params(colors='white', labelsize=9)
    ax_dd.spines[:].set_color('#30363d')
    ax_dd.axhline(15, color='#f85149', lw=1, ls='--', alpha=0.7, label='DD≤15%基準')
    ax_dd.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # --- [3, 0] Calmar棒グラフ ---
    ax_cal = fig.add_subplot(gs[3, 0])
    ax_cal.set_facecolor('#161b22')
    ax_cal.set_title('Calmar比 (年率ROI / MaxDD)', color='white', fontsize=10, pad=5)

    calmars = [r['stats']['calmar'] for r in results if r['stats']]
    brc = ax_cal.bar(labels, calmars, color=cols, alpha=0.85)
    for bar, v in zip(brc, calmars):
        ax_cal.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    f'{v:.2f}', ha='center', va='bottom',
                    color='white', fontsize=9, fontweight='bold')
    ax_cal.set_ylabel('Calmar', color='white', fontsize=9)
    ax_cal.tick_params(colors='white', labelsize=9)
    ax_cal.spines[:].set_color('#30363d')

    # --- [3, 1] Sharpe棒グラフ ---
    ax_sh = fig.add_subplot(gs[3, 1])
    ax_sh.set_facecolor('#161b22')
    ax_sh.set_title('Sharpe比', color='white', fontsize=10, pad=5)

    sharpes = [r['stats']['sharpe'] for r in results if r['stats']]
    brs2 = ax_sh.bar(labels, sharpes, color=cols, alpha=0.85)
    for bar, v in zip(brs2, sharpes):
        ax_sh.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.02,
                   f'{v:.2f}', ha='center', va='bottom',
                   color='white', fontsize=9, fontweight='bold')
    ax_sh.set_ylabel('Sharpe', color='white', fontsize=9)
    ax_sh.tick_params(colors='white', labelsize=9)
    ax_sh.spines[:].set_color('#30363d')

    # --- [4, :] 統計テーブル ---
    ax_tbl = fig.add_subplot(gs[4, :])
    ax_tbl.set_facecolor('#161b22')
    ax_tbl.axis('off')
    ax_tbl.set_title('全シナリオ統計比較', color='white', fontsize=10, pad=5)

    col_hdr = ['シナリオ', '取引数', '勝率', 'PF', 'ROI', '年率ROI',
               'MaxDD', 'Calmar', 'Sharpe', 'Sortino', '期待値', '平均保有']
    rows = []
    for r in results:
        s = r['stats']
        if not s:
            rows.append([r['scenario'], '—'] + ['—'] * 10)
            continue
        rows.append([
            r['scenario'],
            str(s['n_trades']),
            f"{s['win_rate']:.1f}%",
            f"{s['profit_factor']:.2f}",
            f"{s['roi']:+.1f}%",
            f"{s['annual_roi']:+.1f}%",
            f"{s['max_dd']:.1f}%",
            f"{s['calmar']:.2f}",
            f"{s['sharpe']:.2f}",
            f"{s['sortino']:.2f}",
            f"${s['expectancy']:,.0f}",
            f"{s['avg_hold_h']:.1f}h",
        ])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_hdr,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    sc_colors = {r['tag']: r['color'] for r in results}
    sc_tags   = [r['tag'] for r in results]
    for (row_i, col_i), cell in tbl.get_celld().items():
        cell.set_edgecolor('#30363d')
        cell.set_text_props(color='white')
        if row_i == 0:
            cell.set_facecolor('#0d1117')
            cell.set_text_props(color='#58a6ff', fontweight='bold')
        elif 1 <= row_i <= len(results):
            tag = sc_tags[row_i - 1]
            c   = sc_colors.get(tag, '#1c2128')
            cell.set_facecolor(f'{c}22')  # 透明度付き背景
            if col_i == 0:
                cell.set_text_props(color=c, fontweight='bold')
        else:
            cell.set_facecolor('#1c2128')

    # --- [5, :] 月次PnL比較 (ベースライン vs D複合) ---
    ax_mp = fig.add_subplot(gs[5, :])
    ax_mp.set_facecolor('#161b22')
    ax_mp.set_title(
        '月次PnL% 比較 (A: ベースライン  vs  D: 複合フィルター)',
        color='white', fontsize=10, pad=5,
    )

    base_r = next((r for r in results if r['tag'] == 'baseline'), None)
    comb_r = next((r for r in results if r['tag'] == 'combined'), None)

    # 共通月インデックスを作成してアライン
    all_months = pd.period_range(START, END, freq='M')
    months_str = [str(m) for m in all_months]
    x_pos = np.arange(len(all_months))

    if base_r and base_r['stats']:
        mp  = base_r['stats']['monthly_pnl_pct'].reindex(all_months, fill_value=0)
        ax_mp.bar(x_pos - 0.2, mp.values, 0.38,
                  color=['#26a641' if v >= 0 else '#f85149' for v in mp.values],
                  alpha=0.75, label='A: ベースライン')

    if comb_r and comb_r['stats']:
        mp2 = comb_r['stats']['monthly_pnl_pct'].reindex(all_months, fill_value=0)
        ax_mp.bar(x_pos + 0.2, mp2.values, 0.38,
                  color=['#bc8cff' if v >= 0 else '#ff7b72' for v in mp2.values],
                  alpha=0.75, label='D: 複合フィルター')

    ax_mp.set_xticks(x_pos)
    ax_mp.set_xticklabels(months_str, color='white', fontsize=8, rotation=30)
    ax_mp.axhline(0, color='white', lw=0.6, alpha=0.5)
    ax_mp.set_ylabel('PnL (%)', color='white', fontsize=9)
    ax_mp.tick_params(colors='white', labelsize=8)
    ax_mp.spines[:].set_color('#30363d')
    ax_mp.legend(fontsize=8.5, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"\nチャート保存: {outpath}")


# ================================================================== #
#  メイン                                                             #
# ================================================================== #
def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --- データ読み込み ---
    print("データ読み込み中...")
    b4  = load_ohlc(os.path.join(root, 'data', 'ohlc', 'XAGUSD_2025_4h.csv'))
    b15 = load_ohlc(os.path.join(root, 'data', 'ohlc', 'XAGUSD_2025_15m.csv'))

    # --- USD強弱プロキシ計算 ---
    print("USD強弱プロキシ計算中...")
    usd_strength = calc_usd_strength(lookback=20, data_root=root)
    blocked_days = (usd_strength.dropna() >= USD_BLOCK_THRESHOLD).sum()
    total_days   = usd_strength.dropna().loc[START:END]
    print(f"  期間内でUSDブロック: {(total_days >= USD_BLOCK_THRESHOLD).sum()} / "
          f"{len(total_days)} 営業日 "
          f"({(total_days >= USD_BLOCK_THRESHOLD).sum() / max(len(total_days),1) * 100:.1f}%)")

    # --- シナリオ構築 ---
    scenarios = build_scenarios(b15, usd_strength)

    # --- 全シナリオ実行 ---
    results = run_all_scenarios(b4, b15, scenarios)

    # --- サマリー表示 ---
    print_summary(results)

    # --- ベスト選出 ---
    valid = [(r, r['stats']) for r in results if r['stats']]
    if valid:
        best_calmar = max(valid, key=lambda x: x[1]['calmar'])
        best_dd     = min(valid, key=lambda x: x[1]['max_dd'])
        best_roi    = max(valid, key=lambda x: x[1]['roi'])
        safe_dd15   = [x for x in valid if x[1]['max_dd'] <= 15]
        best_safe   = max(safe_dd15, key=lambda x: x[1]['calmar']) if safe_dd15 else None

        print(f"\n{'★':>2} Calmar最大   : {best_calmar[0]['scenario']}"
              f" → Calmar={best_calmar[1]['calmar']:.2f}"
              f"  ROI={best_calmar[1]['roi']:+.1f}%"
              f"  DD={best_calmar[1]['max_dd']:.1f}%")
        print(f"{'★':>2} ROI最大       : {best_roi[0]['scenario']}"
              f" → ROI={best_roi[1]['roi']:+.1f}%"
              f"  DD={best_roi[1]['max_dd']:.1f}%")
        print(f"{'★':>2} DD最小        : {best_dd[0]['scenario']}"
              f" → DD={best_dd[1]['max_dd']:.1f}%"
              f"  ROI={best_dd[1]['roi']:+.1f}%")
        if best_safe:
            print(f"{'★':>2} DD≤15%で推奨  : {best_safe[0]['scenario']}"
                  f" → ROI={best_safe[1]['roi']:+.1f}%"
                  f"  DD={best_safe[1]['max_dd']:.1f}%"
                  f"  Calmar={best_safe[1]['calmar']:.2f}")

    # --- チャート出力 ---
    os.makedirs(os.path.join(root, 'results'), exist_ok=True)
    outpath = os.path.join(root, 'results', 'filter_comparison.png')
    plot_comparison(results, usd_strength, outpath)
    print(f"\nチャート: results/filter_comparison.png")


if __name__ == '__main__':
    main()
