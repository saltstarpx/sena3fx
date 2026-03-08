"""
Silver (XAG_USD) リスク率 最適化スキャン
==========================================
デュアルTF戦略のリスク率を 0.5% 〜 10% で細分化して比較。

最適化指標:
  - Calmar比率 (年率ROI / MaxDD) ← メイン
  - Sharpe / Sortino
  - 最終資産 / MaxDD トレードオフ

実行: python scripts/risk_optimization_silver.py
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

# backtest_dual_tf から共通関数をインポート
from scripts.backtest_dual_tf import (
    load_ohlc, run_backtest, quantitative_analysis
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ #
#  スキャン設定                                                       #
# ------------------------------------------------------------------ #
RISK_LEVELS = [
    0.005, 0.010, 0.015, 0.020, 0.025, 0.030,
    0.035, 0.040, 0.045, 0.050, 0.060, 0.070,
    0.080, 0.090, 0.100,
]

BASE_PARAMS = dict(
    ema_days       = 21,
    dc_lookback    = 20,
    swing_window   = 3,
    swing_lookback = 10,
    rr_target      = 2.5,
    min_rr         = 2.0,
    long_only      = True,
    start_date     = '2025-09-01',
    end_date       = '2026-02-28',
)

INIT_CASH = 10_000_000

# ------------------------------------------------------------------ #
#  スキャン実行                                                       #
# ------------------------------------------------------------------ #
def run_risk_scan():
    print("=" * 65)
    print("  Silver (XAG_USD) リスク率最適化スキャン")
    print(f"  期間: 2025-09 〜 2026-02  /  初期資金: ¥{INIT_CASH:,}")
    print(f"  スキャン対象: {len(RISK_LEVELS)} レベル")
    print("=" * 65)

    # データを一度だけ読み込む
    b4  = load_ohlc(os.path.join(ROOT, 'data/ohlc/XAGUSD_2025_4h.csv'))
    b15 = load_ohlc(os.path.join(ROOT, 'data/ohlc/XAGUSD_2025_15m.csv'))

    results = []

    for risk_pct in RISK_LEVELS:
        trades, eq, final_cash = run_backtest(
            b4, b15,
            instrument  = 'XAG_USD',
            init_cash   = INIT_CASH,
            risk_pct    = risk_pct,
            **BASE_PARAMS,
        )
        if not trades:
            continue

        stats = quantitative_analysis(trades, eq, INIT_CASH, 'XAG_USD')
        results.append({
            'risk_pct'    : risk_pct,
            'risk_pct_pct': risk_pct * 100,
            'n_trades'    : stats['n_trades'],
            'win_rate'    : stats['win_rate'],
            'pf'          : stats['profit_factor'],
            'roi'         : stats['roi'],
            'annual_roi'  : stats['annual_roi'],
            'final_cash'  : final_cash,
            'max_dd'      : stats['max_dd'],
            'sharpe'      : stats['sharpe'],
            'sortino'     : stats['sortino'],
            'calmar'      : stats['calmar'],
            'expectancy'  : stats['expectancy'],
            'avg_hold_h'  : stats['avg_hold_h'],
            'max_win_s'   : stats['max_win_streak'],
            'max_loss_s'  : stats['max_loss_streak'],
            'monthly_pnl' : stats['monthly_pnl_pct'],
            'equity_curve': eq,
        })
        pct_str = f"{risk_pct*100:.1f}%"
        print(f"  [{pct_str:>5}]  ROI:{stats['roi']:+.1f}%  "
              f"DD:{stats['max_dd']:.1f}%  "
              f"Calmar:{stats['calmar']:.2f}  "
              f"Sharpe:{stats['sharpe']:.2f}  "
              f"PF:{stats['profit_factor']:.2f}")

    return results

# ------------------------------------------------------------------ #
#  チャート生成                                                       #
# ------------------------------------------------------------------ #
def plot_risk_optimization(results, outpath):
    df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ('monthly_pnl', 'equity_curve')}
                        for r in results])

    SILVER = '#C0C0C0'
    GOLD   = '#FFD700'
    GREEN  = '#26a641'
    RED    = '#f85149'
    BLUE   = '#58a6ff'
    CYAN   = '#3fb950'
    ORANGE = '#e3b341'

    # 最適点を特定
    best_calmar_idx = df['calmar'].idxmax()
    best_sharpe_idx = df['sharpe'].idxmax()
    best_roi_idx    = df['roi'].idxmax()
    # MaxDD ≤ 15% 制約下での最高ROI
    safe_df = df[df['max_dd'] <= 15.0]
    best_safe_idx = safe_df['roi'].idxmax() if len(safe_df) > 0 else best_calmar_idx

    best_calmar_risk = df.loc[best_calmar_idx, 'risk_pct_pct']
    best_sharpe_risk = df.loc[best_sharpe_idx, 'risk_pct_pct']
    best_roi_risk    = df.loc[best_roi_idx,    'risk_pct_pct']
    best_safe_risk   = df.loc[best_safe_idx,   'risk_pct_pct']

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor('#0d1117')

    gs = gridspec.GridSpec(
        5, 2, figure=fig,
        hspace=0.40, wspace=0.35,
        top=0.93, bottom=0.04, left=0.07, right=0.97
    )

    x = df['risk_pct_pct'].values

    def ax_style(ax, title, ylabel='', xlabel='リスク率 (%)'):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color='white', fontsize=11, pad=6)
        ax.set_xlabel(xlabel, color='white', fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        ax.spines[:].set_color('#30363d')
        ax.grid(axis='y', color='#30363d', lw=0.5, alpha=0.7)

    def mark_best(ax, risk_val, color, label):
        ax.axvline(risk_val, color=color, lw=1.5, ls='--', alpha=0.85)
        ylim = ax.get_ylim()
        ax.text(risk_val + 0.1, ylim[1] * 0.95,
                label, color=color, fontsize=7.5, fontweight='bold', va='top')

    # ---- タイトル ----
    fig.text(
        0.5, 0.965,
        'Silver (XAG_USD) リスク率最適化スキャン\n'
        '4H EMA21 × 15m DC20 + スイングSL | 期間: 2025-09 〜 2026-02',
        ha='center', va='top', color='white',
        fontsize=13, fontweight='bold', linespacing=1.6
    )

    # ---- [0,:] エクイティカーブ比較 ----
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.set_facecolor('#161b22')
    ax_eq.set_title('エクイティカーブ比較 (各リスク率)', color='white', fontsize=11, pad=6)

    cmap = plt.get_cmap('plasma')
    n_curves = len(results)
    for i_r, r in enumerate(results):
        eq  = r['equity_curve']
        rp  = r['risk_pct_pct']
        col = cmap(i_r / max(n_curves - 1, 1))
        lw  = 2.5 if rp in (best_calmar_risk, best_safe_risk) else 0.9
        alpha = 1.0 if rp in (best_calmar_risk, best_safe_risk) else 0.45
        lbl = f"{rp:.1f}%  (ROI:{r['roi']:+.0f}%)" if rp in (best_calmar_risk, best_safe_risk) else f"{rp:.1f}%"
        ax_eq.plot(eq.index, eq.values / 1e6, color=col, lw=lw, alpha=alpha, label=lbl)

    ax_eq.axhline(INIT_CASH / 1e6, color='white', lw=0.8, ls='--', alpha=0.4)
    ax_eq.set_ylabel('資産 (百万円)', color='white', fontsize=9)
    ax_eq.tick_params(colors='white', labelsize=8)
    ax_eq.spines[:].set_color('#30363d')
    ax_eq.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d',
                 labelcolor='white', loc='upper left', ncol=3)

    # ---- [1,0] ROI vs リスク率 ----
    ax_roi = fig.add_subplot(gs[1, 0])
    bar_colors = [GREEN if v >= 0 else RED for v in df['roi']]
    ax_roi.bar(x, df['roi'], color=bar_colors, alpha=0.8, width=0.4)
    ax_roi.axhline(0, color='white', lw=0.6)
    ax_style(ax_roi, 'ROI vs リスク率', ylabel='ROI (%)')
    mark_best(ax_roi, best_safe_risk,   ORANGE, f'安全最適\n{best_safe_risk:.1f}%')
    mark_best(ax_roi, best_roi_risk,    CYAN,   f'最大ROI\n{best_roi_risk:.1f}%')

    # ---- [1,1] MaxDD vs リスク率 ----
    ax_dd = fig.add_subplot(gs[1, 1])
    dd_colors = [GREEN if v <= 10 else (ORANGE if v <= 15 else RED) for v in df['max_dd']]
    ax_dd.bar(x, df['max_dd'], color=dd_colors, alpha=0.8, width=0.4)
    ax_dd.axhline(10, color=ORANGE, lw=1, ls='--', alpha=0.7, label='DD 10%')
    ax_dd.axhline(15, color=RED,    lw=1, ls='--', alpha=0.7, label='DD 15%')
    ax_style(ax_dd, '最大ドローダウン vs リスク率', ylabel='MaxDD (%)')
    ax_dd.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
    mark_best(ax_dd, best_safe_risk, ORANGE, f'DD≤15%最適\n{best_safe_risk:.1f}%')

    # ---- [2,0] Calmar vs リスク率 ----
    ax_cal = fig.add_subplot(gs[2, 0])
    ax_cal.plot(x, df['calmar'], color=GOLD, lw=2, marker='o', ms=5)
    ax_cal.fill_between(x, df['calmar'], alpha=0.15, color=GOLD)
    ax_style(ax_cal, 'Calmar比率 vs リスク率 (高いほど優秀)', ylabel='Calmar')
    mark_best(ax_cal, best_calmar_risk, GOLD, f'Calmar最高\n{best_calmar_risk:.1f}%')
    # 最高値にマーカー
    best_cal = df.loc[best_calmar_idx]
    ax_cal.scatter([best_cal['risk_pct_pct']], [best_cal['calmar']],
                   color=GOLD, s=100, zorder=5)
    ax_cal.text(best_cal['risk_pct_pct'] + 0.15, best_cal['calmar'],
                f"  {best_cal['calmar']:.2f}", color=GOLD, fontsize=9, fontweight='bold')

    # ---- [2,1] Sharpe / Sortino vs リスク率 ----
    ax_sh = fig.add_subplot(gs[2, 1])
    ax_sh.plot(x, df['sharpe'],  color=BLUE,   lw=2, marker='o', ms=4, label='Sharpe')
    ax_sh.plot(x, df['sortino'], color=SILVER,  lw=2, marker='s', ms=4, label='Sortino')
    ax_style(ax_sh, 'Sharpe / Sortino vs リスク率', ylabel='比率')
    ax_sh.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
    mark_best(ax_sh, best_sharpe_risk, BLUE, f'Sharpe最高\n{best_sharpe_risk:.1f}%')

    # ---- [3,0] 最終資産 vs リスク率 ----
    ax_fin = fig.add_subplot(gs[3, 0])
    fc_m = df['final_cash'] / 1e6
    ax_fin.bar(x, fc_m, color=SILVER, alpha=0.75, width=0.4)
    ax_fin.axhline(INIT_CASH / 1e6, color='white', lw=0.8, ls='--', alpha=0.5, label='元本')
    ax_style(ax_fin, '最終資産 vs リスク率', ylabel='資産 (百万円)')
    ax_fin.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
    mark_best(ax_fin, best_safe_risk, ORANGE, f'推奨\n{best_safe_risk:.1f}%')

    # ---- [3,1] ROI vs MaxDD バブルチャート (リスク-リターンマップ) ----
    ax_rr = fig.add_subplot(gs[3, 1])
    ax_rr.set_facecolor('#161b22')
    scatter_c = df['calmar'].values
    sc = ax_rr.scatter(
        df['max_dd'], df['roi'],
        c=scatter_c, cmap='RdYlGn', s=120,
        alpha=0.85, zorder=3
    )
    for _, row in df.iterrows():
        ax_rr.text(row['max_dd'] + 0.1, row['roi'] + 0.5,
                   f"{row['risk_pct_pct']:.1f}%",
                   color='white', fontsize=6.5, ha='left', alpha=0.85)
    # 推奨点を強調
    best_row = df.loc[best_safe_idx]
    ax_rr.scatter([best_row['max_dd']], [best_row['roi']],
                  color=ORANGE, s=220, zorder=5, marker='*',
                  label=f"推奨 {best_row['risk_pct_pct']:.1f}%")
    ax_rr.axvline(15, color=RED, lw=1, ls='--', alpha=0.6, label='DD 15% 上限')
    cb = plt.colorbar(sc, ax=ax_rr, fraction=0.04)
    cb.set_label('Calmar', color='white', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white', fontsize=7)
    ax_rr.set_title('リスク-リターンマップ (色 = Calmar比率)', color='white', fontsize=10, pad=5)
    ax_rr.set_xlabel('最大ドローダウン (%)', color='white', fontsize=9)
    ax_rr.set_ylabel('ROI (%)', color='white', fontsize=9)
    ax_rr.tick_params(colors='white', labelsize=8)
    ax_rr.spines[:].set_color('#30363d')
    ax_rr.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # ---- [4,:] 統計テーブル ----
    ax_tbl = fig.add_subplot(gs[4, :])
    ax_tbl.set_facecolor('#0d1117')
    ax_tbl.axis('off')
    ax_tbl.set_title('全リスク率 統計サマリー', color='white', fontsize=11, pad=6)

    cols = ['リスク率', 'ROI%', '年率ROI%', 'MaxDD%',
            'Calmar', 'Sharpe', 'Sortino', 'PF', '勝率%',
            'トレード', '期待値$K', '最終¥M']
    rows = []
    for _, row in df.iterrows():
        rows.append([
            f"{row['risk_pct_pct']:.1f}%",
            f"{row['roi']:+.1f}",
            f"{row['annual_roi']:+.1f}",
            f"{row['max_dd']:.1f}",
            f"{row['calmar']:.2f}",
            f"{row['sharpe']:.2f}",
            f"{row['sortino']:.2f}",
            f"{row['pf']:.2f}",
            f"{row['win_rate']:.1f}",
            f"{int(row['n_trades'])}",
            f"{row['expectancy']/1000:.0f}",
            f"{row['final_cash']/1e6:.2f}",
        ])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=cols,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for (r, c), cell in tbl.get_celld().items():
        base_col = '#1c2128' if r % 2 == 0 else '#161b22'
        # 推奨行をハイライト
        if r > 0:
            row_risk = float(rows[r-1][0].replace('%',''))
            if abs(row_risk - best_safe_risk) < 0.01:
                base_col = '#2d3748'
            elif abs(row_risk - best_calmar_risk) < 0.01:
                base_col = '#1a2a1a'
        cell.set_facecolor('#0d1117' if r == 0 else base_col)
        cell.set_edgecolor('#30363d')
        cell.set_text_props(color='white')
        if r == 0:
            cell.set_text_props(color=GOLD, fontweight='bold')
    tbl.scale(1, 1.4)

    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"\nチャート保存: {outpath}")

# ------------------------------------------------------------------ #
#  メイン                                                             #
# ------------------------------------------------------------------ #
def main():
    results = run_risk_scan()

    if not results:
        print("結果なし")
        return

    df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ('monthly_pnl', 'equity_curve')}
                        for r in results])

    # 最適点サマリー
    best_calmar = df.loc[df['calmar'].idxmax()]
    best_sharpe = df.loc[df['sharpe'].idxmax()]
    best_roi    = df.loc[df['roi'].idxmax()]
    safe_df     = df[df['max_dd'] <= 15.0]
    best_safe   = safe_df.loc[safe_df['roi'].idxmax()] if len(safe_df) > 0 else best_calmar

    print("\n" + "=" * 65)
    print("  最適化結果サマリー")
    print("=" * 65)
    print(f"\n★ Calmar最大   : リスク {best_calmar['risk_pct_pct']:.1f}%  "
          f"→ Calmar={best_calmar['calmar']:.2f}  ROI={best_calmar['roi']:+.1f}%  "
          f"DD={best_calmar['max_dd']:.1f}%")
    print(f"★ Sharpe最大   : リスク {best_sharpe['risk_pct_pct']:.1f}%  "
          f"→ Sharpe={best_sharpe['sharpe']:.2f}  ROI={best_sharpe['roi']:+.1f}%  "
          f"DD={best_sharpe['max_dd']:.1f}%")
    print(f"★ ROI最大      : リスク {best_roi['risk_pct_pct']:.1f}%  "
          f"→ ROI={best_roi['roi']:+.1f}%  DD={best_roi['max_dd']:.1f}%")
    print(f"★ DD≤15%で推奨 : リスク {best_safe['risk_pct_pct']:.1f}%  "
          f"→ ROI={best_safe['roi']:+.1f}%  DD={best_safe['max_dd']:.1f}%  "
          f"Calmar={best_safe['calmar']:.2f}")

    outpath = os.path.join(ROOT, 'results', 'silver_risk_optimization.png')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plot_risk_optimization(results, outpath)
    print(f"チャート: results/silver_risk_optimization.png")

if __name__ == '__main__':
    main()
