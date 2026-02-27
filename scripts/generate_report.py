"""
パフォーマンスレポート生成スクリプト
=====================================
合格戦略 (50+/yr AND PF>1.3) の詳細チャートを results/ に出力する。

生成ファイル:
  results/report_{戦略名}.png  … 戦略ごとの詳細6パネルチャート
  results/report_comparison.png … 全合格戦略の比較チャート

実行:
  python scripts/generate_report.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── カラーテーマ ──
BG       = '#0d0d1a'
PANEL    = '#111130'
GREEN    = '#00ff88'
RED      = '#ff4444'
BLUE     = '#4488ff'
YELLOW   = '#ffcc00'
GRAY     = '#aaaaaa'
ORANGE   = '#ff8800'
PURPLE   = '#cc44ff'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


# ───────────────────────────────────────────────
# データ読み込み
# ───────────────────────────────────────────────

def load_ohlc():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ohlc_dir = os.path.join(base, 'data', 'ohlc')

    def _load(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close']].dropna()
        return df[(df['high'] - df['low']) > 0].sort_index()

    bars_4h = _load(os.path.join(ohlc_dir, 'XAUUSD_4h.csv'))
    bars_1d = _load(os.path.join(ohlc_dir, 'XAUUSD_1d.csv'))
    print(f"[Data] 4H={len(bars_4h)} bars  1D={len(bars_1d)} bars")
    return bars_4h, bars_1d


# ───────────────────────────────────────────────
# 戦略実行
# ───────────────────────────────────────────────

def run_strategies():
    from lib.backtest import BacktestEngine
    from lib.yagami import sig_dc_fast, sig_aggressive_union

    bars_4h, _ = load_ohlc()
    TRADE_START = '2024-01-01'  # ウォームアップ終了後からカウント

    def make_engine():
        return BacktestEngine(
            init_cash=10_000_000, risk_pct=0.02,
            default_sl_atr=1.5, default_tp_atr=4.5,   # RR3
            slippage_pips=0.3, pip=0.1,
            use_dynamic_sl=False,
            pyramid_entries=0,
            trail_start_atr=0.0,
            trail_dist_atr=0.0,
            breakeven_rr=0.0,
            partial_tp_rr=0.0,
            exit_on_signal=False,
            long_biased=True,
            target_max_dd=0.30,
            target_min_wr=0.20,
            target_rr_threshold=3.0,
            target_min_trades=5,
        )

    strategies = [
        ('NT_4H_UNION2d38_RR3', sig_aggressive_union('4h', ema_days=21, lookback_days_dc=2, rsi_thresh=38)),
        ('NT_4H_UNION3d45_RR3', sig_aggressive_union('4h', ema_days=21, lookback_days_dc=3, rsi_thresh=45)),
        ('NT_4H_DC2d_RR3',      sig_dc_fast('4h', lookback_days=2, ema_filter=False, confirm_bars=0)),
        ('NT_4H_UNION3d40_RR3', sig_aggressive_union('4h', ema_days=21, lookback_days_dc=3, rsi_thresh=40)),
    ]

    results = {}
    for name, sig_fn in strategies:
        engine = make_engine()
        result = engine.run(bars_4h, sig_fn, freq='4h', name=name, trade_start=TRADE_START)
        if result:
            results[name] = result
            n = result['total_trades']
            pf = result['profit_factor']
            tpy = result['trades_per_year']
            roi = result['total_return_pct']
            print(f"  {name}: {n}trades  Yr={tpy:.0f}  PF={pf:.3f}  ROI={roi:.1f}%")
    return results


# ───────────────────────────────────────────────
# チャート生成ヘルパー
# ───────────────────────────────────────────────

def _equity_and_dd(trades, init_cash=10_000_000):
    """エクイティカーブとドローダウン系列を計算"""
    times, equity, dd_series = [], [], []
    cash, peak = init_cash, init_cash
    times.append(pd.Timestamp('2024-01-01'))
    equity.append(init_cash)
    dd_series.append(0.0)

    for t in trades:
        cash += t['pnl']
        if cash > peak:
            peak = cash
        dd = (peak - cash) / peak * 100
        exit_time = pd.Timestamp(t['exit_time'])
        times.append(exit_time)
        equity.append(cash)
        dd_series.append(dd)

    return times, equity, dd_series


def _monthly_pnl(trades):
    """月次PnL集計"""
    monthly = {}
    for t in trades:
        et = pd.Timestamp(t['exit_time'])
        key = f"{et.year}-{et.month:02d}"
        monthly[key] = monthly.get(key, 0) + t['pnl']
    keys = sorted(monthly.keys())
    return keys, [monthly[k] for k in keys]


def _setup_ax(ax, title=''):
    ax.set_facecolor(PANEL)
    if title:
        ax.set_title(title, color='white', fontsize=10, pad=6)
    ax.tick_params(colors=GRAY, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#334')
    ax.grid(color='#223', linewidth=0.4, linestyle='--', alpha=0.6)


# ───────────────────────────────────────────────
# 1戦略 詳細チャート (6パネル)
# ───────────────────────────────────────────────

def plot_strategy_report(name, result, bars_4h, save_path):
    trades = result.get('trades', [])
    if not trades:
        print(f"  [skip] {name}: no trades")
        return

    times, equity, dd_series = _equity_and_dd(trades)
    month_keys, month_pnl = _monthly_pnl(trades)

    wins   = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    pf   = result['profit_factor']
    wr   = result['win_rate_pct']
    rr   = result['rr_ratio']
    dd   = result['max_drawdown_pct']
    roi  = result['total_return_pct']
    tpy  = result['trades_per_year']
    n    = result['total_trades']
    avg_dur = result['avg_duration_hours']

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        f'{name}  |  XAUUSD 4H  (2024/01 ~ 2026/02  本物データ)',
        color='white', fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32,
                           top=0.93, bottom=0.06, left=0.06, right=0.97)

    # ── Panel 1: エクイティカーブ ──
    ax1 = fig.add_subplot(gs[0, :2])
    _setup_ax(ax1, 'Equity Curve  (1000万円 → ?万円)')
    equity_man = [e / 10_000 for e in equity]
    color_eq = GREEN if equity[-1] > equity[0] else RED
    ax1.plot(times, equity_man, color=color_eq, lw=1.8, zorder=3)
    ax1.fill_between(times, equity_man[0], equity_man, alpha=0.15, color=color_eq, zorder=2)
    ax1.axhline(equity_man[0], color=GRAY, lw=0.8, ls='--', alpha=0.6)
    ax1.axhline(equity_man[0] * 2, color=YELLOW, lw=0.8, ls=':', alpha=0.5, label='2x目標')
    ax1.axhline(equity_man[0] * 3, color=ORANGE, lw=0.8, ls=':', alpha=0.5, label='3x目標')
    ax1.set_ylabel('資産 (万円)', color=GRAY, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor='white', loc='upper left')

    # ── Panel 2: KPIサマリー ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL)
    for sp in ax2.spines.values():
        sp.set_color('#334')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_xticks([]); ax2.set_yticks([])

    kpis = [
        ('PF (プロフィットファクター)', f'{pf:.3f}', GREEN if pf >= 1.3 else RED),
        ('WR (勝率)', f'{wr:.1f}%', GREEN if wr >= 35 else YELLOW),
        ('実効RR (平均勝ち/負け)', f'{rr:.2f}', GREEN if rr >= 2.5 else YELLOW),
        ('MaxDD (最大ドローダウン)', f'{dd:.1f}%', GREEN if dd <= 25 else (YELLOW if dd <= 30 else RED)),
        ('ROI (総リターン)', f'+{roi:.1f}%', GREEN if roi > 0 else RED),
        ('Yr/回 (年間トレード数)', f'{tpy:.0f}回/年', GREEN if tpy >= 50 else RED),
        ('総トレード数', f'{n}回', GRAY),
        ('平均保有時間', f'{avg_dur:.1f}h', GRAY),
        ('勝ち数 / 負け数', f'{len(wins)} / {len(losses)}', GRAY),
    ]

    y_pos = 0.92
    ax2.set_title('KPI サマリー', color='white', fontsize=10, pad=6)
    for label, value, color in kpis:
        ax2.text(0.05, y_pos, label, color=GRAY, fontsize=8, va='center')
        ax2.text(0.97, y_pos, value, color=color, fontsize=9, va='center',
                 ha='right', fontweight='bold')
        ax2.axhline(y_pos - 0.025, color='#223', lw=0.5, alpha=0.8)
        y_pos -= 0.10

    # ── Panel 3: ドローダウンカーブ ──
    ax3 = fig.add_subplot(gs[1, :2])
    _setup_ax(ax3, 'Drawdown Curve  (ドローダウン推移)')
    ax3.fill_between(times, 0, [-d for d in dd_series], color=RED, alpha=0.4, zorder=2)
    ax3.plot(times, [-d for d in dd_series], color=RED, lw=1.2, zorder=3)
    ax3.axhline(-30, color=ORANGE, lw=0.8, ls='--', alpha=0.7, label='-30% 基準線')
    ax3.axhline(-10, color=YELLOW, lw=0.8, ls=':', alpha=0.5, label='-10%')
    ax3.set_ylabel('Drawdown (%)', color=GRAY, fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor='white', loc='lower left')

    # ── Panel 4: 月次PnL ──
    ax4 = fig.add_subplot(gs[1, 2])
    _setup_ax(ax4, 'Monthly P&L  (月次損益)')
    colors_m = [GREEN if p > 0 else RED for p in month_pnl]
    pnl_man = [p / 10_000 for p in month_pnl]
    x_pos = range(len(month_keys))
    ax4.bar(x_pos, pnl_man, color=colors_m, alpha=0.85, width=0.7)
    ax4.axhline(0, color=GRAY, lw=0.8)
    tick_step = max(1, len(month_keys) // 8)
    ax4.set_xticks(list(x_pos)[::tick_step])
    ax4.set_xticklabels([month_keys[i][2:] for i in range(0, len(month_keys), tick_step)],
                        fontsize=7, rotation=30)
    ax4.set_ylabel('PnL (万円)', color=GRAY, fontsize=9)
    # 月勝率をタイトルに追加
    win_months  = sum(1 for p in month_pnl if p > 0)
    lose_months = sum(1 for p in month_pnl if p <= 0)
    ax4.set_title(f'Monthly P&L  (月勝率 {win_months}/{len(month_pnl)})', color='white', fontsize=10, pad=6)

    # ── Panel 5: 個別トレードPnL散布 ──
    ax5 = fig.add_subplot(gs[2, :2])
    _setup_ax(ax5, 'Trade P&L  (個別トレード損益  ─ 連続Gapを意識)')

    entry_times = [pd.Timestamp(t['entry_time']) for t in trades]
    pnl_vals    = [t['pnl'] / 10_000 for t in trades]
    colors_t    = [GREEN if p > 0 else RED for p in pnl_vals]
    sizes_t     = [max(20, abs(p) * 30) for p in pnl_vals]

    ax5.scatter(entry_times, pnl_vals, c=colors_t, s=sizes_t, alpha=0.7, zorder=3, edgecolors='none')
    ax5.axhline(0, color=GRAY, lw=0.8, ls='--')

    # 移動平均PnL (10取引)
    if len(pnl_vals) >= 10:
        pnl_s = pd.Series(pnl_vals, index=entry_times)
        ma = pnl_s.rolling(10, min_periods=1).mean()
        ax5.plot(ma.index, ma.values, color=BLUE, lw=1.5, alpha=0.8, label='移動平均(10)')
        ax5.legend(fontsize=8, facecolor=PANEL, labelcolor='white')

    ax5.set_ylabel('PnL (万円)', color=GRAY, fontsize=9)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))

    # ── Panel 6: 連敗・WR分析 + 問題点ハイライト ──
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL)
    for sp in ax6.spines.values():
        sp.set_color('#334')
    ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
    ax6.set_xticks([]); ax6.set_yticks([])
    ax6.set_title('問題点 & 注意事項', color=YELLOW, fontsize=10, pad=6)

    # 連敗ストリーク計算
    max_streak = 0
    cur_streak = 0
    for t in trades:
        if t['pnl'] <= 0:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    avg_win_k  = np.mean([t['pnl'] for t in wins]) / 10_000  if wins   else 0
    avg_loss_k = np.mean([t['pnl'] for t in losses]) / 10_000 if losses else 0

    issues = []
    # 自動的に問題点を検出
    if wr < 35:
        issues.append(('⚠ 低勝率: 3回に2回は負け', RED))
        issues.append((f'  最大連敗 {max_streak}回 → 精神面に注意', RED))
    if dd > 20:
        issues.append((f'⚠ DD{dd:.0f}%: {int(10000*dd/100)}万円相当の', YELLOW))
        issues.append(('  含み損に耐える局面あり', YELLOW))
    if tpy > 100:
        issues.append(('⚠ 週3回ペース: ルール厳守が必須', YELLOW))
        issues.append(('  シグナルを選ばず全てエントリー', YELLOW))
    issues.append(('ℹ ロングオンリー: 下落相場は不利', BLUE))
    issues.append(('ℹ SL=1.5ATR固定 (スウィング非使用)', BLUE))
    issues.append((f'ℹ 平均保有 {avg_dur:.0f}h ≈ {avg_dur/24:.1f}日', GRAY))
    issues.append((f'ℹ 平均勝ち: +{avg_win_k:.1f}万 / 平均負け: {avg_loss_k:.1f}万', GRAY))

    y_pos = 0.93
    for text, color in issues:
        ax6.text(0.04, y_pos, text, color=color, fontsize=8, va='center',
                 fontfamily='monospace')
        y_pos -= 0.10

    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  [保存] {save_path}")


# ───────────────────────────────────────────────
# 4戦略比較チャート
# ───────────────────────────────────────────────

def plot_comparison(all_results, save_path):
    """合格4戦略のエクイティカーブを1枚に重ねて比較"""
    names = list(all_results.keys())
    colors_list = [GREEN, BLUE, YELLOW, PURPLE]
    short_names = {
        'NT_4H_UNION2d38_RR3': 'UNION(DC2d+RSI38) RR3',
        'NT_4H_UNION3d45_RR3': 'UNION(DC3d+RSI45) RR3',
        'NT_4H_DC2d_RR3':      'DC2d RR3',
        'NT_4H_UNION3d40_RR3': 'UNION(DC3d+RSI40) RR3',
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor=BG)
    fig.suptitle(
        '合格戦略 比較レポート  |  ミッション5.0達成: 年50回以上 AND PF>1.3',
        color='white', fontsize=13, fontweight='bold', y=0.98
    )

    # ── A: エクイティカーブ比較 ──
    ax_eq = axes[0, 0]
    _setup_ax(ax_eq, 'Equity Curve 比較  (元本 1000万円)')
    ax_eq.axhline(1000, color=GRAY, lw=0.8, ls='--', alpha=0.5)
    ax_eq.axhline(2000, color=YELLOW, lw=0.8, ls=':', alpha=0.4, label='2x')
    ax_eq.axhline(3000, color=ORANGE, lw=0.8, ls=':', alpha=0.4, label='3x')

    for (name, result), color in zip(all_results.items(), colors_list):
        trades = result.get('trades', [])
        if not trades:
            continue
        times, equity, _ = _equity_and_dd(trades)
        equity_man = [e / 10_000 for e in equity]
        label = f"{short_names.get(name, name)}  ROI={result['total_return_pct']:.0f}%"
        ax_eq.plot(times, equity_man, color=color, lw=1.8, label=label, alpha=0.9)

    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax_eq.set_ylabel('資産 (万円)', color=GRAY, fontsize=9)
    ax_eq.legend(fontsize=7.5, facecolor=PANEL, labelcolor='white', loc='upper left')

    # ── B: ドローダウン比較 ──
    ax_dd = axes[0, 1]
    _setup_ax(ax_dd, 'Drawdown 比較')
    ax_dd.axhline(-30, color=ORANGE, lw=0.8, ls='--', alpha=0.6, label='-30% 限度')

    for (name, result), color in zip(all_results.items(), colors_list):
        trades = result.get('trades', [])
        if not trades:
            continue
        times, _, dd_series = _equity_and_dd(trades)
        ax_dd.plot(times, [-d for d in dd_series], color=color, lw=1.5, alpha=0.85,
                   label=short_names.get(name, name))

    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax_dd.set_ylabel('Drawdown (%)', color=GRAY, fontsize=9)
    ax_dd.legend(fontsize=7.5, facecolor=PANEL, labelcolor='white')

    # ── C: KPI バー比較 ──
    ax_kpi = axes[1, 0]
    _setup_ax(ax_kpi, 'KPI 比較  (PF / WR / MaxDD)')

    x = np.arange(len(names))
    w = 0.22
    pf_vals  = [all_results[n]['profit_factor']     for n in names]
    wr_vals  = [all_results[n]['win_rate_pct'] / 10  for n in names]  # 0〜10スケール
    dd_vals  = [all_results[n]['max_drawdown_pct']   for n in names]
    tpy_vals = [all_results[n]['trades_per_year'] / 30 for n in names]  # 0〜10スケール

    ax_kpi.bar(x - 1.5*w, pf_vals,  width=w, color=GREEN,  alpha=0.85, label='PF')
    ax_kpi.bar(x - 0.5*w, wr_vals,  width=w, color=BLUE,   alpha=0.85, label='WR÷10')
    ax_kpi.bar(x + 0.5*w, dd_vals,  width=w, color=RED,    alpha=0.85, label='MaxDD%')
    ax_kpi.bar(x + 1.5*w, tpy_vals, width=w, color=YELLOW, alpha=0.85, label='Yr/30')

    ax_kpi.axhline(1.3, color=GREEN, lw=0.8, ls='--', alpha=0.5)
    ax_kpi.set_xticks(x)
    short_n = [n.replace('NT_4H_', '') for n in names]
    ax_kpi.set_xticklabels(short_n, fontsize=8, rotation=10)
    ax_kpi.legend(fontsize=8, facecolor=PANEL, labelcolor='white')

    # ── D: サマリーテーブル ──
    ax_tbl = axes[1, 1]
    ax_tbl.set_facecolor(PANEL)
    for sp in ax_tbl.spines.values():
        sp.set_color('#334')
    ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(0, 1)
    ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
    ax_tbl.set_title('合格戦略 サマリーテーブル', color='white', fontsize=10, pad=6)

    headers = ['戦略', 'Yr/回', 'PF', 'WR%', 'DD%', 'ROI%']
    col_x   = [0.01, 0.37, 0.53, 0.65, 0.77, 0.89]
    y_top   = 0.90

    for hdr, cx in zip(headers, col_x):
        ax_tbl.text(cx, y_top, hdr, color=YELLOW, fontsize=8.5, fontweight='bold', va='center')
    ax_tbl.axhline(y_top - 0.04, color='#445', lw=0.8)

    y_row = y_top - 0.10
    for (name, result), color in zip(all_results.items(), colors_list):
        vals = [
            short_names.get(name, name)[:22],
            f"{result['trades_per_year']:.0f}",
            f"{result['profit_factor']:.3f}",
            f"{result['win_rate_pct']:.1f}",
            f"{result['max_drawdown_pct']:.1f}",
            f"+{result['total_return_pct']:.0f}",
        ]
        for val, cx in zip(vals, col_x):
            ax_tbl.text(cx, y_row, val, color=color, fontsize=8, va='center',
                        fontfamily='monospace')
        ax_tbl.axhline(y_row - 0.04, color='#223', lw=0.4, alpha=0.6)
        y_row -= 0.10

    y_note = y_row - 0.06
    notes = [
        '【戦略設計】',
        '  固定ATR SL (1.5×ATR) + 固定RR3 TP',
        '  トレーリングストップなし',
        '  リスク/トレード: 2% (MaxDD抑制)',
        '  ロングバイアス (4H足)',
        '',
        '【問題点】',
        '  ・WR≈33% → 連敗は避けられない',
        '  ・ロングオンリー → 下落相場では不利',
        '  ・ATR SL → 急騰後の揺り戻しで刈られやすい',
        '  ・週3回ペース → 全シグナル実行の規律必須',
    ]
    for note in notes:
        color_n = YELLOW if note.startswith('【') else (RED if '問題点' in note or '・' in note else GRAY)
        ax_tbl.text(0.01, y_note, note, color=color_n, fontsize=7.5, va='center',
                    fontfamily='monospace')
        y_note -= 0.065

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  [保存] {save_path}")


# ───────────────────────────────────────────────
# メイン
# ───────────────────────────────────────────────

def main():
    print('=' * 60)
    print('パフォーマンスレポート生成')
    print('合格戦略 (50+/yr AND PF>1.3) 詳細分析')
    print('=' * 60)

    print('\n[1] 戦略実行中...')
    all_results = run_strategies()

    bars_4h, _ = load_ohlc()

    print('\n[2] 個別レポート生成中...')
    for name, result in all_results.items():
        safe = name.replace('/', '-')
        save_path = os.path.join(RESULTS_DIR, f'report_{safe}.png')
        plot_strategy_report(name, result, bars_4h, save_path)

    print('\n[3] 比較チャート生成中...')
    comp_path = os.path.join(RESULTS_DIR, 'report_comparison.png')
    plot_comparison(all_results, comp_path)

    print('\n' + '=' * 60)
    print('完了! results/ に以下のファイルを出力:')
    for name in all_results:
        print(f'  report_{name}.png')
    print('  report_comparison.png')
    print('=' * 60)


if __name__ == '__main__':
    main()
