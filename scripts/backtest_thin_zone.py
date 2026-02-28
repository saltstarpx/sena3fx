"""
薄いゾーン × バックテスト 比較分析
=====================================
エントリー価格が「薄いゾーン（価格帯滞在時間が少ない帯域）」にあるか否かで
トレードパフォーマンスを分割し、以下を検証する:

  1. 勝率         (全体 / 薄いゾーン内 / 薄いゾーン外)
  2. 平均RR        (同上)
  3. 最大ドローダウン (全体)
  4. 薄いゾーン内外でのパフォーマンス差 (PF, WR, RR, 平均PnL)

  さらにロット0.7 × ストップ1.3設定の有効性をシミュレーション比較する。

実行方法:
  python scripts/backtest_thin_zone.py
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
import matplotlib.gridspec as gridspec

from lib.backtest import BacktestEngine
from lib.yagami import (
    sig_yagami_A, sig_yagami_B,
    sig_yagami_full_filter, sig_yagami_london_ny,
)
from price_zone_analyzer import load_thin_zones, is_thin_zone

OHLC_1H_PATH = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_1h.csv')
OHLC_4H_PATH = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
RESULTS_DIR  = os.path.join(ROOT, 'results')


# ------------------------------------------------------------------ #
#  データ読み込み                                                     #
# ------------------------------------------------------------------ #
def load_ohlc(path: str) -> pd.DataFrame:
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


# ------------------------------------------------------------------ #
#  薄いゾーンタグ付け                                                #
# ------------------------------------------------------------------ #
def tag_thin_zones(trades: list, thin_zones: list) -> list:
    """各トレードにエントリー価格ベースの薄いゾーンフラグを付与"""
    for t in trades:
        t['in_thin_zone'] = is_thin_zone(t['entry_price'], thin_zones)
    return trades


# ------------------------------------------------------------------ #
#  メトリクス計算                                                     #
# ------------------------------------------------------------------ #
def calc_metrics(group: list, label: str) -> dict:
    """トレードリストから主要指標を計算"""
    if not group:
        return {
            'label': label, 'count': 0,
            'win_rate_pct': 0.0, 'profit_factor': 0.0,
            'avg_rr': 0.0, 'avg_pnl': 0.0,
        }
    wins   = [t for t in group if t['pnl'] > 0]
    losses = [t for t in group if t['pnl'] <= 0]
    total_win  = sum(t['pnl'] for t in wins)   if wins   else 0.0
    total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1.0
    wr = len(wins) / len(group) * 100
    pf = total_win / total_loss if total_loss > 0 else 999.0
    avg_win  = np.mean([t['pnl'] for t in wins])   if wins   else 0.0
    avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 1.0
    rr = avg_win / avg_loss if avg_loss > 0 else 0.0
    return {
        'label':          label,
        'count':          len(group),
        'win_rate_pct':   round(wr, 1),
        'profit_factor':  round(pf, 3),
        'avg_rr':         round(rr, 2),
        'avg_pnl':        round(float(np.mean([t['pnl'] for t in group])), 2),
    }


def split_metrics(trades: list):
    """薄いゾーン内/外に分割してメトリクスを返す"""
    thin   = [t for t in trades if t.get('in_thin_zone')]
    normal = [t for t in trades if not t.get('in_thin_zone')]
    return calc_metrics(thin, '薄いゾーン内'), calc_metrics(normal, '薄いゾーン外')


def calc_max_drawdown(trades: list, init_cash: float = 5_000_000) -> float:
    """最大ドローダウン(%)を計算"""
    if not trades:
        return 0.0
    cash = init_cash
    peak = cash
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x['exit_time']):
        cash += t['pnl']
        if cash > peak:
            peak = cash
        dd = (peak - cash) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100


# ------------------------------------------------------------------ #
#  ロット/ストップ調整シミュレーション                               #
# ------------------------------------------------------------------ #
def simulate_lot_stop_adjustment(trades: list,
                                  lot_scale: float = 0.7,
                                  stop_scale: float = 1.3) -> list:
    """
    薄いゾーン内のトレードにロット縮小をシミュレート。

    【近似モデル】
    - lot_scale 0.7 → PnL × 0.7 (線形近似)
    - stop_scale 1.3 → SL距離拡大 → TP距離も比例拡大 (dynamic_rr維持)
      実際の影響はRR維持のため PnL スケールのみ変わる
    """
    adjusted = []
    for t in trades:
        t2 = dict(t)
        if t.get('in_thin_zone'):
            t2['pnl'] = t['pnl'] * lot_scale
        adjusted.append(t2)
    return adjusted


# ------------------------------------------------------------------ #
#  表示ユーティリティ                                                #
# ------------------------------------------------------------------ #
def print_comparison(thin_stats: dict, normal_stats: dict, title: str = '') -> None:
    """薄いゾーン内/外比較テーブルを出力"""
    if title:
        print(f"\n  ── {title} ──")
    line = f"  {'指標':18s} {'薄いゾーン内':>12s} {'薄いゾーン外':>12s} {'差':>10s}"
    print(line)
    print(f"  {'─'*18} {'─'*12} {'─'*12} {'─'*10}")
    metrics = [
        ('トレード数',     'count',         ''),
        ('勝率',          'win_rate_pct',   '%'),
        ('プロフィットF',  'profit_factor',  ''),
        ('平均RR',        'avg_rr',         ''),
        ('平均PnL',       'avg_pnl',        'pip'),
    ]
    for label, key, unit in metrics:
        v1 = thin_stats.get(key, 0)
        v2 = normal_stats.get(key, 0)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = v1 - v2
            if key == 'count':
                print(f"  {label:18s} {v1:>11.0f}{unit} {v2:>11.0f}{unit} {diff:>+9.0f}")
            else:
                print(f"  {label:18s} {v1:>10.2f}{unit} {v2:>10.2f}{unit} {diff:>+9.2f}{unit}")


# ------------------------------------------------------------------ #
#  可視化                                                             #
# ------------------------------------------------------------------ #
def plot_thin_zone_comparison(all_results: list,
                               outpath: str) -> None:
    """薄いゾーン内/外の比較チャートを生成"""
    if not all_results:
        return

    df = pd.DataFrame(all_results)
    strategies = df['strategy'].tolist()
    x = np.arange(len(strategies))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        '薄いゾーン内/外 パフォーマンス比較\n(XAUUSD 2025-2026 やがみ戦略)',
        color='white', fontsize=13, fontweight='bold', y=0.98
    )

    THIN_COLOR   = '#f85149'
    NORMAL_COLOR = '#58a6ff'
    bg = '#161b22'

    plot_cfg = [
        (axes[0, 0], 'thin_wr',  'normal_wr',  '勝率 (%)',         '%'),
        (axes[0, 1], 'thin_pf',  'normal_pf',  'プロフィットF',    ''),
        (axes[1, 0], 'thin_rr',  'normal_rr',  '平均RR',           ''),
        (axes[1, 1], 'thin_pnl', 'normal_pnl', '平均PnL (pip)',    'pip'),
    ]

    for ax, thin_col, normal_col, ylabel, unit in plot_cfg:
        ax.set_facecolor(bg)
        thin_vals   = df[thin_col].tolist()
        normal_vals = df[normal_col].tolist()
        b1 = ax.bar(x - w/2, thin_vals,   w, label='薄いゾーン内',
                    color=THIN_COLOR,   alpha=0.85)
        b2 = ax.bar(x + w/2, normal_vals, w, label='薄いゾーン外',
                    color=NORMAL_COLOR, alpha=0.85)
        ax.set_title(ylabel, color='white', fontsize=11, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=25, ha='right', fontsize=8, color='white')
        ax.tick_params(colors='white', labelsize=8)
        ax.spines[:].set_color('#30363d')
        ax.set_facecolor(bg)
        ax.legend(fontsize=8, facecolor=bg, edgecolor='#30363d', labelcolor='white')
        ax.axhline(0, color='#30363d', lw=0.8)
        # 値ラベル
        for bar in b1:
            h = bar.get_height()
            if h != 0:
                ax.text(bar.get_x() + bar.get_width()/2, h,
                        f'{h:.1f}{unit}', ha='center', va='bottom',
                        fontsize=7, color='white')
        for bar in b2:
            h = bar.get_height()
            if h != 0:
                ax.text(bar.get_x() + bar.get_width()/2, h,
                        f'{h:.1f}{unit}', ha='center', va='bottom',
                        fontsize=7, color='white')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  比較チャート保存: {outpath}")


# ------------------------------------------------------------------ #
#  メイン                                                             #
# ------------------------------------------------------------------ #
def run() -> None:
    print("=" * 65)
    print("  薄いゾーン × バックテスト 比較分析 (やがみメソッド)")
    print("=" * 65)

    # --- データ ---
    print("\n[1] データ読み込み")
    if not os.path.exists(OHLC_1H_PATH):
        print(f"  ERROR: {OHLC_1H_PATH} が見つかりません")
        sys.exit(1)
    df_1h = load_ohlc(OHLC_1H_PATH)
    df_4h = load_ohlc(OHLC_4H_PATH)
    thin_zones = load_thin_zones()

    print(f"  1H: {len(df_1h)} 本  ({df_1h.index[0].date()} 〜 {df_1h.index[-1].date()})")
    print(f"  4H: {len(df_4h)} 本  ({df_4h.index[0].date()} 〜 {df_4h.index[-1].date()})")
    print(f"  薄いゾーン: {len(thin_zones)} ゾーン")
    for z in thin_zones:
        dist = (z['low'] + z['high']) / 2 - df_4h['close'].iloc[-1]
        print(f"    ${z['low']:>7,.0f}–${z['high']:>7,.0f}  ({z['bar_count']:>3}本)  {dist:+.0f}$")

    # --- バックテストエンジン ---
    print("\n[2] バックテスト実行")
    engine = BacktestEngine(
        init_cash        = 5_000_000,
        risk_pct         = 0.05,
        default_sl_atr   = 2.0,
        default_tp_atr   = 4.0,
        pyramid_entries  = 0,
        target_max_dd    = 0.20,
        target_min_wr    = 0.35,
        target_rr_threshold = 2.0,
        target_min_trades   = 20,
    )

    strategies = [
        ('YagamiA_1H',      sig_yagami_A(freq='1h'),       df_1h, '1h', df_4h),
        ('YagamiB_1H',      sig_yagami_B(freq='1h'),       df_1h, '1h', df_4h),
        ('YagamiA_4H',      sig_yagami_A(freq='4h'),       df_4h, '4h', None),
        ('YagamiB_4H',      sig_yagami_B(freq='4h'),       df_4h, '4h', None),
        ('YagamiLonNY_1H',  sig_yagami_london_ny(freq='1h'), df_1h, '1h', df_4h),
        ('YagamiFull_1H',   sig_yagami_full_filter(freq='1h'), df_1h, '1h', df_4h),
    ]

    all_results = []
    all_trades_combined = []

    for strat_name, sig_func, bars, freq, htf in strategies:
        print(f"\n  [{strat_name}]")
        result = engine.run(
            data=bars, signal_func=sig_func,
            freq=freq, name=strat_name, htf_bars=htf,
        )
        if result is None or result.get('total_trades', 0) == 0:
            print(f"    → トレードなし (スキップ)")
            continue

        trades = tag_thin_zones(result.get('trades', []), thin_zones)
        all_trades_combined.extend(trades)

        n    = result['total_trades']
        wr   = result['win_rate_pct']
        pf   = result['profit_factor']
        rr   = result['rr_ratio']
        dd   = result['max_drawdown_pct']
        pasy = result.get('trades_per_year', 0)
        ok   = result['passed']

        thin_cnt   = sum(1 for t in trades if t['in_thin_zone'])
        normal_cnt = n - thin_cnt

        print(f"    全体: {n}回  WR={wr:.1f}%  PF={pf:.3f}  RR={rr:.2f}"
              f"  DD={dd:.1f}%  TpY={pasy:.0f}  {'✓合格' if ok else '✗不合格'}")
        print(f"    薄いゾーン内: {thin_cnt}回  薄いゾーン外: {normal_cnt}回")

        # 内/外分割
        thin_stats, normal_stats = split_metrics(trades)
        print_comparison(thin_stats, normal_stats)

        # ロット0.7調整シミュレーション
        adj = simulate_lot_stop_adjustment(trades, lot_scale=0.7, stop_scale=1.3)
        adj_thin, adj_normal = split_metrics(adj)
        adj_dd = calc_max_drawdown(adj)
        print(f"\n    [ロット0.7調整後] DD={adj_dd:.1f}%  (元DD={dd:.1f}%)")
        print_comparison(adj_thin, adj_normal, 'ロット調整後')

        all_results.append({
            'strategy':      strat_name,
            'total_trades':  n,
            'win_rate_pct':  wr,
            'profit_factor': pf,
            'rr_ratio':      rr,
            'max_dd_pct':    dd,
            'passed':        ok,
            # 薄いゾーン内
            'thin_count':    thin_stats['count'],
            'thin_wr':       thin_stats['win_rate_pct'],
            'thin_pf':       thin_stats['profit_factor'],
            'thin_rr':       thin_stats['avg_rr'],
            'thin_pnl':      thin_stats['avg_pnl'],
            # 薄いゾーン外
            'normal_count':  normal_stats['count'],
            'normal_wr':     normal_stats['win_rate_pct'],
            'normal_pf':     normal_stats['profit_factor'],
            'normal_rr':     normal_stats['avg_rr'],
            'normal_pnl':    normal_stats['avg_pnl'],
            # 差分
            'wr_diff':       thin_stats['win_rate_pct'] - normal_stats['win_rate_pct'],
            'pf_diff':       thin_stats['profit_factor'] - normal_stats['profit_factor'],
            'rr_diff':       thin_stats['avg_rr'] - normal_stats['avg_rr'],
            # 調整後
            'adj_dd_pct':    adj_dd,
            'adj_thin_wr':   adj_thin['win_rate_pct'],
            'adj_thin_pf':   adj_thin['profit_factor'],
        })

    if not all_results:
        print("\n[!] 有効なバックテスト結果が0件 — データ量不足の可能性があります")
        return

    # ---------------------------------------------------------------- #
    #  総合サマリー                                                     #
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 65)
    print("  [3] 総合サマリー: 薄いゾーン設定の有効性判断")
    print("=" * 65)

    df_r = pd.DataFrame(all_results)
    avg_wr_thin   = df_r['thin_wr'].mean()
    avg_wr_normal = df_r['normal_wr'].mean()
    avg_pf_thin   = df_r['thin_pf'].mean()
    avg_pf_normal = df_r['normal_pf'].mean()
    avg_rr_thin   = df_r['thin_rr'].mean()
    avg_rr_normal = df_r['normal_rr'].mean()

    print(f"\n  平均勝率    薄いゾーン {avg_wr_thin:.1f}%  vs  通常 {avg_wr_normal:.1f}%  "
          f"(差 {avg_wr_thin - avg_wr_normal:+.1f}%)")
    print(f"  平均PF      薄いゾーン {avg_pf_thin:.3f}  vs  通常 {avg_pf_normal:.3f}  "
          f"(差 {avg_pf_thin - avg_pf_normal:+.3f})")
    print(f"  平均RR      薄いゾーン {avg_rr_thin:.2f}   vs  通常 {avg_rr_normal:.2f}   "
          f"(差 {avg_rr_thin - avg_rr_normal:+.2f})")

    # 全体ドローダウン (全戦略トレード合算)
    combined_dd = calc_max_drawdown(all_trades_combined)
    print(f"\n  全戦略合算 最大ドローダウン: {combined_dd:.1f}%")

    # 判断ロジック
    print(f"\n  ── 判断 ──")
    wr_gap = avg_wr_thin - avg_wr_normal
    rr_gap = avg_rr_thin - avg_rr_normal

    if wr_gap < -5:
        print(f"  ✓ 薄いゾーンで勝率が {abs(wr_gap):.1f}% 低い")
        print(f"    → ロット0.7・ストップ1.3 は正しい設定。本番適用を推奨。")
    elif abs(wr_gap) <= 5:
        print(f"  △ 薄いゾーンと通常の勝率差は {abs(wr_gap):.1f}% 以内 (誤差レベル)")
        print(f"    → ロット調整より「チェイスエントリー許可」だけ残す選択肢を検討。")
    else:
        print(f"  ? 薄いゾーンで勝率が {wr_gap:.1f}% 高い — 逆パターンを確認。")

    if rr_gap < -0.3:
        print(f"\n  ✓ 薄いゾーンでRRが {abs(rr_gap):.2f} 悪い")
        print(f"    → ストップ1.3倍では不足の可能性。ストップ2.0倍を検討。")
    elif rr_gap > 0.3:
        print(f"\n  △ 薄いゾーンでRRが良好 → 現設定を維持。")

    # ---------------------------------------------------------------- #
    #  CSV保存                                                          #
    # ---------------------------------------------------------------- #
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'thin_zone_backtest_comparison.csv')
    df_r.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  結果CSV: {csv_path}")

    # ---------------------------------------------------------------- #
    #  可視化                                                           #
    # ---------------------------------------------------------------- #
    img_path = os.path.join(RESULTS_DIR, 'thin_zone_comparison.png')
    plot_thin_zone_comparison(all_results, img_path)

    print(f"\n{'='*65}")
    print(f"  完了")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    run()
