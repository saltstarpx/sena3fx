"""
P0-1修正効果検証バックテスト
================================
C1フィルター: extract_levels() → extract_levels_binned() 切替の効果を定量検証。

Before: extract_levels(min_touches=2) → C1充足率99.7%（フィルター無効）
After:  extract_levels_binned(n_bins=40, min_freq_pct=0.06) → C1充足率30-40%（有効）

RunID: RUN-20260305-004
"""
import os
import sys
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib.backtest import BacktestEngine
from lib.yagami import (
    yagami_signal, sig_yagami_A, sig_yagami_B,
    sig_yagami_london_ny, sig_maedai_yagami_union
)
from lib.levels import extract_levels, extract_levels_binned, is_at_level
from lib.candle import detect_single_candle, detect_price_action, detect_trendless
from lib.patterns import detect_chart_patterns
from lib.timing import detect_bar_update_timing, session_filter

# ──────────────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────────────
def load_ohlc(path):
    df = pd.read_csv(path)
    try:
        dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if hasattr(dt, 'dt') and dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    df['datetime'] = dt
    df = df.set_index('datetime').sort_index()
    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    return df[cols].astype(float)


# ──────────────────────────────────────────────────────
# C1充足率の計測（Before/After比較）
# ──────────────────────────────────────────────────────
def measure_c1_hit_rate(bars: pd.DataFrame, use_binned: bool = False,
                         n_bins: int = 40, min_freq_pct: float = 0.06):
    """C1（レジサポ近傍）の充足率を計測する"""
    from lib.yagami import analyze_bars
    df = analyze_bars(bars, '4h')

    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(14).mean().values

    total = 0
    c1_hit = 0
    levels_cache = None
    levels_update_idx = -100

    for i in range(20, len(df)):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        if df['trendless'].iloc[i]:
            continue

        if i - levels_update_idx >= 10:
            if use_binned:
                levels_cache = extract_levels_binned(
                    df.iloc[max(0, i-100):i+1], n_bins=n_bins, min_freq_pct=min_freq_pct)
            else:
                levels_cache = extract_levels(df.iloc[max(0, i-100):i+1])
            levels_update_idx = i

        at_level, _ = is_at_level(c[i], levels_cache, atr[i], 0.8)
        total += 1
        if at_level:
            c1_hit += 1

    return c1_hit, total, c1_hit / total * 100 if total > 0 else 0


# ──────────────────────────────────────────────────────
# バックテスト実行
# ──────────────────────────────────────────────────────
def run_bt(df, sig_func, label):
    engine = BacktestEngine(
        init_cash=5_000_000,
        risk_pct=0.02,
        default_sl_atr=1.5,
        default_tp_atr=3.0,
        pyramid_entries=0,
        target_max_dd=0.30,
        target_min_wr=0.35,
        target_rr_threshold=1.5,
        target_min_trades=10,
    )
    try:
        result = engine.run(data=df, signal_func=sig_func)
        return {
            'strategy': label,
            'pf': result.get('profit_factor', 0),
            'wr': result.get('win_rate_pct', 0),
            'sharpe': result.get('sharpe_ratio', 0),
            'calmar': result.get('calmar_ratio', 0),
            'mdd': result.get('max_drawdown_pct', 0),
            'trades': result.get('total_trades', 0),
            'total_return': result.get('total_return_pct', 0),
        }
    except Exception as e:
        return {
            'strategy': label,
            'pf': 0, 'wr': 0, 'sharpe': 0, 'calmar': 0,
            'mdd': 0, 'trades': 0, 'total_return': 0,
            'error': str(e)
        }


def main():
    print("=" * 70)
    print("RUN-20260305-004: P0-1修正効果検証（C1フィルター切替）")
    print("=" * 70)

    # データ読み込み（XAUUSD 4H）
    data_path = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_4h.csv')
    df = load_ohlc(data_path)
    print(f"データ: {data_path}")
    print(f"期間: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}本")

    # ──────────────────────────────────────────────────────
    # Step 1: C1充足率の計測
    # ──────────────────────────────────────────────────────
    print("\n[Step 1] C1充足率の計測...")
    print("  Before (extract_levels)...", end='', flush=True)
    hit_before, total_before, rate_before = measure_c1_hit_rate(df, use_binned=False)
    print(f" {rate_before:.1f}% ({hit_before}/{total_before})")

    print("  After  (extract_levels_binned n_bins=40, min_freq=0.06)...", end='', flush=True)
    hit_after, total_after, rate_after = measure_c1_hit_rate(df, use_binned=True, n_bins=40, min_freq_pct=0.06)
    print(f" {rate_after:.1f}% ({hit_after}/{total_after})")

    # ──────────────────────────────────────────────────────
    # Step 2: バックテスト比較（Before vs After）
    # ──────────────────────────────────────────────────────
    print("\n[Step 2] バックテスト比較...")

    # Beforeシグナル（旧extract_levels）
    # ※ 旧実装は一時的にパッチして計測
    import lib.yagami as yagami_mod
    import lib.levels as levels_mod

    # --- Before: 旧extract_levelsを使うシグナル ---
    _orig_extract = levels_mod.extract_levels_binned  # 現在はbinnedに変更済み

    # Beforeを再現するため一時的にextract_levels_binnedをextract_levelsに差し替え
    def _old_extract_levels_binned(bars, n_bins=40, min_freq_pct=0.06, **kwargs):
        return extract_levels(bars)  # 旧実装を模倣

    levels_mod.extract_levels_binned = _old_extract_levels_binned

    print("  Before (旧C1) バックテスト実行中...")
    variants_before = [
        ('Yagami_A_Before', sig_yagami_A('4h')(df)),
        ('Yagami_B_Before', sig_yagami_B('4h')(df)),
        ('Yagami_LonNY_Before', sig_yagami_london_ny('4h')(df)),
    ]

    results_before = []
    for label, sig in variants_before:
        r = run_bt(df, lambda bars, s=sig: s, label)
        results_before.append(r)
        print(f"    {label}: PF={r['pf']:.3f}, WR={r['wr']:.1f}%, Trades={r['trades']}")

    # --- After: 新extract_levels_binnedを使うシグナル ---
    levels_mod.extract_levels_binned = _orig_extract  # 元に戻す

    print("  After (新C1) バックテスト実行中...")
    variants_after = [
        ('Yagami_A_After', sig_yagami_A('4h')(df)),
        ('Yagami_B_After', sig_yagami_B('4h')(df)),
        ('Yagami_LonNY_After', sig_yagami_london_ny('4h')(df)),
    ]

    results_after = []
    for label, sig in variants_after:
        r = run_bt(df, lambda bars, s=sig: s, label)
        results_after.append(r)
        print(f"    {label}: PF={r['pf']:.3f}, WR={r['wr']:.1f}%, Trades={r['trades']}")

    # ──────────────────────────────────────────────────────
    # Step 3: 結果集計
    # ──────────────────────────────────────────────────────
    print("\n[Step 3] 結果集計...")

    all_results = results_before + results_after
    df_results = pd.DataFrame(all_results)

    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    out_path = os.path.join(ROOT, 'results', 'run004_p0_c1_fix.csv')
    df_results.to_csv(out_path, index=False)
    print(f"  結果保存: {out_path}")

    # ──────────────────────────────────────────────────────
    # Step 4: 比較サマリー
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("C1充足率比較:")
    print(f"  Before: {rate_before:.1f}%  →  After: {rate_after:.1f}%  (Δ={rate_after - rate_before:+.1f}%)")
    print()
    print("バックテスト比較:")
    print(f"{'戦略':<30} {'PF':>6} {'WR%':>6} {'Sharpe':>8} {'MDD%':>6} {'N':>5}")
    print("-" * 65)
    for r in all_results:
        tag = "▲AFTER" if "After" in r['strategy'] else "  BEFORE"
        print(f"{r['strategy']:<30} {r['pf']:>6.3f} {r['wr']:>6.1f} {r['sharpe']:>8.3f} {r['mdd']:>6.1f} {r['trades']:>5}")

    # Before/After 平均比較
    before_pf = np.mean([r['pf'] for r in results_before])
    after_pf = np.mean([r['pf'] for r in results_after])
    before_wr = np.mean([r['wr'] for r in results_before])
    after_wr = np.mean([r['wr'] for r in results_after])

    print()
    print(f"平均PF:  Before={before_pf:.3f}  →  After={after_pf:.3f}  (Δ={after_pf - before_pf:+.3f})")
    print(f"平均WR:  Before={before_wr:.1f}%  →  After={after_wr:.1f}%  (Δ={after_wr - before_wr:+.1f}%)")
    print("=" * 70)

    return {
        'c1_rate_before': rate_before,
        'c1_rate_after': rate_after,
        'avg_pf_before': before_pf,
        'avg_pf_after': after_pf,
        'avg_wr_before': before_wr,
        'avg_wr_after': after_wr,
        'results': all_results,
    }


if __name__ == '__main__':
    main()
