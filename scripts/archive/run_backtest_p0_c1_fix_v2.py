"""
P0-1修正効果検証バックテスト v2
================================
C1フィルター: extract_levels() → extract_levels_binned() 切替の効果を定量検証。

Before: extract_levels(min_touches=2) → C1充足率99.7%（フィルター無効）
After:  extract_levels_binned(n_bins=40, min_freq_pct=0.06) → C1充足率30-40%（有効）

RunID: RUN-20260305-004
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib.backtest import BacktestEngine
import lib.levels as levels_mod
from lib.levels import extract_levels, extract_levels_binned, is_at_level

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
# C1充足率の計測
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
def run_bt_with_sig_func(df, sig_factory_fn, freq, label):
    """sig_factoryを呼び出してシグナルを生成し、バックテストを実行する"""
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
        sig_func = sig_factory_fn(freq)
        result = engine.run(data=df, signal_func=sig_func, freq=freq)
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
            'error': str(e)[:100],
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
    print(f"データ: {os.path.basename(data_path)}")
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
    # Step 2: バックテスト比較
    # ──────────────────────────────────────────────────────
    print("\n[Step 2] バックテスト比較...")

    # Before: extract_levelsに一時差し替え
    def _mock_binned(bars, n_bins=40, min_freq_pct=0.06, **kw):
        return extract_levels(bars)

    from lib.yagami import sig_yagami_A, sig_yagami_B, sig_yagami_london_ny

    print("  Before (旧C1=extract_levels) バックテスト実行中...")
    levels_mod.extract_levels_binned = _mock_binned
    # キャッシュをリセット
    import importlib, lib.yagami as yagami_mod
    importlib.reload(yagami_mod)
    from lib.yagami import sig_yagami_A as sig_A_before, sig_yagami_B as sig_B_before
    from lib.yagami import sig_yagami_london_ny as sig_lon_before

    results_before = []
    for factory, lbl in [(sig_A_before, 'Yagami_A_Before'),
                          (sig_B_before, 'Yagami_B_Before'),
                          (sig_lon_before, 'Yagami_LonNY_Before')]:
        r = run_bt_with_sig_func(df, factory, '4h', lbl)
        results_before.append(r)
        print(f"    {lbl}: PF={r['pf']:.3f}, WR={r['wr']:.1f}%, Trades={r['trades']}")

    # After: 本物のextract_levels_binnedに戻す
    levels_mod.extract_levels_binned = extract_levels_binned
    importlib.reload(yagami_mod)
    from lib.yagami import sig_yagami_A as sig_A_after, sig_yagami_B as sig_B_after
    from lib.yagami import sig_yagami_london_ny as sig_lon_after

    print("  After (新C1=extract_levels_binned) バックテスト実行中...")
    results_after = []
    for factory, lbl in [(sig_A_after, 'Yagami_A_After'),
                          (sig_B_after, 'Yagami_B_After'),
                          (sig_lon_after, 'Yagami_LonNY_After')]:
        r = run_bt_with_sig_func(df, factory, '4h', lbl)
        results_after.append(r)
        print(f"    {lbl}: PF={r['pf']:.3f}, WR={r['wr']:.1f}%, Trades={r['trades']}")

    # ──────────────────────────────────────────────────────
    # Step 3: 結果保存
    # ──────────────────────────────────────────────────────
    all_results = results_before + results_after
    df_results = pd.DataFrame(all_results)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    out_path = os.path.join(ROOT, 'results', 'run004_p0_c1_fix.csv')
    df_results.to_csv(out_path, index=False)
    print(f"\n  結果保存: {out_path}")

    # ──────────────────────────────────────────────────────
    # Step 4: サマリー出力
    # ──────────────────────────────────────────────────────
    before_pf = np.mean([r['pf'] for r in results_before if r['trades'] > 0]) if any(r['trades'] > 0 for r in results_before) else 0
    after_pf = np.mean([r['pf'] for r in results_after if r['trades'] > 0]) if any(r['trades'] > 0 for r in results_after) else 0
    before_wr = np.mean([r['wr'] for r in results_before if r['trades'] > 0]) if any(r['trades'] > 0 for r in results_before) else 0
    after_wr = np.mean([r['wr'] for r in results_after if r['trades'] > 0]) if any(r['trades'] > 0 for r in results_after) else 0

    print("\n" + "=" * 70)
    print("【C1充足率比較】")
    print(f"  Before: {rate_before:.1f}%  →  After: {rate_after:.1f}%  (Δ={rate_after - rate_before:+.1f}%)")
    print()
    print("【バックテスト比較】")
    print(f"{'戦略':<30} {'PF':>6} {'WR%':>6} {'Sharpe':>8} {'MDD%':>6} {'N':>5}")
    print("-" * 65)
    for r in all_results:
        err = f" [ERR: {r.get('error','')[:30]}]" if 'error' in r else ""
        print(f"{r['strategy']:<30} {r['pf']:>6.3f} {r['wr']:>6.1f} {r['sharpe']:>8.3f} {r['mdd']:>6.1f} {r['trades']:>5}{err}")
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
    }


if __name__ == '__main__':
    main()
