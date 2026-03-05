"""
RUN-20260305-008: C4条件根本見直し + B評価緩和バックテスト
Claude Code連絡（EntryID 20260305-006）への対応

検証内容:
1. C4充足率の改善: パターン追加（ピンバー、インサイドバーブレイク、MA乖離）
2. B評価緩和（3条件→2条件）でWalk-Forward再検証
3. C4をオプション扱い（C1+C2+C3で3条件=B評価）
"""
import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib.backtest import BacktestEngine
from lib.yagami import yagami_signal, analyze_bars
from lib.levels import extract_levels_binned, is_at_level

DATA_DIR = os.path.join(ROOT, 'data', 'ohlc')
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_ohlc(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            return None
    return df.dropna(subset=['open', 'high', 'low', 'close'])

# ── C4拡張版シグナル関数 ──────────────────────────────────────────────
def sig_yagami_c4_extended(min_grade='B'):
    """
    C4条件を拡張したやがみシグナル。
    追加パターン:
    - ピンバー（単体ローソク足レベル）
    - インサイドバーブレイク（前足の高値/安値を抜けた）
    - MA乖離（EMA20から1ATR以上乖離→リバーサル候補）
    - ダブルトップ/ボトム（C3と重複するが、C4でも認識）
    """
    def _f(bars):
        df = analyze_bars(bars, freq='4h')
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)

        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd.Series(tr).rolling(14).mean().values
        ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values

        signals = pd.Series(index=df.index, dtype=object)
        levels_cache = None
        levels_update_idx = -100
        min_score = {'A': 4, 'B': 3, 'C': 2}.get(min_grade, 3)

        for i in range(20, n):
            if np.isnan(atr[i]) or atr[i] == 0:
                continue
            if df['trendless'].iloc[i]:
                continue

            if i - levels_update_idx >= 10:
                levels_cache = extract_levels_binned(
                    df.iloc[max(0, i-100):i+1], n_bins=40, min_freq_pct=0.06)
                levels_update_idx = i

            score = 0
            direction = 0

            # C1: レジサポ
            at_level, level_type = is_at_level(c[i], levels_cache, atr[i], 0.8)
            if at_level:
                score += 1
                if level_type == 'support':
                    direction += 1
                elif level_type == 'resistance':
                    direction -= 1

            # C2: ローソク足強弱
            ctype = df['candle_type'].iloc[i]
            strong_bull = ('big_bull', 'engulf_bull', 'hammer', 'pinbar_bull')
            strong_bear = ('big_bear', 'engulf_bear', 'inv_hammer', 'pinbar_bear')
            if ctype in strong_bull:
                score += 1; direction += 1
            elif ctype in strong_bear:
                score += 1; direction -= 1
            elif ctype == 'pullback_bull':
                direction -= 1; score += 1
            elif ctype == 'pullback_bear':
                direction += 1; score += 1

            # C3: プライスアクション
            pa = df['pa_signal'].iloc[i]
            if pa in ('reversal_low', 'double_bottom', 'body_align_support'):
                score += 1; direction += 1
            elif pa in ('reversal_high', 'double_top', 'body_align_resist'):
                score += 1; direction -= 1
            elif pa == 'wick_fill_bear':
                score += 1; direction -= 1
            elif pa == 'wick_fill_bull':
                score += 1; direction += 1

            # C4（拡張版）: チャートパターン + 追加条件
            cp = df['chart_pattern'].iloc[i]
            c4_hit = False
            c4_dir = 0

            # 既存パターン
            if cp in ('inv_hs_long', 'flag_bull', 'wedge_bull',
                      'triangle_break_bull', 'ascending_tri'):
                c4_hit = True; c4_dir = 1
            elif cp in ('hs_short', 'flag_bear', 'wedge_bear',
                        'triangle_break_bear', 'descending_tri'):
                c4_hit = True; c4_dir = -1

            # 追加1: インサイドバーブレイク（前足レンジを今足が抜けた）
            if not c4_hit and i >= 2:
                prev_h = h[i-1]
                prev_l = l[i-1]
                prev_range = prev_h - prev_l
                # 前足がインサイドバー（前々足に包まれている）
                if h[i-1] < h[i-2] and l[i-1] > l[i-2]:
                    if c[i] > prev_h + atr[i] * 0.1:
                        c4_hit = True; c4_dir = 1
                    elif c[i] < prev_l - atr[i] * 0.1:
                        c4_hit = True; c4_dir = -1

            # 追加2: EMA20乖離リバーサル（EMA20から2ATR以上離れた後の戻り）
            if not c4_hit and not np.isnan(ema20[i]):
                dist = c[i] - ema20[i]
                if dist < -atr[i] * 2.0 and c[i] > c[i-1]:  # 下方乖離→反発
                    c4_hit = True; c4_dir = 1
                elif dist > atr[i] * 2.0 and c[i] < c[i-1]:  # 上方乖離→反落
                    c4_hit = True; c4_dir = -1

            # 追加3: 3連続同方向足後の反転（やがみ: 「3本連続後の反転」）
            if not c4_hit and i >= 3:
                three_bull = all(c[j] > o[j] for j in range(i-3, i))
                three_bear = all(c[j] < o[j] for j in range(i-3, i))
                if three_bull and c[i] < o[i]:  # 3連陽後の陰線
                    c4_hit = True; c4_dir = -1
                elif three_bear and c[i] > o[i]:  # 3連陰後の陽線
                    c4_hit = True; c4_dir = 1

            if c4_hit:
                score += 1
                direction += c4_dir

            # C5: 足更新タイミング
            if df['bar_update'].iloc[i]:
                score += 1

            # グレード判定・シグナル生成
            if score >= min_score:
                if direction > 0:
                    signals.iloc[i] = 'long'
                elif direction < 0:
                    signals.iloc[i] = 'short'

        return signals
    return _f


# ── B評価緩和（C4なしでも3条件=B評価）シグナル ──────────────────────
def sig_yagami_no_c4(min_grade='B'):
    """
    C4条件を除外した4条件（C1+C2+C3+C5）でのやがみシグナル。
    B評価=2条件以上（実質的な緩和）。
    """
    def _f(bars):
        df = analyze_bars(bars, freq='4h')
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        n = len(df)

        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd.Series(tr).rolling(14).mean().values

        signals = pd.Series(index=df.index, dtype=object)
        levels_cache = None
        levels_update_idx = -100
        # C4除外のため最大スコアは4 → B評価=2条件以上
        min_score = {'A': 3, 'B': 2, 'C': 1}.get(min_grade, 2)

        for i in range(20, n):
            if np.isnan(atr[i]) or atr[i] == 0:
                continue
            if df['trendless'].iloc[i]:
                continue

            if i - levels_update_idx >= 10:
                levels_cache = extract_levels_binned(
                    df.iloc[max(0, i-100):i+1], n_bins=40, min_freq_pct=0.06)
                levels_update_idx = i

            score = 0
            direction = 0

            # C1
            at_level, level_type = is_at_level(c[i], levels_cache, atr[i], 0.8)
            if at_level:
                score += 1
                if level_type == 'support': direction += 1
                elif level_type == 'resistance': direction -= 1

            # C2
            ctype = df['candle_type'].iloc[i]
            strong_bull = ('big_bull', 'engulf_bull', 'hammer', 'pinbar_bull')
            strong_bear = ('big_bear', 'engulf_bear', 'inv_hammer', 'pinbar_bear')
            if ctype in strong_bull: score += 1; direction += 1
            elif ctype in strong_bear: score += 1; direction -= 1
            elif ctype in ('pullback_bull', 'pullback_bear'):
                score += 1
                direction += (1 if ctype == 'pullback_bear' else -1)

            # C3
            pa = df['pa_signal'].iloc[i]
            if pa in ('reversal_low', 'double_bottom', 'body_align_support'):
                score += 1; direction += 1
            elif pa in ('reversal_high', 'double_top', 'body_align_resist'):
                score += 1; direction -= 1
            elif pa in ('wick_fill_bear', 'wick_fill_bull'):
                score += 1
                direction += (1 if pa == 'wick_fill_bull' else -1)

            # C5（C4スキップ）
            if df['bar_update'].iloc[i]:
                score += 1

            if score >= min_score:
                if direction > 0: signals.iloc[i] = 'long'
                elif direction < 0: signals.iloc[i] = 'short'

        return signals
    return _f


# ── バックテスト実行 ──────────────────────────────────────────────────
ENGINE_KWARGS = dict(
    init_cash=5_000_000,
    risk_pct=0.02,
    default_sl_atr=1.5,
    default_tp_atr=4.5,
    trail_start_atr=2.0,
    trail_dist_atr=1.0,
    pyramid_entries=0,
    use_dynamic_sl=True,
    sl_n_confirm=2,
    sl_min_atr=0.5,
)

CONFIGS = [
    ('XAUUSD_4h.csv', '4h', 'Yagami_A_original',   lambda bars: yagami_signal(bars, '4h', min_grade='A')),
    ('XAUUSD_4h.csv', '4h', 'Yagami_B_original',   lambda bars: yagami_signal(bars, '4h', min_grade='B')),
    ('XAUUSD_4h.csv', '4h', 'Yagami_C4ext_A',      sig_yagami_c4_extended('A')),
    ('XAUUSD_4h.csv', '4h', 'Yagami_C4ext_B',      sig_yagami_c4_extended('B')),
    ('XAUUSD_4h.csv', '4h', 'Yagami_NoC4_B',       sig_yagami_no_c4('B')),
    ('XAUUSD_4h.csv', '4h', 'Yagami_NoC4_A',       sig_yagami_no_c4('A')),
    ('USDJPY_4h.csv', '4h', 'Yagami_C4ext_B_USDJPY', sig_yagami_c4_extended('B')),
    ('EURUSD_4h.csv', '4h', 'Yagami_C4ext_B_EURUSD', sig_yagami_c4_extended('B')),
]

print("=" * 70)
print("RUN-20260305-008: C4条件根本見直し + B評価緩和バックテスト")
print("=" * 70)

results = []
for fname, freq, name, sig_func in CONFIGS:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [SKIP] {fname}")
        continue
    df = load_ohlc(fpath)
    if df is None or len(df) < 100:
        print(f"  [SKIP] {fname} データ不足")
        continue

    engine = BacktestEngine(**ENGINE_KWARGS)
    try:
        res = engine.run(data=df, signal_func=sig_func, freq=freq, name=name)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        continue
    if res is None:
        print(f"  [SKIP] {name}: 結果なし")
        continue

    n = res.get('total_trades', 0)
    pf = res.get('profit_factor', 0)
    wr = res.get('win_rate_pct', 0)
    mdd = res.get('max_drawdown_pct', 0)
    total_ret = res.get('total_return_pct', 0)
    rr = res.get('rr_ratio', 0)
    passed = (pf >= 1.2 and n >= 15 and mdd <= 25)

    print(f"  {name:<35} PF={pf:.3f} WR={wr:.1f}% MDD={mdd:.1f}% N={n} {'✓' if passed else '✗'}")
    results.append({
        'strategy': name,
        'pf': round(pf, 3),
        'wr_pct': round(wr, 1),
        'mdd_pct': round(mdd, 1),
        'n_trades': n,
        'rr': round(rr, 2),
        'total_return_pct': round(total_ret, 2),
        'passed': passed,
    })

df_res = pd.DataFrame(results)
out_csv = os.path.join(RESULTS_DIR, 'run008_c4_b_eval.csv')
df_res.to_csv(out_csv, index=False)
print(f"\n結果保存: {out_csv}")

print("\n" + "=" * 70)
print("【比較サマリー】")
print(f"{'戦略':<35} {'PF':>6} {'WR%':>6} {'MDD%':>6} {'N':>5} {'判定'}")
print("-" * 70)
for _, row in df_res.sort_values('pf', ascending=False).iterrows():
    mark = '✓ PASS' if row['passed'] else '✗'
    print(f"  {row['strategy']:<33} {row['pf']:>6.3f} {row['wr_pct']:>6.1f} {row['mdd_pct']:>6.1f} {row['n_trades']:>5} {mark}")

print(f"\n合格: {df_res['passed'].sum()}/{len(df_res)} 戦略")
if len(df_res) > 0:
    best = df_res.loc[df_res['pf'].idxmax()]
    print(f"最良: {best['strategy']} PF={best['pf']:.3f}, WR={best['wr_pct']:.1f}%, N={best['n_trades']}")
print("=" * 70)
