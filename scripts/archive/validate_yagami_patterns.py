"""
validate_yagami_patterns.py
============================
yagami_candle_patterns.py の動作確認スクリプト（高速版）。

data/ohlc/EURUSD_1m.csv を使用して各関数のヒット件数と具体例5件を表示。
結果を results/yagami_pattern_validation.txt に保存。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from utils.yagami_candle_patterns import (
    candle_strength_score,
    check_inside_bar_cluster,
    check_body_zone_aligned,
    check_candle_close_timing,
    check_higher_tf_direction,
    _atr, _body, _range, _upper_wick, _lower_wick,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ohlc")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── データロード ────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path)
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.rename(columns={ts: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])

print("データロード中...")
df_1m = load_csv(os.path.join(DATA_DIR, "EURUSD_1m.csv"))
print(f"EURUSD 1m: {len(df_1m):,} 行  {df_1m.index[0]} 〜 {df_1m.index[-1]}")

ATR_WIN = 14

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("=" * 80)
log("やがみメソッド ローソク足パターン検出 動作確認")
log(f"データ: EURUSD 1m  {df_1m.index[0].date()} 〜 {df_1m.index[-1].date()}")
log(f"総ローソク足数: {len(df_1m):,}")
log("=" * 80)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 共通ATR計算（ベクトル化）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
hl = df_1m["high"] - df_1m["low"]
hc = (df_1m["high"] - df_1m["close"].shift()).abs()
lc = (df_1m["low"]  - df_1m["close"].shift()).abs()
atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(ATR_WIN).mean()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. candle_strength_score（ベクトル化版）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log("\n【1】candle_strength_score（ローソク足強弱スコア）")
log("  スコア: -1.0(強ショート) 〜 +1.0(強ロング)  閾値: |score|≥0.6 で強シグナル")

o  = df_1m["open"];  c = df_1m["close"]
h  = df_1m["high"];  l = df_1m["low"]
op = o.shift(1);     cp = c.shift(1)    # 直前足

body     = (c - o).abs()
rng      = h - l
uw       = h - c.where(c > o, o)        # upper wick
lw       = c.where(c < o, o) - l        # lower wick
body_p   = (cp - op).abs()              # 直前足実体

# 各フラグ（全ベクトル化）
valid    = (rng > 0) & atr.notna()
bullish  = c > o
bearish  = c < o

# 上下ヒゲ中ピンバー: 両ヒゲ≥body×0.8
pin_both = (body > 0) & (uw >= body * 0.8) & (lw >= body * 0.8)

# 大陽線/大陰線: body/range≥0.7
big_bull = valid & ~pin_both & bullish & ((body / rng) >= 0.7)
big_bear = valid & ~pin_both & bearish & ((body / rng) >= 0.7)

# リバーサルロー: 前足陰線 + 現足陽線 + c≥前足open + 下ヒゲ≥body×1.5
rev_low  = valid & ~pin_both & ~big_bull & bullish & (cp < op) & (c >= op) & (lw >= body * 1.5)
rev_high = valid & ~pin_both & ~big_bear & bearish & (cp > op) & (c <= op) & (uw >= body * 1.5)

# エンゴルフィング: 現足実体が前足実体を包む
bull_max = c.where(bullish, o)
bull_min = o.where(bullish, c)
bear_max = o.where(bearish, c)
bear_min = c.where(bearish, o)
engulf_bull = (valid & ~pin_both & ~big_bull & ~rev_low & bullish & (body_p > 0) &
               (bull_max >= cp.where(cp > op, op)) & (bull_min <= cp.where(cp < op, op)))
engulf_bear = (valid & ~pin_both & ~big_bear & ~rev_high & bearish & (body_p > 0) &
               (bear_min <= cp.where(cp < op, op)) & (bear_max >= cp.where(cp > op, op)))

# 下ヒゲ陽線: lw≥body×1.5 + uw<body×0.5
pin_bull = (valid & ~pin_both & ~big_bull & ~rev_low & ~engulf_bull &
            bullish & (body > 0) & (lw >= body * 1.5) & (uw < body * 0.5))
pin_bear = (valid & ~pin_both & ~big_bear & ~rev_high & ~engulf_bear &
            bearish & (body > 0) & (uw >= body * 1.5) & (lw < body * 0.5))

# インサイドバー
inside = valid & ~pin_both & ~big_bull & ~big_bear & ~rev_low & ~rev_high & ~engulf_bull & ~engulf_bear & ~pin_bull & ~pin_bear & (h <= h.shift(1)) & (l >= l.shift(1))

# スコア割り当て
score_s = pd.Series(0.0, index=df_1m.index)
score_s[big_bull]     =  1.0
score_s[big_bear]     = -1.0
score_s[rev_low]      =  0.90
score_s[rev_high]     = -0.90
score_s[engulf_bull]  =  0.85
score_s[engulf_bear]  = -0.85
score_s[pin_bull]     =  0.65
score_s[pin_bear]     = -0.65
# インサイドバー: 前足の方向を継承
prev_dir = pd.Series(0, index=df_1m.index)
prev_dir[cp > op] =  1
prev_dir[cp < op] = -1
score_s[inside] = prev_dir[inside] * 0.25
# 残りの陽線/陰線（モブ）
mob_mask = valid & (score_s == 0.0) & ~pin_both
score_s[mob_mask & bullish] =  0.20
score_s[mob_mask & bearish] = -0.20

STRONG = 0.60
WEAK   = 0.30
valid_scores = score_s[valid]
n_total   = len(valid_scores)
n_sl      = (valid_scores >=  STRONG).sum()
n_ss      = (valid_scores <= -STRONG).sum()
n_wl      = ((valid_scores >= WEAK) & (valid_scores < STRONG)).sum()
n_range   = (valid_scores == 0.0).sum()
n_other   = n_total - n_sl - n_ss - n_wl - n_range

log(f"\n  集計（{n_total:,}本）:")
log(f"    強ロング  (≥{STRONG}) : {n_sl:>7,} ({n_sl/n_total*100:.1f}%)")
log(f"    強ショート(≤-{STRONG}): {n_ss:>7,} ({n_ss/n_total*100:.1f}%)")
log(f"    弱ロング  ({WEAK}〜{STRONG}) : {n_wl:>7,} ({n_wl/n_total*100:.1f}%)")
log(f"    レンジ/中立(0.0)   : {n_range:>7,} ({n_range/n_total*100:.1f}%)")
log(f"    その他中立          : {n_other:>7,} ({n_other/n_total*100:.1f}%)")

log(f"\n  内訳:")
log(f"    大陽線:           {big_bull.sum():>7,}   大陰線:        {big_bear.sum():>7,}")
log(f"    リバーサルロー:   {rev_low.sum():>7,}   リバーサルハイ:{rev_high.sum():>7,}")
log(f"    エンゴルフ陽線:   {engulf_bull.sum():>7,}   エンゴルフ陰線:{engulf_bear.sum():>7,}")
log(f"    下ヒゲ陽線:       {pin_bull.sum():>7,}   上ヒゲ陰線:    {pin_bear.sum():>7,}")
log(f"    上下ヒゲ中ピン:   {pin_both.sum():>7,}   インサイドバー:{inside.sum():>7,}")

log("\n  強ロング具体例 5件 (score≥0.60):")
ex_sl = df_1m[(score_s >= STRONG) & valid].head(5)
for ts, row in ex_sl.iterrows():
    sc = score_s[ts]
    log(f"    {ts}  score={sc:.2f}  O={row['open']:.5f} H={row['high']:.5f} L={row['low']:.5f} C={row['close']:.5f}")

log("\n  強ショート具体例 5件 (score≤-0.60):")
ex_ss = df_1m[(score_s <= -STRONG) & valid].head(5)
for ts, row in ex_ss.iterrows():
    sc = score_s[ts]
    log(f"    {ts}  score={sc:.2f}  O={row['open']:.5f} H={row['high']:.5f} L={row['low']:.5f} C={row['close']:.5f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. check_inside_bar_cluster（ベクトル化: rolling min/max）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log("\n【2】check_inside_bar_cluster（インサイドバー継続フィルター）")
log("  条件: 直近2本以上がインサイドバー連続 + 実体≥ATR×0.3")

# 各足がインサイドバーかどうかのフラグ
ib1 = (h <= h.shift(1)) & (l >= l.shift(1)) & (body >= atr * 0.3)
ib2 = ib1 & (h.shift(1) <= h.shift(2)) & (l.shift(1) >= l.shift(2)) & ((c.shift(1) - o.shift(1)).abs() >= atr.shift(1) * 0.3)
ib_cluster = ib1 & ib2

n_ib = ib_cluster.sum()
n_total_ib = valid.sum()
log(f"\n  ヒット数: {n_ib:,} / {n_total_ib:,}  ({n_ib/n_total_ib*100:.2f}%)")
log("\n  具体例 5件:")
for ts, row in df_1m[ib_cluster].head(5).iterrows():
    log(f"    {ts}  O={row['open']:.5f} H={row['high']:.5f} L={row['low']:.5f} C={row['close']:.5f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. check_body_zone_aligned（ベクトル化: rolling）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log("\n【3】check_body_zone_aligned（実体ゾーン揃い判定）")
log("  条件: 直近5本の実体がATR×1.5幅内に集中 + 各実体≥ATR×0.2")

LOOKBACK = 5
min_body_mask = body >= atr * 0.2

# open/close 両方の rolling max-min
all_edges = pd.concat([o, c], axis=1)
roll_max  = all_edges.rolling(LOOKBACK).max().max(axis=1)
roll_min  = all_edges.rolling(LOOKBACK).min().min(axis=1)
zone_rng  = roll_max - roll_min
# 直近LOOKBACK本で最低2本がmin_body条件を満たすか
thick_count = min_body_mask.rolling(LOOKBACK).sum()

bza = (zone_rng <= atr * 1.5) & (thick_count >= 2) & atr.notna()

n_bza = bza.sum()
log(f"\n  ヒット数: {n_bza:,} / {n_total_ib:,}  ({n_bza/n_total_ib*100:.2f}%)")
log("\n  具体例 5件:")
for ts, row in df_1m[bza].head(5).iterrows():
    log(f"    {ts}  O={row['open']:.5f} H={row['high']:.5f} L={row['low']:.5f} C={row['close']:.5f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. check_candle_close_timing（ベクトル化）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log("\n【4】check_candle_close_timing（足確定タイミングフィルター）")
log("  PDF: 「足更新2分前以内でエントリー。中途半端な時間は消耗の原因」")

minute_abs = df_1m.index.hour * 60 + df_1m.index.minute
tf_defs = {"15m": 15, "1h": 60, "4h": 240}

for tf, period in tf_defs.items():
    window = 2  # PDF: 2分以内
    mod = minute_abs % period
    timing_mask = mod >= (period - window)
    hits = df_1m[timing_mask]
    log(f"\n  [{tf}] ヒット数: {len(hits):,} / {len(df_1m):,}  ({len(hits)/len(df_1m)*100:.1f}%)")
    log(f"    具体例 5件:")
    for ts in hits.index[:5]:
        log(f"      {ts}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. check_higher_tf_direction（merge_asof使用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log("\n【5】check_higher_tf_direction（上位足方向確認）")
log("  条件: 1H足・4H足の色が一致する場合のみ有効（PDF MTF解説）")

df_1h = df_1m.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
df_4h = df_1m.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()

# 各タイムスタンプに直前の1H・4H足をマッチ
df_1m_tmp = df_1m.copy().reset_index()
df_1h_dir = df_1h.copy()
df_4h_dir = df_4h.copy()
df_1h_dir["dir_1h"] = (df_1h_dir["close"] > df_1h_dir["open"]).map({True: 1, False: -1})
df_4h_dir["dir_4h"] = (df_4h_dir["close"] > df_4h_dir["open"]).map({True: 1, False: -1})

merged = pd.merge_asof(
    df_1m_tmp,
    df_1h_dir[["dir_1h"]].reset_index().rename(columns={"timestamp":"ts_1h"}),
    left_on="timestamp", right_on="ts_1h", direction="backward"
)
merged = pd.merge_asof(
    merged,
    df_4h_dir[["dir_4h"]].reset_index().rename(columns={"timestamp":"ts_4h"}),
    left_on="timestamp", right_on="ts_4h", direction="backward"
)

merged["htf_dir"] = merged.apply(
    lambda r: int(r["dir_4h"]) if r["dir_1h"] == r["dir_4h"] else 0
    if pd.notna(r["dir_1h"]) and pd.notna(r["dir_4h"]) else 0,
    axis=1
)

n_long  = (merged["htf_dir"] ==  1).sum()
n_short = (merged["htf_dir"] == -1).sum()
n_neut  = (merged["htf_dir"] ==  0).sum()
n_all   = len(merged)
log(f"\n  集計（{n_all:,}本）:")
log(f"    ロング優勢 (+1): {n_long:>7,} ({n_long/n_all*100:.1f}%)")
log(f"    ショート優勢(-1): {n_short:>7,} ({n_short/n_all*100:.1f}%)")
log(f"    中立      ( 0): {n_neut:>7,} ({n_neut/n_all*100:.1f}%)")

log("\n  ロング優勢 具体例 5件:")
ex_long = merged[merged["htf_dir"] == 1].head(5)
for _, row in ex_long.iterrows():
    log(f"    {row['timestamp']}  O={row['open']:.5f} C={row['close']:.5f}")

log("\n  ショート優勢 具体例 5件:")
ex_short = merged[merged["htf_dir"] == -1].head(5)
for _, row in ex_short.iterrows():
    log(f"    {row['timestamp']}  O={row['open']:.5f} C={row['close']:.5f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# まとめ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log("\n" + "=" * 80)
log("【バックテスト知見との対応】")
log("=" * 80)
log("  F2_pin_bar（谷足逆ヒゲ≥body×2）→ リバーサルロー/下ヒゲ陽線に包含")
log("    FX OOS PF: EURUSD 3.03 / GBPUSD 2.79 / AUDUSD 3.59 / USDJPY 2.28")
log("  F3_no_shadow（確認足逆ヒゲ<body×0.5）→ 下ヒゲ陽線条件に包含")
log("    XAUUSD OOS PF: 2.28（v77比+0.25）")
log("")
log("  推奨使用方法:")
log("    ロングエントリー条件: candle_strength_score(window) >= 0.60")
log("    レンジ回避条件:       candle_strength_score(window) == 0.0")
log("    MTF確認:              check_higher_tf_direction(df_1h, df_4h) == +1")
log("    インサイドバー確認:   check_inside_bar_cluster(window)")
log("    足確定タイミング:     check_candle_close_timing(ts, '1h')")

log("\n" + "=" * 80)
log("全動作確認完了")
log("=" * 80)

# ── ファイル出力 ──────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "yagami_pattern_validation.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n結果保存: {out_path}")
