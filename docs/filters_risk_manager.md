# Teammate C: Filters & Risk Manager Documentation

## 概要

ポートフォリオ全体のリスクを調整する市場環境フィルターの開発・評価。
Teammate A (Yagami) と Teammate B (Maedai) のシグナルに対して
市場環境に応じたフィルタリングを適用し、不利な環境でのトレードを回避する。

**評価基準**:
- ポートフォリオ全体の最大ドローダウン (MDD) 低減率
- ボラティリティ低減率
- Sharpe Ratio 改善率

実装: `strategies/market_filters.py`

## 1. USD強弱フィルター

### 概要

XAUUSDの逆モメンタムからUSD強弱のプロキシを算出し、
USDが強い期間のロングシグナルを除去する。

### 関数

| 関数 | 説明 |
|------|------|
| `calc_usd_strength(bars, lookback=20, rank_window=100)` | USD強弱スコア (0-100) を算出 |
| `usd_strength_filter(bars, threshold=75.0)` | `True` = USD強 → ロング回避 |
| `wrap_signal_with_usd_filter(sig_func, bars, threshold)` | 既存シグナルにフィルター適用 |
| `make_usd_filtered_signal(sig_factory, threshold=75)` | シグナルファクトリへのフィルター組込み |

### ロジック

```
1. XAUUSD の lookback 本リターンを算出
2. ローリングパーセンタイルランク (rank_window 本) に変換
3. USD強弱 = 100 - gold_momentum_percentile
   (金が下落 = USD強、金が上昇 = USD弱)
4. USD強弱 >= threshold のとき、ロングシグナルを除去
   (ショートシグナルは影響を受けない)
```

### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `threshold` | 75 | USD強弱の閾値 (上位25%でフィルター発動) |
| `lookback` | 20 | モメンタム計算の振り返り本数 |
| `rank_window` | 100 | パーセンタイルランクの窓幅 |

### 使用例

```python
from strategies.market_filters import make_usd_filtered_signal
from lib.yagami import sig_yagami_A

# ファクトリにフィルターを組み込む
filtered_sig = make_usd_filtered_signal(sig_yagami_A, threshold=75)
result = engine.run(data=df, signal_func=filtered_sig(freq='4h'))
```

## 2. 季節フィルター

### 概要

特定の月を除外することで、季節性に基づく不利な期間のトレードを回避する。
`BacktestEngine.run()` の `allowed_months` パラメータで制御。

### 定義済みプリセット

| 定数 | 許可月 | 説明 |
|------|--------|------|
| `SEASON_ALL` | `None` (全月) | フィルターなし |
| `SEASON_SKIP_JUL` | 7月除外 | 夏枯れ対策 (7月) |
| `SEASON_SKIP_JUL_SEP` | 7月+9月除外 | 夏枯れ + 9月の不安定さ回避 |
| `SEASON_ACTIVE` | 1-3月, 10-12月 | 冬季+秋季のみ (活発な時期) |
| `SEASON_SKIP_SUMMER` | 6-9月除外 | 夏季全般を除外 |

### 有効性評価

```python
from strategies.market_filters import seasonal_effectiveness

result = seasonal_effectiveness(trades, skip_months=(7, 9))
# result['included_months_pnl']  → 除外月以外のPnL合計
# result['excluded_months_pnl']  → 除外月のPnL合計
# result['monthly_breakdown']    → 月別のcount/pnl/wins
```

## 3. ボラティリティレジーム

### 概要

ATR比率 (現在ATR / 長期ATR) に基づいてボラティリティレジームを判定し、
極端なボラ環境でのトレードを回避する。

`sig_yagami_filtered()` および `sig_yagami_vol_regime()` で使用。

### ロジック

```
ATR比率 = ATR(14) / ATR(14).rolling(100).mean()

- ATR比率 < 0.6 → 低ボラ環境 → シグナル除去
- ATR比率 > 2.2 → 高ボラ環境 → シグナル除去
- 0.6 <= ATR比率 <= 2.2 → 通常環境 → シグナル通過
```

出典: botter Advent Calendar 2024 #22「消えたエッジの話」

## 4. トレンドレジーム

### 概要

EMA200の方向でトレンドレジームを判定し、逆トレンド方向のシグナルを除去する。

`sig_yagami_filtered()` および `sig_yagami_trend_regime()` で使用。

### ロジック

```
EMA200 の傾き (直近値 vs 5本前) で方向判定:
- EMA200 上昇中 → ショートシグナル除去
- EMA200 下降中 → ロングシグナル除去
```

出典: ラリーウィリアムズ式 x botter コミュニティ知見

## 5. 時刻アノマリー

### 概要

ロンドンオープン (08:00 UTC) / NYオープン (13:00-14:00 UTC) の
時間帯に限定したエントリーフィルター。

`sig_yagami_prime_time()` で使用。

### ロジック

- アジア時間 (00:00-07:00 UTC) のブレイクアウトは除外
- ロンドン/NYセッション時間のみシグナルを通過

出典: botter Advent Calendar 2024 時刻アノマリー記事

## フィルター統合

`sig_yagami_full_filter()` は上記すべてのフィルターを統合:

```python
sig_yagami_filtered(
    freq='1h',
    min_grade='B',
    use_vol_regime=True,    # ボラティリティレジーム
    use_trend_regime=True,  # トレンドレジーム
    use_time_filter=True,   # 時刻アノマリー
    use_momentum=True,      # MTFモメンタム
)
```

さらに `make_usd_filtered_signal()` でUSD強弱フィルターを追加可能。

## 評価方法

フィルターの効果は以下の指標で測定する:

| 指標 | 計算方法 | 目標 |
|------|----------|------|
| MDD低減率 | `1 - (filtered_MDD / unfiltered_MDD)` | > 20% |
| Sharpe改善率 | `filtered_Sharpe / unfiltered_Sharpe - 1` | > 15% |
| ボラ低減率 | `1 - (filtered_vol / unfiltered_vol)` | > 10% |

フィルター適用前後のバックテスト結果を比較し、
過度なトレード数減少なく上記指標が改善されていれば採用する。
