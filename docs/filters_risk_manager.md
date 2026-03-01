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

### 適用範囲に関する重要な知見 (v9検証結果)

**結論:**

> **本フィルターは逆張り要素を含む戦略（Yagami系など）でのドローダウン抑制に効果が期待できる。
> 一方、純粋なトレンドフォロー戦略（Maedai系など）では、エントリーシグナル自体がフィルター条件と
> 重なるため、効果は限定的である。**

**根拠:**

Donchianブレイク戦略（DC30_EMA200）は、ゴールドの上昇相場（= USD弱い相場）で
シグナルを発する。このため、「USD強い時にロング除去」というフィルターが発動する
タイミングに、そもそもDCブレイクシグナルが存在しない。結果として、フィルターが
除去すべき対象トレードがゼロとなり、効果が現れない。

一方、RSI押し目エントリー（Yagami系）は相場の反転局面で機能するため、
USDが強い局面での誤ったロングエントリーが発生しうる。そこにフィルターを
適用することで、質の低いトレードを除去できる。

**v9 検証比較表 (XAUUSD 2023-2026, 4H/1H):**

| 戦略タイプ | 戦略名 | PF (前) | MDD% (前) | Sharpe (前) | PF (後) | MDD% (後) | Sharpe (後) | 効果 |
|:----------:|--------|:-------:|:---------:|:-----------:|:-------:|:---------:|:-----------:|------|
| Maedai (トレンドフォロー) | DC30_EMA200 | 2.495 | 10.5% | 1.414 | 2.495 | 10.5% | 1.414 | **なし** |
| Yagami (逆張り要素あり) | YagamiFull_1H | 1.196 | 30.8% | 0.748 | 1.200 | 29.0% | 0.749 | **MDD -1.8%** |

**適用ガイドライン:**

| 戦略タイプ | USD強弱フィルター適用 | 推奨 |
|:----------:|:--------------------:|------|
| Donchianブレイクアウト系 | 不要 | 適用しない |
| RSI押し目・逆張り系 | 有効 | `threshold=75` で適用 |
| Union（複合）系 | 任意 | 有効だが効果は限定的 |

---

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

## 戦略ごとの季節フィルター設定 (v9 決定事項)

v8バックテスト結果に基づき、季節フィルターを戦略ごとに最適化:

| 戦略 | 季節フィルター | 理由 |
|------|--------------|------|
| **YagamiFull_1H** | `SEASON_SKIP_JUL_SEP` (7月+9月除外) | v8実績: 7月・9月が損失月として確認 |
| **YagamiA_4H** | `SEASON_ALL` (全月対象) | v8実績: 9月がプラス月 → 除外は機会損失 |
| **DC30_EMA200** | `SEASON_ALL` (全月対象) | Maedai系はトレンドフォロー、季節性低 |
| **Union_4H** | `SEASON_ALL` (全月対象) | Union戦略は季節フィルター不要 |

**v9 バックテスト検証結果 (2023-2026, XAUUSD_4H/1H):**

| 比較 | PF | MDD% | Sharpe | Calmar | Trades |
|------|:--:|:----:|:------:|:------:|:------:|
| YagamiFull_1H (全月) | 1.196 | 30.8% | 0.748 | 1.089 | 129 |
| YagamiFull_1H (7月+9月除外) | 1.163 | 30.5% | 0.666 | 0.873 | 121 |

> **注意**: 長期データ(2023-2026)では7月+9月の影響が限定的だが、
> v8(2025単年)での損失月確認を優先し、7月+9月除外をデフォルト採用。

## USD強弱フィルター横展開 (v9 検証結果)

v8で YagamiB_4H の Sharpe を -0.063 → 0.342 に改善したフィルターを横展開:

**検証結果 (2023-2026, XAUUSD_4H/1H):**

| 戦略 | PF | MDD% | Sharpe | Calmar | Trades |
|------|:--:|:----:|:------:|:------:|:------:|
| DC30_EMA200 | 2.495 | 10.5% | 1.414 | 3.877 | 19 |
| DC30_EMA200+USD | 2.495 | 10.5% | 1.414 | 3.877 | 19 |
| YagamiFull_1H | 1.196 | 30.8% | 0.748 | 1.089 | 129 |
| YagamiFull_1H+USD | 1.200 | 29.0% | 0.749 | 1.163 | 130 |

**考察:**
- **DC30_EMA200**: Donchianブレイクはゴールドラリー時に発生するため、USD強時と重ならない。フィルター効果なし。
- **YagamiFull_1H+USD**: MDD が 30.8% → 29.0% に微改善 (-1.8%)。Calmar も 1.089 → 1.163 に改善。

---

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
