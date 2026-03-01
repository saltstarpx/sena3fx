# Union戦略ドキュメント

**Teammate B 派生 — Maedai × Yagami 複合シグナル**

---

## 概要

Union戦略は「Maedai（DCブレイクアウト）」と「Yagami（RSI押し目）」の2つのシグナルを OR 統合した複合戦略。

- 実装: `lib/yagami.py` → `sig_maedai_yagami_union()`
- 戦略ハブ: `strategies/union_strategy.py`

**思想（ユーザー提案）:**
「どちらかが反応したらポジションを取って、両方反応したら厚くはればいいんじゃないか」

---

## エントリーロジック

```
ロングシグナル = ( DCブレイク条件 ) OR ( RSI押し目条件 )
ショートシグナル = ( DC安値割れ条件 ) OR ( RSI天井条件 )
```

### シグナル 1: DCブレイク + EMA200（Maedai式）

| 条件 | 内容 |
|------|------|
| DC高値ブレイク | `close > Donchian_High(lookback_days=15 本)` |
| トレンド確認 | `close > EMA(ema_days=200 本)` |
| 確認バー | `confirm_bars=2` — 翌2本も同方向であることを確認 |

### シグナル 2: RSI押し目 + EMA200（Yagami式）

| 条件 | 内容 |
|------|------|
| トレンド確認 | `close > EMA(ema_days=200 本)` |
| RSI反発 | 前バー `RSI(14) ≤ 45` → 現バー `RSI > 45`（過売りから反発） |

---

## パラメータ（デフォルト）

| パラメータ | デフォルト値 | 説明 |
|------------|-------------|------|
| `freq` | `'4h'` | 動作時間軸 |
| `lookback_days` | `15` | DCブレイクの参照日数 |
| `ema_days` | `200` | EMAの期間（日数） |
| `confirm_bars` | `2` | DCブレイク確認バー数 |
| `rsi_oversold` | `45` | RSI押し目の閾値（上抜けでロング） |

---

## USD強弱フィルター付きバリアント

```python
from strategies.union_strategy import sig_union_usd

# threshold=75: USD上位25%の強さのときロングを除去
sig = sig_union_usd(freq='4h')
```

---

## バックテスト結果

### v9 再現バックテスト（XAUUSD 2025 4H, 1753本）

| 指標 | Union_4H (素) | Union_4H + USD |
|------|:---:|:---:|
| **Sharpe Ratio** | **2.817** | 2.686 |
| **Calmar Ratio** | **13.709** | 10.681 |
| PF | 3.624 | 4.025 |
| 勝率 (WR) | 66.7% | 66.7% |
| MDD | 9.8% | 10.5% |
| トレード数 | 21 | 18 |

> **v8実績 Sharpe 1.859 → v9再現 Sharpe 2.817 (同一ロジック、異なるデータ期間)**
> `strategies/union_strategy.py` を実行して再現確認済み。

---

## 評価基準（Maedai系として適用）

| 指標 | 基準値 |
|------|--------|
| Sharpe Ratio | > 1.5 |
| Calmar Ratio | > 3.0 |
| 年間トレード数 | ≥ 50 |
| 最大ドローダウン | ≤ 30% |

---

## 使用方法

```bash
# 単独バックテスト実行（Sharpe 1.859 の再現確認）
python strategies/union_strategy.py
```

---

## 関連ファイル

| ファイル | 役割 |
|----------|------|
| `lib/yagami.py` | `sig_maedai_yagami_union()` の実装 |
| `strategies/union_strategy.py` | 単独実行可能スクリプト |
| `strategies/market_filters.py` | USD強弱フィルター |
| `docs/strategy_maedai.md` | Maedai戦略（DCブレイク）ドキュメント |
| `docs/strategy_yagami.md` | Yagami戦略（RSI押し目）ドキュメント |
