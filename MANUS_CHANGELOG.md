# Manus AI 変更履歴

このファイルは、Manus AIがリポジトリに対して行った変更を記録するログです。
Claude Codeとの連携において、何が変更されたかを把握するために使用してください。

---

## [2026-03-01] 開発フレームワークの刷新 — 戦略ポートフォリオ導入

**変更者:** Claude Code (prompt_for_claude_code_v8.md)
**変更種別:** 開発フレームワークの刷新

### 変更内容

1. **成功バイアス回避**: 戦略ポートフォリオ (Yagami, Maedai, Risk Manager) アプローチを導入
2. **USD強弱フィルター**: `strategies/market_filters.py` に実装。XAUUSD逆モメンタムからUSD強弱を算出、上位25%でロング回避
3. **評価軸多元化**: `lib/backtest.py` の `_report()` に Sharpe Ratio, Calmar Ratio を追加
4. **ファイル構成分離**:
   - `strategies/` ディレクトリ新設 (market_filters.py, yagami_rules.py, maedai_breakout.py)
   - `docs/` に戦略別ドキュメント (strategy_yagami.md, strategy_maedai.md, filters_risk_manager.md)
5. **Maedai DCパラメータ探索**: DC30+EMA200 が Sharpe=1.060, Calmar=3.558 で最適
6. **季節フィルター再検証**: 7+9月除外で YagamiFull DD 40.2%→29.9%, Sharpe 1.545→1.581

### バックテスト結果サマリー

| 指標 | 素 | +USD Filter | 差分 |
|:---|:---|:---|:---|
| 平均PF | 1.790 | 1.881 | +0.091 |
| 平均DD | 29.7% | 27.5% | -2.3% |
| 平均Sharpe | 1.164 | 1.212 | +0.048 |

### 追加・変更ファイル

| ファイル | 説明 |
|:---|:---|
| `strategies/__init__.py` | 戦略ポートフォリオ ハブ |
| `strategies/market_filters.py` | USD強弱フィルター + 季節フィルター |
| `strategies/yagami_rules.py` | Teammate A: Yagami戦略バリアント |
| `strategies/maedai_breakout.py` | Teammate B: Maedai戦略バリアント |
| `scripts/backtest_portfolio.py` | 統合バックテスト (Task 1-3) |
| `lib/backtest.py` | Sharpe/Calmar Ratio 追加 |
| `docs/strategy_yagami.md` | Yagami戦略ドキュメント |
| `docs/strategy_maedai.md` | Maedai戦略ドキュメント |
| `docs/filters_risk_manager.md` | Risk Managerドキュメント |

---

## [2026-02-27] 本物のマーケットデータ追加

**変更者:** Manus AI
**変更種別:** データ追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `data/ohlc/XAUUSD_1d.csv` | 日足データ (2019-2026, 1,801バー) |
| `data/ohlc/XAUUSD_1h.csv` | 1時間足データ (2023-2026, 13,693バー) |
| `data/ohlc/XAUUSD_4h.csv` | 4時間足データ (2023-2026, 3,714バー) |
| `data/ohlc/XAUUSD_8h.csv` | 8時間足データ (2023-2026, 1,937バー) |
| `data/ohlc/README.md` | データの説明書 |
| `MANUS_CHANGELOG.md` | この変更履歴ファイル |

### 変更理由

Claude Codeが外部API（Dukascopy等）にアクセスできず、GBM（幾何ブラウン運動）による合成データを使用してバックテストを行っていた。合成データでのバックテスト結果は本番環境での再現性が保証されないため、Manus側でYahoo Finance（GC=F: 金先物）から本物のマーケットデータを取得し、リポジトリに追加した。

### Claude Codeへの指示

1. `data/ohlc/` 配下のCSVファイルをバックテストのデータソースとして使用すること
2. 合成データ（GBM生成）は破棄し、本物のデータに完全に置き換えること
3. `scripts/fetch_data.py` のデータ読み込みパスを `data/ohlc/` に変更すること

### データ品質

- ソース: Yahoo Finance (GC=F)
- NULL値: なし（全ファイル検証済み）
- 価格範囲: $1,266 〜 $5,586（日足基準）
- 注意: GC=F（金先物）とXAUUSD（スポット金）には微小な価格差がある
