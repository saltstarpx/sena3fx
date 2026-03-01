# Manus AI 変更履歴

このファイルは、Manus AIがリポジトリに対して行った変更を記録するログです。
Claude Codeとの連携において、何が変更されたかを把握するために使用してください。

---

## [2026-03-01] 開発フレームワークの刷新

**変更者:** Claude Code
**変更種別:** 開発フレームワークの刷新
**指示書:** prompt_for_claude_code_v8.md

### 変更内容

- **戦略ポートフォリオアプローチ導入**: 成功バイアス回避のため、3チーム体制による戦略ポートフォリオ (Yagami, Maedai, Risk Manager) アプローチを導入
- **USD強弱フィルター実装**: `strategies/market_filters.py` に `calc_usd_strength()` を実装。XAUUSDの逆モメンタムからUSD強弱プロキシを算出し、threshold=75 (上位25%) でロングシグナルを除去
- **評価指標追加**: Sharpe Ratio, Calmar Ratio を `lib/backtest.py` の `_report()` メソッドに追加
- **ドキュメントとファイル構成の分離**: `strategies/` ディレクトリに戦略別ファイル (`yagami_rules.py`, `maedai_breakout.py`, `market_filters.py`) を配置、`docs/` にチーム別ドキュメントを整備
- **Maedai戦略のDonchianパラメータ探索スクリプト追加**: `strategies/maedai_breakout.py` に `DC_PARAM_GRID` (DC期間: 10/15/20/30/40, EMA: 100/200) と `maedai_dc_variants()` を追加

### 追加・変更ファイル

| ファイル | 説明 |
|:---|:---|
| `strategies/__init__.py` | 戦略ポートフォリオ ハブ |
| `strategies/market_filters.py` | Teammate C: USD強弱フィルター + 季節フィルター |
| `strategies/yagami_rules.py` | Teammate A: Yagami戦略バリアント |
| `strategies/maedai_breakout.py` | Teammate B: Maedai戦略 + Donchianパラメータグリッド |
| `lib/backtest.py` | `_report()` に Sharpe Ratio / Calmar Ratio 追加 |
| `docs/strategy_yagami.md` | Teammate A 戦略ドキュメント |
| `docs/strategy_maedai.md` | Teammate B 戦略ドキュメント |
| `docs/filters_risk_manager.md` | Teammate C フィルター・リスク管理ドキュメント |
| `overheat_monitor.py` | XAUT/XAUUSD 過熱度モニター |
| `price_zone_analyzer.py` | 価格帯滞在時間ヒストグラム (薄いゾーン検出) |

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
