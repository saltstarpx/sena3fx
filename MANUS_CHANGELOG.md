# Manus AI 変更履歴

このファイルは、Manus AIがリポジトリに対して行った変更を記録するログです。
Claude Codeとの連携において、何が変更されたかを把握するために使用してください。

---

## [2026-03-01] v9成果の統合・フォワードテスト準備・戦略ダッシュボード構築

**変更者:** Claude Code
**変更種別:** v9成果の統合、フォワードテスト準備、戦略ダッシュボード構築
**指示書:** prompt_for_claude_code_v10_updated.md

### 変更内容

- **mainブランチへのマージ**: `claude/add-trading-backtest-ePJat` → `main` のローカルマージ完了。v8-v9の全成果を main に統合 (125ファイル)。リモートへのプッシュは `claude/` ブランチ制約のため dev ブランチ経由で管理。
- **Union戦略フォワードテスト監視**: `monitors/forward_union.py` を新規作成。最新OHLCを読み込みUnion戦略シグナルをログ出力（1時間cronまたはwatchモード対応）。シグナルは `trade_logs/forward_union_signals.csv` に記録。
- **USD強弱フィルター適用ガイドライン**: `docs/filters_risk_manager.md` の「USD強弱フィルター」セクションに「逆張り系に有効、トレンドフォロー系に効果限定的」という知見と適用ガイドラインを追記。v9検証比較表を根拠として引用。
- **バックテスト自動ログ**: `lib/backtest.py` の `_report()` を修正し、バックテスト完了時に `results/performance_log.csv` へ自動追記する `_append_performance_log()` を追加。
- **戦略ダッシュボード**: `dashboard.html` を新規作成。Plotly.js により Sharpe Ratio 棒グラフ・MDD 棒グラフ・リスク/リターン散布図・全戦略テーブルを表示。インライン CSV と動的フェッチの両対応。

### 主要バックテスト結果 (v10 シード)

| 戦略 | TF | Sharpe | PF | MDD% | WR% | Trades | 判定 |
|------|:--:|:------:|:--:|:----:|:---:|:------:|------|
| Union_4H | 4H | **2.817** | 3.6245 | 9.82% | 66.7% | 21 | PASS |
| DC30_EMA200 | 4H | 1.414 | 2.4948 | 10.52% | 57.9% | 19 | PASS |
| DC30_EMA200+USD | 4H | 1.414 | 2.4948 | 10.52% | 57.9% | 19 | PASS |
| YagamiFull_1H | 1H | 0.748 | 1.1958 | 30.8% | 35.7% | 129 | CHECK |
| YagamiFull_1H_S | 1H | 0.666 | 1.1627 | 30.52% | 35.5% | 121 | CHECK |
| YagamiA_4H | 4H | 0.668 | 1.1076 | 49.9% | 40.2% | 164 | CHECK |

### 追加・変更ファイル

| ファイル | 説明 |
|:---|:---|
| `monitors/__init__.py` | monitors パッケージ初期化 |
| `monitors/forward_union.py` | Union戦略フォワードテスト監視スクリプト |
| `docs/filters_risk_manager.md` | USD強弱フィルター適用ガイドライン追記 |
| `lib/backtest.py` | `_append_performance_log()` 追加 (auto CSV log) |
| `results/performance_log.csv` | バックテスト自動ログ (v10シード実行済み) |
| `dashboard.html` | 戦略パフォーマンス ダッシュボード (Plotly.js) |

---

---

## [2026-03-01] v8成果の検証と戦略確立

**変更者:** Claude Code
**変更種別:** v8成果の検証と戦略確立
**指示書:** prompt_for_claude_code_v9.md

### 変更内容

- **v8成果物のプッシュ確認**: `strategies/`・`docs/` は `claude/add-trading-backtest-ePJat` ブランチ上に既にプッシュ済みを確認
- **Union戦略の確立**: `strategies/union_strategy.py` として単独実行可能スクリプトを作成。`docs/strategy_union.md` にロジックと実績を記録。v9再現バックテストで Sharpe 2.817 (目標1.5超え) を確認
- **USD強弱フィルター横展開**: DC30_EMA200 (Maedai) と YagamiFull_1H (Yagami) にフィルター適用。DC30はDonchianブレイクがUSD強時と重ならないためフィルター効果なし。YagamiFull_1H ではMDD 30.8%→29.0% (-1.8%) に微改善
- **季節フィルターの戦略別最適化**: YagamiFull_1H は 7月+9月除外 (`SEASON_SKIP_JUL_SEP`) をデフォルト採用。YagamiA_4H は 9月がプラス月のため全月対象 (`SEASON_ALL`) を採用。`docs/filters_risk_manager.md` に記録

### v9 主要バックテスト結果

**Union戦略 (XAUUSD 2025 4H):**

| 戦略 | PF | WR% | MDD% | Sharpe | Calmar |
|------|:--:|:---:|:----:|:------:|:------:|
| Union_4H (素) | 3.624 | 66.7% | 9.8% | **2.817** | 13.709 |
| Union_4H+USD | 4.025 | 66.7% | 10.5% | 2.686 | 10.681 |

**USD強弱フィルター横展開 (XAUUSD 2023-2026 4H/1H):**

| 戦略 | PF | MDD% | Sharpe | Calmar | 変化 |
|------|:--:|:----:|:------:|:------:|------|
| DC30_EMA200 | 2.495 | 10.5% | 1.414 | 3.877 | — |
| DC30_EMA200+USD | 2.495 | 10.5% | 1.414 | 3.877 | 変化なし |
| YagamiFull_1H | 1.196 | 30.8% | 0.748 | 1.089 | — |
| YagamiFull_1H+USD | 1.200 | 29.0% | 0.749 | 1.163 | MDD -1.8% |

### 追加・変更ファイル

| ファイル | 説明 |
|:---|:---|
| `strategies/union_strategy.py` | Union戦略 単独実行スクリプト |
| `docs/strategy_union.md` | Union戦略ドキュメント (ロジック・パラメータ・実績) |
| `docs/filters_risk_manager.md` | 戦略別季節フィルター決定事項 + USD横展開結果を追記 |

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

---

## [2026-03-02] Claude Code指示書v11追加：「バックテスト vs 現実」— 実績データとの照合

**変更者:** Manus AI
**変更種別:** 指示書追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `prompt_for_claude_code_v11.md` | Claude Code指示書v11（実績データとの照合） |

### 変更理由

Claude Codeがリポジトリに統合した2,642件の実際のCFD取引履歴を分析した結果、**+1,758万円**という驚異的な利益と**勝率70.6%**という高いパフォーマンスが確認された。この「実績」をGround Truth（正解データ）とし、バックテスト戦略がどの程度現実を再現できるかを検証するフェーズに移行する。

### 主なタスク指示

1.  **2026年1月のドローダウン分析:** 実績で-112万円の損失が出た1月を、Union戦略のバックテストで再現できるか検証する。
2.  **実績トレードと戦略シグナルの照合:** 利益が最大だった2025年12月の実績勝ちトレードと、Union戦略の買いシグナルがどの程度一致するかを分析する。
3.  **戦略ダッシュボードへの「現実」追加:** `dashboard.html`に現実の月次損益グラフを追加し、バックテストと現実のパフォーマンスを比較できるようにする。

### Claude Codeへの指示

`prompt_for_claude_code_v11.md`を読み、記載されたタスクを順次実行すること。
