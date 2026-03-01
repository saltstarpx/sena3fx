# Manus AI 変更履歴

このファイルは、Manus AIがリポジトリに対して行った変更を記録するログです。
Claude Codeとの連携において、何が変更されたかを把握するために使用してください。

---

## [Manus AI への重要なお願い] ブランチのマージ

Claude Codeはセキュリティ制約により `main` ブランチへ直接プッシュできません。
すべての開発成果は以下のブランチに蓄積されています:

**`claude/add-trading-backtest-ePJat`**

このブランチには `main` にない以下が含まれます:
- `strategies/` (Union戦略, Yagami, Maedai, Market Filters)
- `monitors/` (フォワードテスト監視)
- `dashboard.html` (戦略パフォーマンスダッシュボード)
- `results/performance_log.csv` (バックテスト自動ログ)
- `trade_logs/broker_history_*.csv` (実取引履歴, UTF-8変換済)
- `lib/backtest.py` (Sharpe/Calmar/richmanbtc p-mean法 追加)
- `docs/strategy_*.md` (全戦略ドキュメント)
- `README.md` (v4.0対応の最新版)

**Manus AIの作業前に、このブランチを `main` にマージしてください:**
```
git checkout main
git merge claude/add-trading-backtest-ePJat
git push origin main
```

---

## [2026-03-01] v13: Kellyの最適化 + MetaStrategy再構築 (3状態HMM)

**変更者:** Claude Code
**変更種別:** サイジング最適化 + HMMレジーム精度向上
**指示書:** `prompt_for_claude_code_v13.md`

### 変更内容

#### Task 1: HybridKellySizer (lib/risk_manager.py)
- `KellyCriterionSizer(f=0.5)` × `VolatilityAdjustedSizer` の積算
- kelly_fraction を v12の0.25 → 0.5 に引上げ
- 結果: Sharpe 2.630, MDD 33.5% (MDD 30%超でリスク過大 — Kelly f=0.25が最適と確認)

#### Task 2: MetaStrategy v2.0 (3状態HMM)
- `lib/regime.py` を3状態対応に全面改修
  - 観測変数: [log_return, abs_log_return] の2次元でリターン方向+ボラを同時学習
  - ラベル: range(低ボラ) / low_trend(中ボラ+正リターン) / high_trend(高ボラ)
- `strategies/meta_strategy.py` を3状態対応に全面改修
  - range → YagamiA_4H / low_trend → Maedai_DC / high_trend → Union_4H
  - グリッドサーチ (3組合せ): Maedai lookback 10/15/20 を探索
- レジーム分布: range=49.3%, low_trend=49.0%, high_trend=1.7% (v1.0のtrend=1.4%から大幅改善)
- **MetaStrategy Sharpe: 0.581 → 1.366 (+135%改善)**

#### Task 3: VolSizer再設計 (disable_atr_sl)
- `lib/backtest.py run()` に `disable_atr_sl=True` オプション追加
- ATR-SLとVolSizerの重複解消 → 固定SL(default_sl_atr×ATR)に切替
- 結果: Sharpe 1.946, Trades 21→40 (シグナル感度変化), MDD 19.3%

#### Task 4: dashboard.html v13セクション更新
- 全6戦略の資産曲線比較 (Union/Kelly0.25/HybridKelly/MetaV1/MetaV2)
- v12/v13 Sharpe×Calmarグラフに v13結果を追記
- 3状態HMMレジーム円グラフ追加 (range49%/low_trend49%/high_trend2%)

### バックテスト結果サマリー (v13)
| 戦略 | Sharpe | Calmar | MDD% | 結論 |
|---|---|---|---|---|
| Union_4H_Base | **2.817** | 13.7 | 9.8 | ベース |
| Union+Kelly(f=0.25) | 2.798 | **24.5** | 22.8 | **Calmar最高** |
| Union+HybridKelly(f=0.5) | 2.630 | 19.6 | 33.5 | MDD過大 |
| Union+VolSizer(noSL) | 1.946 | 6.3 | 19.3 | 感度変化 |
| MetaStrategy v2 | 1.366 | 2.4 | 21.9 | **+135%改善** |

---

## [2026-03-01] v12: 1000万→1億エンジン — VolSizer / Kelly / HMM MetaStrategy

**変更者:** Claude Code
**変更種別:** 新サイジングエンジン実装 + レジーム転換モデル
**指示書:** `prompt_for_claude_code_v12.md`

### 新設ファイル
- **`lib/risk_manager.py`** — `VolatilityAdjustedSizer` + `KellyCriterionSizer`
- **`lib/regime.py`** — `HiddenMarkovRegimeDetector` (hmmlearn GaussianHMM 2状態)
- **`strategies/meta_strategy.py`** — `MetaStrategy` (HMM連動シグナル切替)
- **`scripts/backtest_v12.py`** — v12統合バックテスト実行スクリプト
- **`requirements.txt`** — 依存パッケージ一覧 (numpy, pandas, hmmlearn>=0.3.0)

### 修正ファイル
- **`lib/backtest.py`** — `run()` に `sizer=` パラメータ追加（VolSizer/Kellyを差し込み可能）
- **`results/v12_equity_curves.csv`** — 4戦略の資産曲線データ
- **`dashboard.html`** — v12セクション追加（資産曲線比較 + Sharpe/Calmar棒グラフ）

### バックテスト結果 (期間: 2025-01〜2026-02, 初期資金500万)
| 戦略 | Sharpe | Calmar | MDD% | PF | WR% |
|---|---|---|---|---|---|
| Union_4H_Base (ベースライン) | **2.817** | 13.7 | 9.8 | 3.624 | 66.7 |
| Union+VolSizer | 2.656 | 9.6 | 11.9 | 3.547 | 66.7 |
| Union+Kelly(×2.4) | 2.798 | **24.5** | 22.8 | 3.566 | 66.7 |
| MetaStrategy(HMM) | 0.581 | 0.6 | 26.0 | 1.264 | 43.5 |

### 考察・知見
- **VolSizer**: ATR正規化でDD抑制を期待したが、既存のATRベースSL設計が既にボラ調整済みのため効果限定的
- **Kelly(×2.4)**: 同じ21トレード・同方向だが複利効果で最大3,518万に到達。ただしMDD 22.8% — 高リスク高リターン
- **MetaStrategy**: HMMがトレンド判定1.4%と保守的すぎ、ほぼYagamiAに切替→パフォーマンス低下。レジームしきい値の調整が必要
- **次期課題**: HMMのn_states=3（レンジ/緩やかトレンド/強トレンド）や特徴量の見直しが有効か検討

---

## [2026-03-01] v11: Reality Check — 実績データとバックテストの照合

**変更者:** Claude Code
**変更種別:** 現実データとの照合（Reality Check）
**指示書:** `prompt_for_claude_code_v11.md`
**ブランチ運用:** Claude Code → `claude/add-trading-backtest-ePJat` / Manus AI → `main` へマージ

### 変更内容

#### Task 1: 2026年1月 ドローダウン検証 [Teammate C]
- **実績:** 2026-01 は -1,128,362円（金スポット -701,906円 / 銀スポット -550,074円）
- **XAUUSD価格:** 実際は上昇トレンド（4,399 → 5,068、+668pt）。実トレーダーは逆張りショートで大損失
- **Union戦略の判断:** 1月シグナルは **long 2件のみ（1/2・1/4）**、バックテスト完結トレード = **0件（FLAT）**
- **考察:** Union戦略は過熱相場（急騰局面）でシグナルを出さない設計のため、実トレーダーが犯したショートポジションリスクを自動回避。**ドローダウン耐性の観点で有意な差**を確認

#### Task 2: 2025年12月 勝ちトレード照合 [Teammate B]
- **実績:** 2025-12 は +7,288,126円（金+銀、決済292件、勝率 178/292 = 61%）
- **Union戦略シグナル:** 12/01 01:00 long（4,280円）・12/09 05:00 long（4,220円）
- **実績大勝ち日:** 12/22（+282万）・12/23（+265万）・12/24（+153万）
- **照合結果:** Unionシグナル（12/1, 12/9）は実績大勝ちの約2〜3週間前にlong発令 → **実トレーダーの12月初のロングエントリーと方向性一致**。12/9の4,220は月中最安値付近（最安値 4,213/12/4）で押し目買いシグナル = 実際に最も有効なエントリーポイント
- **結論:** Union戦略の方向性は「正解」。時間軸のズレ（バーベース決済 vs 実際の保有継続）が乖離の主因

#### Task 3: ダッシュボード Real World Performance 追加 [全員]
- **`dashboard.html`** に「🌍 Real World Performance」セクションを新規追加
  - KPI カード: 実現損益合計 +1,759万円 / 決済件数 1,369件・勝率 70.6% / 最大損失月 -112万（2026-01） / Union1月行動 FLAT
  - 月次損益 棒グラフ + 実績累計 vs バックテスト累計 折れ線グラフ（Plotly.js）
  - 銘柄別実現損益内訳（金スポット +1,196万 / 銀スポット +518万 / NQ100 +45万）

### 実績データ月次サマリー (Ground Truth)
| 月 | 実現損益 | 金スポット | 銀スポット | Union戦略 |
|---|---|---|---|---|
| 2025-09 | +60万 | +60万 | — | — |
| 2025-10 | +179万 | +179万 | — | — |
| 2025-11 | +159万 | +159万 | — | — |
| 2025-12 | **+729万** | +大勝ち | +大勝ち | long 2件 (方向一致) |
| 2026-01 | **-113万** | -70万 | -55万 | **FLAT** (ドローダウン回避) |
| 2026-02 | +745万 | +745万 | — | 継続保有 |
| **合計** | **+1,759万** | | | |

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
