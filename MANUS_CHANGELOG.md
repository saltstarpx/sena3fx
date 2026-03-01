# Manus AI 変更履歴

このファイルは、Manus AIがリポジトリに対して行った変更を記録するログです。
Claude Codeとの連携において、何が変更されたかを把握するために使用してください。

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

## [2026-02-28] 通貨強弱分析・XAGクロスペアデータ追加・Claude Code指示書v7

**変更者:** Manus AI
**変更種別:** データ追加 + 分析スクリプト追加 + 指示書追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `data/ohlc/XAGJPY_1d.csv` | 合成XAGJPYデータ (1,257バー) |
| `data/ohlc/XAGJPY_1h.csv` | 合成XAGJPY 1時間足 (13,560バー) |
| `data/ohlc/XAGCHF_1d.csv` | 合成XAGCHFデータ |
| `data/ohlc/XAGCHF_1h.csv` | 合成XAGCHF 1時間足 |
| `data/ohlc/XAGEUR_1d.csv` | 合成XAGEURデータ |
| `data/ohlc/XAGEUR_1h.csv` | 合成XAGEUR 1時間足 |
| `data/ohlc/XAGGBP_1d.csv` | 合成XAGGBPデータ |
| `data/ohlc/XAGGBP_1h.csv` | 合成XAGGBP 1時間足 |
| `data/ohlc/XAGAUD_1d.csv` | 合成XAGAUDデータ |
| `data/ohlc/XAGAUD_1h.csv` | 合成XAGAUD 1時間足 |
| `data/ohlc/XAGNZD_1d.csv` | 合成XAGNZDデータ |
| `data/ohlc/XAGNZD_1h.csv` | 合成XAGNZD 1時間足 |
| `data/ohlc/XAGCAD_1d.csv` | 合成XAGCADデータ |
| `data/ohlc/XAGCAD_1h.csv` | 合成XAGCAD 1時間足 |
| `data/ohlc/USDJPY_1d.csv` | FXペアデータ |
| `data/ohlc/USDJPY_1h.csv` | FXペアデータ |
| `data/ohlc/EURUSD_1d.csv` | FXペアデータ |
| `data/ohlc/EURUSD_1h.csv` | FXペアデータ |
| `data/ohlc/GBPUSD_1d.csv` | FXペアデータ |
| `data/ohlc/GBPUSD_1h.csv` | FXペアデータ |
| `data/ohlc/AUDUSD_1d.csv` | FXペアデータ |
| `data/ohlc/AUDUSD_1h.csv` | FXペアデータ |
| `data/ohlc/NZDUSD_1d.csv` | FXペアデータ |
| `data/ohlc/NZDUSD_1h.csv` | FXペアデータ |
| `data/ohlc/USDCAD_1d.csv` | FXペアデータ |
| `data/ohlc/USDCAD_1h.csv` | FXペアデータ |
| `data/ohlc/USDCHF_1d.csv` | FXペアデータ |
| `data/ohlc/USDCHF_1h.csv` | FXペアデータ |
| `scripts/currency_strength_engine.py` | v1通貨強弱計算エンジン |
| `scripts/currency_strength_v2.py` | v2改良版（ブレンド+加速度） |
| `scripts/currency_strength_v3.py` | v3テクニカル統合版（グリッドサーチ） |
| `scripts/currency_strength_portfolio.py` | ポートフォリオ版（最終版） |
| `results/v3_parameter_sweep.csv` | 96パラメータのグリッドサーチ結果 |
| `results/portfolio_summary.json` | ポートフォリオ分析サマリー |
| `results/dynamic_pair_selection_1d.csv` | v1日足結果 |
| `results/dynamic_pair_selection_1h.csv` | v1時間足結果 |
| `results/dynamic_pair_v2_1d.csv` | v2日足結果 |
| `results/v3_best_result.csv` | v3最適パラメータ結果 |
| `results/portfolio_backtest_1d.csv` | ポートフォリオバックテスト結果 |
| `prompt_for_claude_code_v7.md` | Claude Code指示書v7 |

### 分析結論

**通貨強弱による動的XAGペア切替はXAGUSD固定に勝てない。**

4つのアプローチ（v1〜v3 + ポートフォリオ）で96パラメータ組み合わせを検証した結果、XAGUSD固定を一貫して上回る設定は存在しなかった。

### 有用な発見

1. **XAGCHF**: CHFが弱い時のロングはWR=56.9%、平均+0.24%/日
2. **ポートフォリオ効果**: 最大DD -19.52% vs XAGUSD単独 -36.18%（リスク半減）
3. **USD強弱フィルター**: USDが極端に強い時のロング回避は有効

### Claude Codeへの指示

1. **XAGUSD固定を維持**すること。通貨強弱でペアを切り替えない
2. **USD強弱をフィルターとして統合**: `calc_usd_strength()` を使い、USD上位25%時はロング回避
3. **ポジションサイズ調整**: USD強弱に応じてロットサイズを0.7〜1.0倍に調整
4. 詳細は `prompt_for_claude_code_v7.md` を参照

---

## [2026-03-01] Claude Code指示書v8追加：成功バイアス回避と戦略ポートフォリオ導入

**変更者:** Manus AI
**変更種別:** 指示書追加 + 開発フレームワーク刷新提案

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `prompt_for_claude_code_v8.md` | Claude Code指示書v8（成功バイアス回避・戦略ポートフォリオ） |

### 変更理由

UKI氏の考察記事「Claude Codeで実験がワンパターンになる構造的理由とAgent Teamsでの対処」を分析し、当プロジェクトにも同様の成功バイアスリスク（やがみメソッドへの収斂）が存在することを特定。Agent Teamsの「コンテキスト分離」思想を応用した戦略ポートフォリオアプローチを提案。

### 主な変更内容

1. **3つの仮想チームメイト定義:**
   - Teammate A "Yagami": 高勝率シグナル追求（WR>60%, PF>1.8）
   - Teammate B "Maedai": 高RRトレンドフォロー（Sharpe>1.5, Calmar>3.0）
   - Teammate C "Risk Manager": 市場環境フィルター開発（MDD低減率）

2. **具体的タスク4件:**
   - Task 1: USD強弱フィルターの統合（Manus分析結果の活用）
   - Task 2: 季節フィルターの定量的評価
   - Task 3: Maedai戦略の評価軸変更（Sharpe/Calmar中心）
   - Task 4: ドキュメントとファイル構成の分離

3. **評価軸の多元化:** PF/WR中心からSharpe Ratio, Calmar Ratio, MAR等の複数軸へ

### Claude Codeへの指示

`prompt_for_claude_code_v8.md`を読み、記載されたタスクを順次実行すること。
特に、`CLAUDE.md`の情報密度がやがみメソッドに偏っている問題を認識し、ドキュメント分離を実施すること。

---

## [2026-03-01] Claude Code指示書v9追加：v8成果の深掘りと戦略確立

**変更者:** Manus AI
**変更種別:** 指示書追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `prompt_for_claude_code_v9.md` | Claude Code指示書v9（v8成果の検証と戦略確立） |

### 変更理由

Claude Codeのv8実行結果（pasted_content_6.txt）を分析し、特に「Union戦略（Sharpe 1.859）」と「USD強弱フィルター」の有効性が高いと判断。これらの発見を個別の戦略として確立し、横展開するための具体的なタスクを指示。

### 主なタスク指示

1.  **v8成果物のプッシュ（最優先）:** Claude Codeのローカルにある`strategies/`と`docs/`をリモートにプッシュするよう指示。
2.  **Union戦略の確立:** 正確な定義をドキュメント化し、独立したスクリプトとして再現性を確認する。
3.  **USD強弱フィルターの横展開:** Maedai戦略とYagamiFull_1Hに適用し、効果の普遍性を検証する。
4.  **季節フィルターの最適化:** 戦略ごとにフィルターの適用/非適用を決定する。

### Claude Codeへの指示

`prompt_for_claude_code_v9.md`を読み、記載されたタスクを順次実行すること。
クレジット消費を抑えるため、探索的な分析は行わず、指定されたタスクに集中するよう指示。

---

## [2026-03-02] Claude Code指示書v10更新：戦略パフォーマンス・ダッシュボード構築タスク追加

**変更者:** Manus AI
**変更種別:** 指示書追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `prompt_for_claude_code_v10.md` | Claude Code指示書v10（mainマージ、フォワードテスト準備、戦略ダッシュボード構築） |

### 変更理由

Claude Codeのv9実行結果を分析し、エース戦略候補「Union戦略」の信頼性検証（フォワードテスト）と、判明した知見（USD強弱フィルターの適用範囲）のドキュメント化が必要と判断。

### 主なタスク指示

1.  **mainブランチへのマージ（最優先）:** 開発ブランチ`claude/add-trading-backtest-ePJat`を`main`にマージするよう指示。
2.  **Union戦略のフォワードテスト準備:** シグナル監視スクリプト`monitors/forward_union.py`を作成するよう指示。
3.  **USD強弱フィルターの知見をドキュメント化:** フィルターの適用範囲に関する結論を`docs/filters_risk_manager.md`に記録するよう指示。
4.  **戦略パフォーマンス・ダッシュボード構築:** Dinii社の記事に触発され、バックテスト結果を自動で`performance_log.csv`に記録し、`dashboard.html`で可視化する仕組みの構築を指示。

### Claude Codeへの指示

`prompt_for_claude_code_v10.md`を読み、記載されたタスクを順次実行すること。

---

## [2026-03-02] Claude Code指示書v12追加：「1000万→1億」への3つのエンジン実装

**変更者:** Manus AI
**変更種別:** 指示書追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `prompt_for_claude_code_v12.md` | Claude Code指示書v12（3つの新エンジン実装） |

### 変更理由

水原さんの「俺を超えるbot」という目標達成のため、クオンツ文献調査で得られた3つの理論をプロジェクトに統合する。これにより、DD抑制とリターン最大化、相場環境への適応能力を獲得する。

### 主なタスク指示

1.  **ボラティリティ調整サイジング:** ATRに基づきポジションサイズを動的に調整する`VolatilityAdjustedSizer`を実装。
2.  **フラクショナル・ケリー基準:** バックテスト結果から最適投資比率を計算する`KellyCriterionSizer`を実装。
3.  **レジーム転換モデル:** 隠れマルコフモデル（HMM）を用いて相場環境を判定し、最適な戦略（Yagami/Union）を自動で切り替える`MetaStrategy`を実装。
4.  **ダッシュボード拡張:** 新しいサイジング手法と`MetaStrategy`のパフォーマンスを可視化する。

### Claude Codeへの指示

`prompt_for_claude_code_v12.md`を読み、記載されたタスクを順次実行すること。
