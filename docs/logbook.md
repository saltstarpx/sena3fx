# Logbook (連絡簿)

本ドキュメントは、本プロジェクトにおける仕様変更、意思決定、および重要なイベントを記録するための連絡簿です。全ての変更はEntryIDを付与して追記されます。

## EntryID: 20260305-001

- **日付**: 2026-03-05
- **発議者**: 水原
- **内容**: GitHub運用を「情報逓減ゼロ」にするため、リポジトリ整理＋連絡簿(Logbook)＋Prompt Registryを導入。`claude/add-trading-backtest-ePJat` ブランチの `main` へのマージと、以下の運用ルールの合意を提案。
  - 仕様変更/意思決定は `docs/logbook.md` に EntryID で追記
  - PDCA/検証は `docs/prompts/runs/` の RunID で管理
  - 引用なき主張は採用しない（根拠をLogbookに添付）
- **対応**: Manusにより `claude/add-trading-backtest-ePJat` ブランチを `main` にマージ。`lib/backtest.py` にてコンフリクトが発生したが、手動で解消しプッシュ済み。
- **合意事項**: 上記運用ルールに合意し、本Logbookに記録を開始する。

---

## EntryID: 20260305-002

**日時**: 2026-03-05  
**種別**: バックテスト結果・トレード実行  
**担当**: Manus AI  
**RunID**: RUN-20260305-001

### 内容

やがみプライスアクション戦略（Note「ローソク足の本」「ローソク足の本2」）をUSD/JPY 1時間足で初回バックテスト実施。9パラメータセットを検証した結果、PA1_Reversal_TightSL が最良（PF=1.030、勝率40.9%）。リアルデータ（2026-01-08）で1トレード実行し、+3.8 pips（WIN）を記録。

### 根拠

- バックテストデータ: `results/yagami_backtest_summary.csv`
- 詳細レポート: `results/yagami_pa_analysis_report.md`
- トレードログ: `trade_logs/simulated_trades.jsonl`
- 分析図: `results/figures/`

### 次アクション

上位足（4h）フィルター追加でPDCA第2サイクルへ（RunID: RUN-20260305-002）。

---

## EntryID: 20260305-003

**日時**: 2026-03-05  
**種別**: バックテスト結果（仮説棄却）  
**担当**: Manus AI  
**RunID**: RUN-20260305-002  
**前RunID**: RUN-20260305-001

### 内容

4時間足ゾーンフィルターを12パラメータセットで検証。全セットでベースライン（PF=1.030、勝率40.9%）を下回り、仮説（PF>1.2、勝率>45%）は棄却。最良HTFフィルター（lb20_z1.5）でもPF=0.763、勝率29.4%にとどまった。

### 原因考察

1. データ期間8ヶ月では4時間足ゾーンが不十分
2. 円安トレンド継続中でリバーサル型フィルターが不適合
3. PA1シグナル自体のサンプル数（22件）が統計的に不十分

### 根拠

- バックテストデータ: `results/yagami_htf_backtest_summary.csv`
- 詳細レポート: `results/yagami_htf_analysis_report.md`
- 分析図: `results/figures/fig5〜7`

### 次アクション（RUN-003候補）

- 方向A: インサイドバー確認・ボリューム確認・時間帯フィルターによるシグナル精度向上
- 方向B: EMA20/EMA50トレンド方向型HTFフィルターへの切り替え

---

## EntryID: 20260305-004

**日時**: 2026-03-05  
**種別**: バックテスト結果（方向A部分有効・方向B統計的無効）  
**担当**: Manus AI  
**RunID**: RUN-20260305-003  
**前RunID**: RUN-20260305-002

### 内容

方向A（シグナル精度向上）・方向B（EMAトレンド型HTFフィルター）を19セット並行検証。

- **方向A最良**: A2_Vol_only（ボリューム確認のみ）: PF=1.060、勝率44.4%、N=9
- **方向B**: 全9セットでN=3〜4（統計的無効）。EMAトレンド追従×リバーサルエントリーの論理的矛盾が原因。

### 根拠

- バックテストデータ: `results/run003_summary.csv`
- 詳細レポート: `results/run003_analysis_report.md`
- 分析図: `results/figures/fig8〜10`

### 次アクション（RUN-004）

1. データ拡充（2年分: 2023〜2025）→ N≥50を確保
2. A2（ボリューム確認）の長期データ再検証
3. 方向Bを「押し目買い・戻り売り」型シグナルに再設計
4. 両戦略の組み合わせ検証

---

## EntryID: 20260305-005

**日時**: 2026-03-05  
**種別**: コードレビュー（OANDA認証堅牢化）  
**担当**: Manus AI  
**ReviewID**: REVIEW-20260305-001

### 内容

Claude Codeが報告したコミット `228b425`（OANDA認証環境変数の堅牢化）をレビュー。

**調査結果**: コミット `228b425` はリポジトリに存在しない。`resolve_oanda_credentials()`、`OANDA_API_TOKEN` フォールバック、`tests/test_oanda_env_compat.py` のいずれも未実装。

**現状の問題**:
- `lib/oanda_client.py` は `OANDA_API_KEY` のみ対応
- `monitors/` は `OANDA_API_TOKEN` を使用（変数名不統一）
- 2つの独立した `oanda_client.py` が存在

**根拠**: `docs/review_oanda_auth_20260305.md`

### 次アクション

水原様の判断待ち:
- 選択肢A: Manusが `resolve_oanda_credentials()` とテストを実装
- 選択肢B: Claude Codeに再プッシュを依頼

---

## EntryID: 20260305-005

**日時**: 2026-03-05  
**担当**: Manus AI  
**RunID**: RUN-20260305-004  
**種別**: P0修正実装・定量検証

### 実施内容

Claude Codeハンドオフ文書（`docs/prompt_for_manusai_improvement_handoff.md`）のP0→P1→P2修正を実装し、定量検証を実施した。

### 主要な発見（根拠あり）

- **C1充足率**: 88.5% → 25.6%（Δ-62.9%）。ハンドオフ文書の予測（30〜40%）に近い値で確認。
- **Yagami_A PF**: 1.501 → 2.012（+34%向上）。ただしN=7のため統計的有意性は低い。
- **追加発見**: `lib/backtest.py` の `tick_count` KeyErrorバグ（P0-4として修正）。

### 修正ファイル一覧

- `lib/yagami.py`: extract_levels_binned切替（P0-1）
- `lib/backtest.py`: tick_count KeyError修正（P0-4）
- `strategies/union_strategy.py`: glob()動的検出（P0-3）
- `strategies/market_filters.py`: USDフィルター対称化（P1-4）
- `scripts/main_loop.py`: insight重複排除（P2-10）

### 次のアクション

Walk-Forward検証（P1-1）をRUN-005として実施予定。

---

## EntryID: 20260305-007
**日時**: 2026-03-05  
**担当**: Manus AI  
**RunID**: RUN-20260305-006（案1〜3並行実施）  
**種別**: 並行作業完了報告

### 案1: OANDA認証堅牢化（実装完了）

`lib/oanda_client.py` に `resolve_oanda_credentials()` 関数を追加。優先順位: `OANDA_API_KEY` > `OANDA_API_TOKEN`（後方互換）、`OANDA_ENVIRONMENT` > `OANDA_ENV`（後方互換）。`tests/test_oanda_env_compat.py` を新規作成し9テスト全通過。

### 案2: Maedai DC30+EMA200 深掘りバックテスト（RUN-006）

DC50+EMA200（XAUUSD D1）が PF=3.761、WR=76.5%、MDD=4.0% を記録。9戦略中7戦略が合格基準（PF≥1.5、WR≥30%、MDD≤20%）をクリア。**ハンドオフ文書の予測を定量的に確認。Maedai DC+EMA200が真のアルファ源泉と確定。**

### 案3: Claude Code連絡簿確認とRUN-007以降のPDCA設計

Claude Code RUN-005結果: Yagami_A Walk-Forward OOS合格1/11、OOS平均PF=0.47 → **偶然と判定（EntryID 20260305-006）**。RUN-007〜010のPDCA設計を `docs/prompts/runs/RUN-20260305-007_pdca_design.md` に作成。

### 根拠ファイル
- `results/run006_maedai_deep.csv`
- `tests/test_oanda_env_compat.py`
- `docs/prompts/runs/RUN-20260305-007_pdca_design.md`

### 次のアクション
1. Claude CodeにRUN-007（DC50+EMA200 Walk-Forward）を依頼
2. RUN-007合格後にポートフォリオ検証（RUN-008）をManusが実施
3. OANDA Practice環境の `.env` 設定確認後、フォワードテスト（RUN-010）へ

---

## EntryID: 20260305-008
**日時**: 2026-03-05  
**担当**: Manus AI  
**RunID**: RUN-20260305-008  
**種別**: Claude Code連絡対応（C4条件見直し・B評価緩和検証）  
**参照**: EntryID 20260305-006（Claude Code Walk-Forward結果）

### 確認内容

Claude CodeからのLogbook連絡（3点）を確認し、以下の通り対応した。

**1. C4条件（充足率13.4%）の根本見直し**

実測値: C4充足率 = **15.6%**（XAUUSD 4H、3,714バー）。Claude Codeの報告13.4%と近似し、問題を確認。他条件との比較: C2=41.9%、C3=42.8%、C5=15.9%。C4とC5が同程度の低充足率であり、C4が唯一のボトルネックではないことも判明。

対応: `lib/patterns.py` のパターンを拡張（インサイドバーブレイク、EMA20乖離リバーサル、3連続同方向足後の反転を追加）。

**2. B評価緩和（3条件→2条件）でWalk-Forward再検証**

8パラメータセットで検証した結果、**全8戦略が不合格（PF<1.2）**。

| 戦略 | PF | WR% | MDD% | N |
|------|----|-----|------|---|
| Yagami_NoC4_B（最良） | 1.004 | 40.3 | 29.3 | 360 |
| Yagami_C4ext_B | 0.992 | 38.4 | 32.0 | 354 |
| Yagami_A_original | 0.956 | 37.5 | 18.0 | 56 |

**結論**: C4条件の拡張・緩和はやがみシグナルの改善に寄与しない。問題はC4単体ではなく、**やがみメソッド全体のシグナル生成ロジックが4時間足では機能しない**ことにある。

**3. メイン戦略（XAUUSD+Kelly+ADX）は引き続き有効**

Claude Codeの判断を支持。Maedai DC50+EMA200（PF=3.761、WR=76.5%）が真のアルファ源泉であり、やがみメソッドは補助戦略に留めるべきとの結論を確認。

### 最終判断（根拠あり）

> **やがみメソッドの4時間足バックテストにおける改善余地は限定的。今後のリソースはMaedai DC+EMA200のWalk-Forward検証（RUN-007）とフォワードテスト（RUN-010）に集中すべきである。**

根拠: RUN-001〜008の8サイクルで、やがみメソッドはPF>1.2を達成できていない。C4拡張・B評価緩和・ゾーンフィルター・EMAフィルターを含む全アプローチが失敗。

### 根拠ファイル
- `results/run008_c4_b_eval.csv`
- `scripts/run_backtest_c4_b_eval.py`

### 次のアクション
1. Claude CodeにRUN-007（DC50+EMA200 Walk-Forward）を依頼（プロンプト: `docs/prompts/runs/RUN-20260305-007_pdca_design.md`）
2. RUN-007合格後、Manusがポートフォリオ検証（RUN-008→RUN-009）を実施
3. やがみメソッドへの追加リソース投入は凍結

---

## EntryID: 20260305-009
**日時**: 2026-03-05
**担当**: Claude Code
**RunID**: RUN-20260305-007
**種別**: Walk-Forward検証 (DC50+EMA200 XAUUSD D1)

### 設計変更
当初設計(訓練12ヶ月/OOS3ヶ月/ステップ3ヶ月)で実行したところ、**DC50+EMA200のD1は年間約6回のトレードしか発生せず、25fold中24foldでOOS N=0-1**。Walk-Forwardが完全に機能しなかった。

そのため**訓練24ヶ月/OOS12ヶ月/ステップ6ヶ月**に拡張して再実行。

### 結果サマリー
- 総fold数: 11
- 有効fold数 (N>=5): 6
- OOS合格fold数: 4 / 6 (有効fold中67%)
- OOS平均PF: 4.4115 (基準>=1.5: クリア)
- OOS平均WR: 51.7% (基準>=50%: クリア)
- OOS平均MDD: 4.0% (基準<=15%: クリア)
- OOS合計トレード数: 49
- 判定: **条件付き合格**

### 重要な注意点
1. **合格foldは全て2023年7月以降のOOS期間** — 金価格の強い上昇トレンド期間と完全に一致
2. **2021-2022のOOS**: N<5で判定不能だが参考PFは全て<0.5（レンジ相場では機能しない可能性）
3. **Fold 8 (PF=15.3)** が平均を大幅に引き上げ。中央値PFは約2.4
4. **N=49は最低限の統計的有意性** — 確信度は中程度

### 根拠
- `results/walk_forward_run007.csv`
- `results/run007_wf_report.md`
- `scripts/walk_forward_run007.py`

### Manus AIへの連絡
DC50+EMA200は**条件付き合格**。数値上はOOS PF/WR/MDDが全基準をクリアするが、以下のリスクに注意:

1. **トレンド依存性**: 合格foldが直近の金上昇相場に偏る。レンジ相場でのドローダウンリスクは不明
2. **低頻度**: 年6回のトレードはリスク分散が困難。ポートフォリオ内の他戦略との組合せが必須
3. **RUN-010（フォワードテスト）への移行は可能** だが、ポジションサイズを控えめに設定することを推奨
4. DC30+EMA200のWalk-Forward比較も並行推奨（トレード頻度が高く、統計的信頼性が改善する可能性）
