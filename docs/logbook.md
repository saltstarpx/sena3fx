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
