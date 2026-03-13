# Claude Code への依頼プロンプト
## RunID: RUN-20260305-005 — Walk-Forward検証

---

## 背景・経緯（必読）

リポジトリ: `saltstarpx/sena3fx`（`main` ブランチ）

直前のRUN-20260305-004にて、以下が判明している。

- **P0-1修正済み**: `lib/yagami.py` の C1フィルターを `extract_levels_binned()` に切替済み
- **Yagami_A（A評価のみ）の PF が 1.501 → 2.012 に向上**したが、**N=7 のため統計的有意性が低い**
- `lib/backtest.py` の `tick_count` KeyError バグも修正済み（`'tick_count' in bar.index` の存在確認を追加）

**今回の目的**: N≥50 を確保した Walk-Forward 検証により、Yagami_A PF=2.012 が実力か偶然かを判定する。

---

## 依頼内容

### タスク1: Walk-Forward検証の実装・実行

**対象戦略**: `sig_yagami_A('4h')` のみ（A評価のみ、4時間足）

**データ**: `data/ohlc/XAUUSD_4h.csv`（全期間データ、約3,000本以上）

**Walk-Forward設定**:
- 訓練期間（In-Sample）: 6ヶ月（約1,095本）
- 検証期間（Out-of-Sample）: 2ヶ月（約365本）
- ステップ幅: 2ヶ月（検証期間と同じ）
- ウィンドウ方式: **拡張型**（訓練期間を毎回拡張していく）

**合格基準**（Out-of-Sample の平均値で判定）:
- PF ≥ 1.2
- 勝率 ≥ 40%
- 最大DD ≤ 20%
- N（トレード数）≥ 5 per fold

### タスク2: 結果の保存

以下のファイルを生成・保存すること。

```
results/walk_forward_run005.csv   # fold別の詳細結果
results/run005_wf_report.md       # 分析レポート（Markdown）
```

`walk_forward_run005.csv` のカラム構成:
```
fold, train_start, train_end, oos_start, oos_end,
pf_is, wr_is, n_is,
pf_oos, wr_oos, n_oos, mdd_oos, passed
```

### タスク3: Logbook への記録

`docs/logbook.md` に以下の形式で追記すること。

```markdown
## EntryID: 20260305-006

**日時**: [実行日]
**担当**: Claude Code
**RunID**: RUN-20260305-005
**種別**: Walk-Forward検証

### 結果サマリー
- 総fold数: [N]
- OOS合格fold数: [N] / [総fold数]
- OOS平均PF: [値]
- OOS平均WR: [値]%
- 判定: [合格 / 要改善]

### 根拠
- `results/walk_forward_run005.csv`
- `results/run005_wf_report.md`
```

### タスク4: コミット・プッシュ

作業完了後、以下のコマンドで `main` ブランチにプッシュすること。

```bash
git add results/walk_forward_run005.csv results/run005_wf_report.md docs/logbook.md
git commit -m "feat(RUN-005): Walk-Forward検証実装・実行 [Yagami_A 4H]

- 訓練6ヶ月/検証2ヶ月のウォークフォワード検証
- OOS平均PF: [値], 合格fold: [N]/[総N]
- Logbook: EntryID 20260305-006"
git push origin main
```

---

## 注意事項

1. **`BacktestEngine` の使い方**: `engine.run(data=df, signal_func=sig_func, freq='4h')` で呼び出す。`tick_count` カラムは不要（修正済み）。
2. **データ読み込み**: `data/ohlc/XAUUSD_4h.csv` を使用。タイムゾーン警告は無視してよい（`utc=True` を指定すれば消える）。
3. **統計的有意性の注意**: fold あたり N<5 の場合は `passed=False` として除外すること。
4. **コンフリクト防止**: 作業は新規ブランチ `claude/run005-walk-forward` で行い、完了後に PR を作成すること。Manus が確認後に main にマージする。

---

## 参照ファイル

| ファイル | 内容 |
|---------|------|
| `lib/backtest.py` | BacktestEngineの実装 |
| `lib/yagami.py` | sig_yagami_A等のシグナル関数 |
| `lib/levels.py` | extract_levels_binned（P0-1修正済み） |
| `results/run004_p0_c1_fix.csv` | 直前RUN-004の結果（比較用） |
| `docs/logbook.md` | 連絡簿（EntryID 20260305-005まで記録済み） |

---

*このプロンプトは Manus AI が作成。RunID管理ルール（`docs/prompts/relay_protocol.md`）に準拠。*
