# Claude Code 指示書 v11: 「バックテスト vs 現実」— 実績データとの照合

## 0. ブランチ運用ルールの確定（重要）

GitHubのmainブランチとClaude Codeの作業ブランチ（claude/add-trading-backtest-ePJat）の乖離問題を解決しました。

今後の運用ルール：
* Claude Codeは引き続き`claude/add-trading-backtest-ePJat`ブランチで作業・プッシュする
* Manus AIが定期的に`claude/add-trading-backtest-ePJat`を`main`にマージする
* これにより両者が常に同じコードを参照できる状態を維持する

v11の作業もこのルールに従い、全変更を`claude/add-trading-backtest-ePJat`にプッシュしてください。

## 1. リポジトリ整理の完了承認と最重要発見の確認

| 項目 | 結果 |
|:---|:---|
| **CFD決済件数** | 1,369件 |
| **実現損益合計** | **+17,585,449円** |
| **勝率（WR）** | **70.6%** |
| **金スポット損益** | +11,955,646円（770トレード） |
| **銀スポット損益** | +5,179,127円（507トレード） |

## 2. v11のテーマ：「Reality Check（現実との照合）」

v11の目的は、我々が構築したバックテスト戦略（Union, Yagami, Maedai）が、
この輝かしい実績をどの程度説明できるのかを検証することです。

## 3. v11タスク指示

### Task 1: [Teammate C] 2026年1月のドローダウンを分析せよ

背景: 実績データでは、2026年1月に-1,128,362円という大きな損失月がありました。

アクション:
1. `data/ohlc/`にあるXAUUSDとXAGUSDの価格データから、2026年1月分のみを抽出
2. `strategies/union_strategy.py`のバックテストを実行
3. 結果（PF, MDD, WR, 損益）を報告

### Task 2: [Teammate B] 実績トレードと戦略シグナルを照合せよ

背景: 実績を生み出した「トレーダーの思考」を戦略としてコード化するための第一歩。

アクション:
1. 実績データの中で最も利益が大きかった2025年12月の「金スポット」と「銀スポット」の決済トレードを`trade_logs/`から全て抽出
2. `strategies/union_strategy.py`を2025年12月の価格データでバックテストし、シグナルリストを取得
3. 「実績の勝ちトレード日時」と「Union戦略の買いシグナル日時」を比較・分析

### Task 3: [全員] 戦略ダッシュボードに「現実」を追加せよ

アクション:
1. `dashboard.html`に新しいセクション「Real World Performance」を追加
2. `trade_logs/broker_history_*.csv`から月次の実現損益を計算し、折れ線グラフで表示
3. バックテストの資産曲線と現実の資産曲線を同じ画面で比較できるようにする

## 4. GitHubへのログ記録

変更完了後、`MANUS_CHANGELOG.md`に変更内容を記録し、
すべての変更を`claude/add-trading-backtest-ePJat`ブランチにプッシュ。
