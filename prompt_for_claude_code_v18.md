<<<<<<< HEAD
## Claude Code指示書v18：最終戦略の確立とフォワードテスト準備

### 背景

v17で**極めて重要な発見**がありました。XAGUSDでは逆効果だったADXフィルター(>25)が、XAUUSDではSharpeを+31%、Calmarを+75%向上させ、MDDを16.1%まで半減させるという驚異的な効果を発揮しました。リバモア式ピラミッティングは全滅したため、これを完全に棄却します。

v18では、この発見に基づき**`XAUUSD+Kelly(f=0.25)+ADX(>25)`**を最終的なメイン戦略として確立し、フォワードテスト（本番監視）の準備を整えます。

### Task 1: 最終戦略 `main_strategy.py` の作成

**目的:** 最終的に確定した戦略ロジックを、単一の実行可能なファイルにまとめる。

1.  `strategies/`ディレクトリに`main_strategy.py`を新設します。
2.  このスクリプトは、以下の確定ロジックを実装します。
    - **銘柄:** `XAUUSD`
    - **時間足:** 4時間足
    - **戦略:** `Union_4H_Base` (DCブレイク or RSI反発 + EMA200フィルター)
    - **フィルター:** `ADX(14) > 25`
    - **サイジング:** `KellyCriterionSizer(f=0.25)`
3.  スクリプトはコマンドラインから実行可能とし、バックテスト結果（Sharpe, Calmar, MDD, 最終資産など）を標準出力するようにしてください。これは将来的な本番環境での動作確認を容易にするためです。

### Task 2: フォワードテスト監視スクリプトの更新

**目的:** リアルタイム監視システムを、v18で確定した最終戦略に対応させる。

1.  `monitors/monitor_union_kelly.py`を`monitors/forward_main_strategy.py`にリネームします。
2.  新しい監視スクリプトは、Task 1で作成した`strategies/main_strategy.py`のロジックを呼び出すように変更します。
3.  OANDA APIから`XAUUSD`の4H足データを取得し、`XAUUSD+Kelly+ADX`のシグナルが発生した際に、`trade_logs/forward_signals.csv`にシグナル内容（日時, 銘柄, 方向, 価格）を記録するようにしてください。

### Task 3: 不要な実験ファイルの削除

**目的:** 役目を終えた実験用スクリプトを削除し、リポジトリをクリーンに保つ。

1.  以下の不要になったスクリプトをリポジトリから削除してください。
    - `scripts/grid_search_pyramid.py`
    - `scripts/backtest_xau_final.py`
    - `scripts/backtest_xau_adx_filter.py`

### Task 4: ダッシュボードの更新

**目的:** 最新の最良戦略の結果をダッシュボードに反映させる。

1.  `dashboard.html`を更新し、`XAUUSD+Kelly(f=0.25)+ADX(>25)`のバックテスト結果（Sharpe 2.250, Calmar 11.534, MDD 16.1%）を最も目立つ場所に「**Main Strategy**」として表示してください。
2.  過去の戦略（Union Base, MetaStrategyなど）は比較対象として残しつつも、新しいメイン戦略が最高性能であることが一目でわかるようにレイアウトを調整してください。

### 完了報告

全タスク完了後、変更した全ファイルを`claude/add-trading-backtest-ePJat`ブランチにプッシュし、完了報告を行ってください。特に、`main_strategy.py`が単体で動作すること、フォワードテスト監視が新ロジックに対応したことを報告してください。
=======
# Claude Code v18 指示書: メイン戦略確定・フォワードモニター刷新

## ブランチ運用ルール（必須）
- **Claude Code の作業ブランチ**: `claude/add-trading-backtest-ePJat`
- **Manus AI のブランチ**: `main`

---

## プロジェクト背景

v17でXAUUSD+Kelly(f=0.25)+ADX(14)>25がSharpe=2.250, Calmar=11.534, MDD=16.1%という
圧倒的な結果を達成。v18ではこれをメイン戦略として確定し、
本番運用体制（フォワードモニター）を整備する。

---

## v18 タスク一覧

### Task 1: 最終戦略 `strategies/main_strategy.py` の作成 ✅
- 銘柄: XAUUSD / 時間足: 4時間足
- 戦略: Union_4H_Base + ADX(14)>25 + KellyCriterionSizer(f=0.25)
- `make_signal_func()`, `run_backtest()`, `get_latest_signal()` を提供
- CLIから直接実行可能 (`python strategies/main_strategy.py`)

### Task 2: フォワードモニター刷新 ✅
- `monitors/monitor_union_kelly.py` を `monitors/forward_main_strategy.py` に置き換え
- `trade_logs/forward_signals.csv` にシグナルを記録 (datetime_utc, symbol, direction, price, adx)
- main_strategy.py の `make_signal_func()` を使用

### Task 3: 実験スクリプト削除 ✅
- `scripts/grid_search_pyramid.py` — 削除 (v17完了)
- `scripts/backtest_xau_final.py` — 削除 (v16完了)
- `scripts/backtest_xau_adx_filter.py` — 削除 (本番昇格済)

### Task 4: ダッシュボード更新 ✅
- `XAUUSD+Kelly+ADX(>25)` を最上部に最優秀戦略として表示
- KPIカード: Sharpe=2.250, Calmar=11.534, MDD=16.1%, WR=61.0%, ¥27,175,560
- 4戦略比較チャート (Union_Base / +Kelly / +Pyramid / +ADX)

---

## 実行結果 (達成済み)

### メイン戦略 確定パラメータ

| 項目 | 値 |
|---|---|
| 銘柄 | XAUUSD |
| 時間足 | 4H |
| シグナル | Union_4H_Base (sig_maedai_yagami_union) |
| フィルター | ADX(14) > 25 |
| サイジング | KellyCriterionSizer(f=0.25) |
| Kelly乗数 | 1.13x |

### 確定バックテスト結果 (2023-10-06 〜 2026-02-27)

| 指標 | 値 | 判定 |
|---|---|---|
| Sharpe Ratio | 2.250 | ✅ |
| Calmar Ratio | 11.534 | ✅ |
| Max Drawdown | 16.1% | ✅ |
| Win Rate | 61.0% | ✅ |
| Trades | 41 | ✅ |
| 最終資産 | ¥27,175,560 (+443.5%) | ✅ |

### 戦略進化サマリー (XAUUSD 4H)

| 戦略 | Sharpe | Calmar | MDD% | 採否 |
|---|---|---|---|---|
| Union_XAUUSD_Base | 1.718 | 5.971 | 23.5 | 参考 |
| +Kelly(f=0.25) | 1.717 | 6.574 | 26.3 | 採用 (v16) |
| +Kelly+Pyramid(LV) | 0.030 | -0.178 | 42.9 | ❌ 不採用 |
| +Kelly+ADX(>25) | **2.250** | **11.534** | **16.1** | ✅ **メイン戦略** |
>>>>>>> origin/claude/add-trading-backtest-ePJat
