<<<<<<< HEAD
## Claude Code指示書v16：XAUUSDメイン化とリバモア式ピラミッティング

### 背景

v15でXAUUSDがSharpe 1.718, Calmar 5.971, トレード数70件という圧倒的な結果を出しました。v16ではXAUUSDをメイン戦略に昇格させ、リバモア式ピラミッティングを導入して利益の最大化を目指します。

### Task 1: XAUUSD + KellyCriterionSizer バックテスト

**目的:** XAUUSDにKellyサイジングを適用した場合のパフォーマンスを確認する。

1.  `scripts/`に`backtest_xau_kelly.py`を新設。
2.  このスクリプトは、`XAUUSD`に対し、`Union_4H_Base`戦略と`KellyCriterionSizer(f=0.25)`を組み合わせたバックテストを実行する。
3.  結果を`performance_log.csv`に`XAUUSD+Kelly(f=0.25)`という名前で記録する。

### Task 2: リバモア式ピラミッティングの実装

**目的:** 含み益が出ているポジションに機械的に追加投資する`LivermorePyramidingSizer`を実装する。

1.  `lib/risk_manager.py`に`LivermorePyramidingSizer`クラスを新設。
2.  このSizerは、以下のロジックでポジションサイズを決定する。
    - **初期ポジション:** 最初のシグナルでは、接続された他のSizer（例: Kelly）の計算結果をそのまま使う。
    - **追加ポジション:** ポジション保有中に、価格がエントリー価格から`step_pct`（例: 1%）上昇するごとに、追加のポジションを取る。
    - **追加ロット:** 追加ロットは`pyramid_ratios`（例: `[0.5, 0.3, 0.2]`）に従って減少させる。最初の追加が0.5倍、次が0.3倍…となる。
    - **最大追加回数:** `max_pyramids`（例: 3回）で制限する。
3.  `BacktestEngine`を改造し、`on_bar`で`LivermorePyramidingSizer`がポジションを監視し、追加エントリーできるようにする。

### Task 3: XAUUSD + Kelly + ピラミッティング統合バックテスト

**目的:** 全てを統合した最終形態のパフォーマンスを検証する。

1.  `scripts/`に`backtest_xau_final.py`を新設。
2.  このスクリプトは、`XAUUSD`に対し、`Union_4H_Base`戦略に`LivermorePyramidingSizer`を接続し、そのSizerが内部で`KellyCriterionSizer(f=0.25)`を使う構成でバックテストを実行する。
    - `LivermorePyramidingSizer(step_pct=0.01, pyramid_ratios=[0.5, 0.3, 0.2], max_pyramids=3)`
3.  結果を`performance_log.csv`に`XAUUSD+Kelly+Pyramid`という名前で記録する。

### 完了報告

全タスク完了後、`performance_log.csv`への追記、および新設・変更した全ファイルを`claude/add-trading-backtest-ePJat`ブランチにプッシュし、完了報告を行ってください。
=======
# Claude Code v16 指示書: XAUUSDメイン化とリバモア式ピラミッティング

## ブランチ運用ルール（必須）
- **Claude Code の作業ブランチ**: `claude/add-trading-backtest-ePJat`
- **Manus AI のブランチ**: `main`

---

## プロジェクト背景

v15でXAUUSDがSharpe 1.718, Calmar 5.971, Trades 70 と最強の結果。
v16ではXAUUSDをメイン化し、KellyサイジングとリバモアPyramidingを検証する。

---

## v16 タスク一覧

### Task 1: XAUUSD + KellyCriterionSizer(f=0.25) バックテスト
新設: `scripts/backtest_xau_kelly.py`
- Union_4H_Base (ベースライン) vs XAUUSD+Kelly(f=0.25)
- Kelly乗数: WR=50%, PF=1.828 → f*=0.0566 → 乗数 1.13x

### Task 2: LivermorePyramidingSizer 実装
修正: `lib/risk_manager.py` — `LivermorePyramidingSizer` クラス追加
- `base_sizer`: KellyCriterionSizer 等への委譲
- `reset(entry_price, initial_size)`: 新規ポジション時にコール
- `on_bar(direction, current_price)` → 追加ロット数を返す
修正: `lib/backtest.py` — on_bar() フック追加 (v4.2)

### Task 3: XAUUSD + Kelly + Livermore ピラミッティング 統合バックテスト
新設: `scripts/backtest_xau_final.py`
- 3系列比較: Base / Kelly(f=0.25) / Kelly+Pyramid(LV)

---

## 実行結果 (達成済み)

### Task 1: Kelly(f=0.25)
| 戦略 | Sharpe | Calmar | MDD% | 最終資産 |
|---|---|---|---|---|
| Union_XAUUSD_Base | 1.718 | 5.971 | 23.5 | ¥21,795,420 |
| XAUUSD+Kelly(f=0.25) | 1.717 | 6.574 | 26.3 | ¥25,701,780 (+18%) |

### Task 3: Livermore Pyramiding
| 戦略 | Sharpe | Calmar | MDD% | 最終資産 |
|---|---|---|---|---|
| XAUUSD+Kelly+Pyramid(LV) | 0.030 | -0.178 | 42.9 | ¥4,091,874 |
- 判定: 非推奨 (PF=0.932 < 1.0、MDD 42.9% > 30%目標)
- 原因: step_pct=1%が4H金相場では敏感すぎ、ピラミッド後ポジションがSL多発

### 知見
- **Kelly(f=0.25) は推奨**: +18%最終資産改善、Calmar改善
- **Livermore Pyramid (step_pct=1%) は非推奨**: step_pct=2〜3%で再検証余地あり
>>>>>>> origin/claude/add-trading-backtest-ePJat
