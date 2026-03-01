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
