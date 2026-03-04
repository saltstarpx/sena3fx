# Claude Code v17 指示書: ピラミッドGS + ADXフィルター + 承認リスト厳格化

## ブランチ: `claude/add-trading-backtest-ePJat`

---

## v17 タスク一覧

### Task 1: ピラミッティング・パラメータ グリッドサーチ
新設: `scripts/grid_search_pyramid.py`
- XAUUSD + Kelly(f=0.25) ベースに LivermorePyramidingSizer を探索
- step_pct: [0.02, 0.03, 0.04] × max_pyramids: [1, 2] = 6パターン
- 結果: `results/pyramid_grid_search.csv`

### Task 2: XAUUSD ADXフィルター検証
新設: `scripts/backtest_xau_adx_filter.py`
- A: XAUUSD+Kelly(f=0.25) (フィルターなし)
- B: XAUUSD+Kelly(f=0.25)+ADX(>25)

### Task 3: approved_list.py 承認条件厳格化
修正: `lib/approved_list.py`
- 旧: `Sharpe > 1.0 AND Trades > 20`
- 新: `Sharpe > 1.5 AND Calmar > 5.0`

---

## 実行結果

### Task 1: ピラミッドGS
全6パターンがベースライン (Sharpe=1.717) を下回る → **Livermore Pyramiding 不採用確定**
- ベスト: step_pct=4%, max_pyramids=1 → Sharpe=1.636 (ベースライン比 -0.081)

### Task 2: ADXフィルター — 🔑 重要発見
| 戦略 | Sharpe | Calmar | MDD% | WR% |
|---|---|---|---|---|
| XAUUSD+Kelly | 1.717 | 6.574 | 26.3 | 50.0 |
| **XAUUSD+Kelly+ADX(>25)** | **2.250** | **11.534** | **16.1** | **61.0** |
- Sharpe +31%, MDD -10.2pt, Calmar +75%, WR +11pt — **圧倒的改善**
- XAUUSDはADX平均27.7でトレンド性高く、ADXフィルターと相性抜群

### Task 3: 承認リスト v2.0
- v2.0: XAUUSD のみ承認 (Sharpe=1.718 ✅, Calmar=5.971 ✅)
- XAGUSD: Calmar=2.111 < 5.0 → 非承認

## 次期推奨 (v18)
`XAUUSD+Kelly(f=0.25)+ADX(>25)` をメイン戦略に昇格
Sharpe=2.250, Calmar=11.534, MDD=16.1%, WR=61.0%
