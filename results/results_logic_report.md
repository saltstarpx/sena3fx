# Results Logic Report

- total_files: **360**
- extension_breakdown:
  - .csv: 185
  - .png: 134
  - .md: 26
  - .json: 9
  - .txt: 5
  - .jsonl: 1

## Inferred Logic from Results
- 多数の成果物が **v75-v77 系バージョン比較** と **IS/OOS検証** を中心に構成される。
- 評価軸は主に **PF / 勝率 / ドローダウン / 総損益 / 取引数**。
- 取引実行現実性として **spread/slippage/latency** の影響評価ファイルが併存。
- ポートフォリオ化（multi-pair, multi-asset）と単体戦略比較を反復する探索ロジック。

## Text Signal Counts
- pf: 20
- is: 8
- spread: 3
- v77: 3
- oos: 3
- win rate: 3
- drawdown: 2
- v75: 2
- slippage: 1

## Top CSVs by Total PnL
| file | pnl_col | trades | total_pnl | win_rate | pf |
|---|---:|---:|---:|---:|---:|
| results/f1f3_oos_trades.csv | pnl | 2902 | 24084994.52 | 68.0% | 2.281 |
| results/trade_history_maedai_NY_8H_RSI_PB_45.csv | pnl | 27 | 16526639.89 | 51.9% | 5.383 |
| results/trade_history_maedai_8H_RSI_PB_45.csv | pnl | 27 | 16526639.89 | 51.9% | 5.383 |
| results/xauusd_6m_trades.csv | half_pnl | 274 | 14023781.05 | 75.5% | inf |
| results/multi_fast_trades.csv | half_pnl | 415 | 11458799.70 | 70.1% | inf |
| results/f1f3_is_trades.csv | pnl | 1229 | 9747276.13 | 67.6% | 2.259 |
| results/trade_history_maedai_FLT_12H_DC30_C2.csv | pnl | 6 | 4338533.81 | 50.0% | 7.645 |
| results/trade_history_maedai_12H_DC30d_Confirm2.csv | pnl | 6 | 4338533.81 | 50.0% | 7.645 |
| results/metals_indices_oos_trades.csv | pnl | 1786 | 3099969.48 | 54.1% | 1.181 |
| results/trade_history_maedai_FLT_4H_DC15_C2.csv | pnl | 7 | 1851048.35 | 57.1% | 15.873 |

## Code Logic Bug Fix Applied
- `scripts/archive/compare_all_versions.py` の `Position.str.contains()` が NaN で落ちる不具合を修正（`astype(str)` + `na=False`）。
- 同スクリプトの results パスを絶対パス依存からリポジトリ相対パスへ修正。
