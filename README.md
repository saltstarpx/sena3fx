# sena3fx — 戦略ポートフォリオ PDCAエージェント v4.0

XAUUSD（金スポット）のバックテスト自動化 + やがみメソッド準拠の戦略ポートフォリオ。

---

## 戦略ポートフォリオ

| チーム | 戦略 | 評価基準 | 最新実績 |
|--------|------|----------|---------|
| **A: Yagami** | 高勝率シグナル | WR>60%, PF>1.8 | `strategies/yagami_rules.py` |
| **B: Maedai** | DCブレイクアウト | Sharpe>1.5, Calmar>3.0 | `strategies/maedai_breakout.py` |
| **B: Union** | Maedai×Yagami OR複合 | Sharpe>1.5 | **Sharpe 2.817** / PF 3.624 |
| **C: Risk Manager** | 市場環境フィルター | MDD低減>20% | `strategies/market_filters.py` |

---

## ディレクトリ構成

```
sena3fx/
├── CLAUDE.md                   # AI開発ガイド（Claude Code用）
├── MANUS_CHANGELOG.md          # AI変更履歴（Claude/Manus連携用）
├── README.md                   # このファイル
├── dashboard.html              # 戦略パフォーマンスダッシュボード
├── knowledge.json              # PDCA蓄積知見
│
├── lib/                        # コアエンジン
│   ├── yagami.py               #   シグナル関数群（全戦略）
│   ├── backtest.py             #   バックテストエンジン v4.1
│   ├── candle.py               #   ローソク足パターン
│   ├── patterns.py             #   チャートパターン
│   ├── levels.py               #   レジサポ検出
│   ├── timing.py               #   セッション・タイミング
│   ├── indicators.py           #   テクニカル指標
│   └── ...
│
├── strategies/                 # 戦略ポートフォリオ
│   ├── union_strategy.py       #   Union戦略（★エース: Sharpe 2.817）
│   ├── yagami_rules.py         #   Teammate A
│   ├── maedai_breakout.py      #   Teammate B
│   └── market_filters.py      #   Teammate C（USD強弱・季節フィルター）
│
├── monitors/                   # フォワードテスト監視
│   └── forward_union.py        #   Union戦略シグナル監視（1時間ごと）
│
├── scripts/                    # 実行スクリプト
│   ├── main_loop.py            #   PDCAサイクル自動実行
│   ├── backtest_portfolio.py   #   全戦略一括バックテスト
│   └── fetch_data.py           #   OHLCデータ取得
│
├── docs/                       # ドキュメント + やがみPDF参考資料
│   ├── strategy_union.md       #   Union戦略ドキュメント
│   ├── strategy_yagami.md      #   Yagami戦略ドキュメント
│   ├── strategy_maedai.md      #   Maedai戦略ドキュメント
│   ├── filters_risk_manager.md #   フィルター設定ドキュメント
│   └── *.pdf                   #   やがみメソッドPDF参考資料
│
├── data/ohlc/                  # OHLCデータ（Yahoo Finance GC=F由来）
│   ├── XAUUSD_1h.csv           #   金 1H足（2023-2026, ~13,000本）
│   ├── XAUUSD_4h.csv           #   金 4H足（2023-2026, ~3,700本）
│   ├── XAUUSD_1d.csv           #   金 日足（2019-2026, ~1,800本）
│   ├── XAUUSD_8h.csv           #   金 8H足
│   ├── XAUUSD_2025_*.csv       #   金 2025年限定サブセット（各時間軸）
│   └── XAGUSD_2025_*.csv       #   銀 2025年限定サブセット
│
├── results/                    # バックテスト結果
│   ├── performance_log.csv     #   ★全戦略パフォーマンスログ（自動追記）
│   ├── backtest_maedai_ranking.csv
│   └── *.csv / *.png           #   個別バックテスト結果
│
├── trade_logs/                 # トレード記録
│   ├── broker_history_part1_20260226.csv  # 実トレード履歴 Part1（金・銀CFD）
│   ├── broker_history_part2_20260227.csv  # 実トレード履歴 Part2
│   └── forward_union_signals.csv          # Unionフォワードシグナル記録
│
├── reports/                    # PDCAレポート
├── live/                       # ライブトレードBot（OANDA）
│   ├── bot_v2.py               #   本番Bot（最新版）
│   └── ...
├── backtest/                   # vectorbt補助エンジン
└── archive/                    # v1/v2旧コード（参照用）
```

---

## クイックスタート

```bash
pip install numpy pandas

# Union戦略バックテスト（Sharpe 2.817 確認）
python strategies/union_strategy.py

# Union戦略フォワードテスト監視
python monitors/forward_union.py

# 全戦略PDCAサイクル
python scripts/main_loop.py

# ダッシュボード表示
python -m http.server 8080  # → http://localhost:8080/dashboard.html
```

---

## データについて

`data/ohlc/` のCSVはすべて GitHub で管理されており、Manus AI / Claude Code 双方が参照可能です。

| ファイル | 期間 | 足数 | 用途 |
|--------|------|------|------|
| `XAUUSD_1h.csv` | 2023-2026 | ~13,700 | 主力バックテスト |
| `XAUUSD_4h.csv` | 2023-2026 | ~3,700 | 主力バックテスト |
| `XAUUSD_1d.csv` | 2019-2026 | ~1,800 | 長期トレンド分析 |
| `XAUUSD_2025_*.csv` | 2025 | 各種 | 直近パラメータ検証 |

`results/performance_log.csv` はバックテスト実行ごとに自動追記されます。

---

## Manus AI ↔ Claude Code 連携

| ブランチ | 担当 | 役割 |
|---------|------|------|
| `main` | Manus AI | データ追加・指示書更新 |
| `claude/add-trading-backtest-*` | Claude Code | コード開発・バックテスト |

**変更履歴:** `MANUS_CHANGELOG.md` を参照
