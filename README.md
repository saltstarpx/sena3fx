# sena3fx — YAGAMI改 MTF自動取引エージェント

やがみメソッドをベースに定量・計量分析で再設計した **YAGAMI改** 戦略の自動化システム。
4時間足EMA20によるトレンドフィルター + 1時間足の二番底・二番天井パターン検出。

---

## 現在の稼働状況

| 項目 | 内容 |
|------|------|
| **本番環境** | Exness × MT5（Windows VPS） |
| **稼働モード** | ライブトレード（アダプティブリスク 0.5〜2.0%） |
| **採用7銘柄** | USDJPY / GBPUSD / EURUSD / USDCAD / NZDUSD / XAUUSD / AUDUSD |
| **本番ロジック** | ロジック別に銘柄を割り当て（下記参照） |
| **データ取得** | MT5 Python API（リアルタイム） |
| **PDCA** | 毎日/毎週/毎月レビュー自動生成 → Discord通知 |
| **Cloud Run** | OANDA ペーパートレード継続中（asia-northeast1） |

---

## 戦略概要 — YAGAMI改

### コンセプト

やがみメソッドに基づく**二番底・二番天井パターン**の自動検出。
マルチタイムフレーム（4時間足・1時間足・1分足）を組み合わせ、高確率なエントリーポイントを機械的に判定する。

#### 共通ベースロジック（全銘柄）

| フィルター | 内容 |
|------|------|
| **4H EMA20** | クローズ > EMA20 → Long / 下 → Short（トレンド方向） |
| **KMID** | 直前4H足の実体方向がエントリー方向と一致 |
| **KLOW** | 直前4H足の下ヒゲ比率 < 0.15%（`< 0.0015`） |
| **EMA距離** | 4H終値とEMA20の距離 ≥ ATR×1.0 |
| **パターン** | 1H足 二番底/二番天井（ATR×0.30以内） |
| **SL** | パターンの安値/高値 ± ATR×0.15 |
| **TP** | リスク幅 × 2.5倍（RR=2.5） |
| **半利確** | 1R到達でポジション50%決済 → SLをBE移動 |

---

## 3つのロジック

### Logic-A: GOLDYAGAMI（Goldロジック）

> **「引き算の美学 + 足し算の効果」**で生まれた現行主力ロジック。
> ADXやセッションフィルターを除去（引き算）し、日足EMA20アライメントとE2エントリーを追加（足し算）。
> 定量・計量分析で過学習なしを確認済み（IS-0.52 / OOS+0.13）。

| 項目 | 設定 |
|------|------|
| トレンドフィルター追加 | **日足EMA20方向一致**（クローズ > 日足EMA20） |
| エントリー方式 | **E2方式**: スパイク足除外（レンジ > ATR×2.0をスキップ）、足確定後2〜3分以内 |
| ADX | 不使用（過学習リスクで除去） |
| セッション | 不使用（過学習リスクで除去） |
| **採用銘柄** | **GBPUSD / USDCAD / NZDUSD / XAUUSD** |

**OOS成績（2025/03〜2026/02）**

| 銘柄 | PF | Sharpe | MDD | 月次+ |
|:---:|:---:|:---:|:---:|:---:|
| GBPUSD | 1.86 | 7.12 | 18.3% | 9/9 |
| USDCAD | 2.02 | 5.62 | 22.7% | 8/9 |
| NZDUSD | 1.98 | 5.45 | 20.5% | 8/9 |
| XAUUSD | 3.10 | 3.42 | 8.7% | 9/9 |

---

### Logic-C: オーパーツYAGAMI（v77ピュア）

> **「どこから来たのか分からないほど強い」** ことからオーパーツと命名。
> KMID + KLOW + 4H EMA20 + 1H二番底・二番天井のみのシンプル構成。
> 現行ロジック全ての土台であり、特定銘柄では最強性能を発揮。

| 項目 | 設定 |
|------|------|
| トレンドフィルター | 4H EMA20のみ（日足なし） |
| エントリー方式 | E1方式: 足確定後2分以内の1分足始値 |
| ADX | 不使用 |
| セッション | 不使用 |
| **採用銘柄** | **USDJPY / EURUSD** |

**OOS成績（2025/03〜2026/02）**

| 銘柄 | PF | Sharpe | MDD | 月次+ |
|:---:|:---:|:---:|:---:|:---:|
| USDJPY | 2.15 | 6.18 | 21.6% | 9/9 |
| EURUSD | 1.81 | 6.18 | 22.7% | 9/9 |

> USDJPY単体ではv77基準（PF 4.96）だが、ポートフォリオ統合後のOOS期間結果。

---

### Logic-B: ADX+Streak

> **「トレンドの強さ × 方向一貫性」** で厳選する補完ロジック。
> ADX≥20（強トレンド判定の業界標準）+ 直近4本の4H足が同方向（Streak≥4）の組み合わせ。
> GOLDロジックが僅差で劣後する銘柄（AUDUSD）に適用。

| 項目 | 設定 |
|------|------|
| ADXフィルター | 4H ADX ≥ 20（強トレンド判定） |
| Streakフィルター | 直近4本の4H足が同方向 |
| エントリー方式 | E1方式 |
| 日足EMA20 | 不使用 |
| **採用銘柄** | **AUDUSD** |

**OOS成績（2025/03〜2026/02）**

| 銘柄 | PF | Sharpe | MDD | 月次+ |
|:---:|:---:|:---:|:---:|:---:|
| AUDUSD | 2.03 | 3.66 | 23.2% | 7/9 |

---

## 採用7銘柄 × ポートフォリオ統合結果

| 優先 | 銘柄 | ロジック | OOS PF | Sharpe | 運用フェーズ |
|:---:|------|:---:|:---:|:---:|---|
| 1 | **USDJPY** | Logic-C（オーパーツ） | 2.15 | 6.18 | Phase1（即導入） |
| 2 | **GBPUSD** | Logic-A（GOLD） | 1.86 | 7.12 | Phase1（即導入） |
| 3 | **EURUSD** | Logic-C（オーパーツ） | 1.81 | 6.18 | Phase1（即導入） |
| 4 | **USDCAD** | Logic-A（GOLD） | 2.02 | 5.62 | Phase1（即導入） |
| 5 | **NZDUSD** | Logic-A（GOLD） | 1.98 | 5.45 | Phase2（3ヶ月後） |
| 6 | **XAUUSD** | Logic-A（GOLD） | 3.10 | 3.42 | Phase2（別カテゴリ） |
| 7 | **AUDUSD** | Logic-B（ADX+Streak） | 2.03 | 3.66 | Phase2（月次安定確認後） |

**ポートフォリオ統合（全7銘柄同時運用）**

| 指標 | 値 |
|------|:---:|
| PF | **1.97** |
| 年率Sharpe | **7.32** |
| MDD | **12.98%** |

---

## 運用フェーズ（リスクステージング）

| フェーズ | リスク/銘柄 | 対象 | 目的 |
|------|:---:|---|---|
| **Phase1** | 0.5% | 上位4銘柄（USDJPY/GBPUSD/EURUSD/USDCAD） | 執行確認・滑り確認 |
| **Phase2** | 1.0% | 全7銘柄 | 本番再現性確認（**標準運用**） |
| **Phase3** | 1.5〜2.0% | 全7銘柄 | 最大化（Sharpe 5.0+確認後） |

---

## ディレクトリ構成

```
sena3fx/
├── README.md
├── CLAUDE.md                        # AI開発ガイド（Claude Code用）
├── TRADING_MEMO.md                  # 本番運用ルール書
│
├── production/                      # ★ Exness×MT5 本番ライブトレーダー
│   ├── mt5_bot.py                   # MT5自動売買ボット本体
│   ├── review_manager.py            # PDCA レビュー管理（日次/週次/月次）
│   ├── signal_engine.py             # シグナル生成エンジン
│   ├── requirements.txt
│   ├── SETUP.md                     # VPSセットアップ手順
│   └── .env.example                 # 環境変数テンプレート
│
├── strategies/
│   ├── current/
│   │   ├── yagami_mtf_v79.py        # ★ YAGAMI改 本体（FX/METALS カテゴリ別フィルター）
│   │   └── yagami_mtf_v78.py        # 参照用（過学習チェック済）
│   └── archive/                     # v1〜v77（履歴）
│
├── cloud_run/                       # Cloud Run 本番コード（OANDA ペーパートレード）
│   ├── main.py                      # FastAPI エンドポイント
│   ├── Dockerfile
│   ├── requirements.txt
│   └── strategies/
│       ├── yagami_mtf_v79.py        # 本番用（FX/METALS）
│       ├── yagami_mtf_v77.py        # USDJPY本番用
│       └── archive/
│
├── scripts/                         # バックテスト・分析スクリプト
│   ├── backtest_logic_comparison.py # ★ Logic-A/B/C 比較バックテスト
│   ├── backtest_final_optimized.py  # ★ 最終採用銘柄確定バックテスト
│   ├── backtest_portfolio_integration.py  # ★ ポートフォリオ統合分析
│   ├── fetch_*.py                   # データ取得（OANDA API）
│   └── archive/                     # 旧バックテストスクリプト全件
│
├── data/                            # バックテスト用CSVデータ
│   ├── {symbol}_is_*.csv            # IS期間（2024/7〜2025/2）
│   ├── {symbol}_oos_*.csv           # OOS期間（2025/3〜2026/2）
│   └── ohlc/                        # 全期間1ファイル（大文字銘柄名）
│
├── utils/
│   ├── risk_manager.py              # AdaptiveRiskManager / SYMBOL_CONFIG
│   └── position_manager.py
│
├── results/                         # バックテスト結果（CSV・PNG）
│   ├── backtest_final_optimized.csv # 最終採用7銘柄の結果
│   ├── backtest_portfolio_integration.csv
│   └── approved_universe.json
│
├── trade_logs/                      # 本番・ペーパートレード実績ログ
├── docs/
│   ├── strategy_development_log_v60_v76.md
│   └── learning_materials/          # やがみ氏PDF教材
└── tests/
```

---

## セットアップ

### バックテスト環境

```bash
cd /home/user/sena3fx
pip install -r requirements.txt
python scripts/backtest_final_optimized.py
```

### 本番（Exness MT5 VPS）

```bash
cd production/
cp .env.example .env
# .env にMT5認証情報・Discord URLを記載
pip install -r requirements.txt
python mt5_bot.py
```

詳細手順: `production/SETUP.md` 参照

### Cloud Run（OANDA ペーパートレード）

```
OANDA_ACCOUNT_ID=xxx-xxx-xxxxxxxx-xxx
OANDA_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
GCS_BUCKET_NAME=sena3fx-paper-trading
```

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/run` | POST | 1サイクル実行（シグナル判定・注文） |
| `/report` | GET | 定時レポートをDiscordへ送信 |
| `/health` | GET | ヘルスチェック |
| `/status` | GET | 現在のオープンポジション一覧 |

---

## 注意事項

- `gcp-key.json` / `.env` は `.gitignore` 対象。コミット禁止
- USDJPYに1分足データは存在しない（15m/1h/4hのみ）
- `strategies/current/` と `cloud_run/strategies/` は常に同一内容を保つこと
- 旧バックテストスクリプトは `scripts/archive/`、旧戦略は `strategies/archive/` を参照
- `data/ohlc/` の銘柄名は大文字（`AUDUSD_1m.csv`）、`data/` は小文字（`audusd_1m.csv`）
