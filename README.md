# sena3fx — YAGAMI改 MTF自動取引エージェント

やがみメソッドをベースに定量・計量分析で再設計した **YAGAMI改** 戦略の自動化システム。
4時間足EMA20によるトレンドフィルター + 1時間足の二番底・二番天井パターン検出。
Google Cloud Run + OANDA API で本番取引を24時間稼働。

---

## 現在の稼働状況

| 項目 | 内容 |
|------|------|
| **稼働環境** | Google Cloud Run（asia-northeast1、min-instances=1） |
| **稼働モード** | **本番取引（OANDA API）** |
| **戦略名** | **YAGAMI改**（v80統一 / Logic-A / Logic-C 銘柄別最強選択） |
| **取引ペア** | 全8銘柄（下記ポートフォリオ参照） |
| **リスク** | 資産規模連動（EQUITY_RISK_TABLE） |
| **実行間隔** | 毎分（Cloud Scheduler） |
| **データ取得** | OANDA REST API（1分・1時間・4時間足） |
| **状態管理** | Google Cloud Storage（positions.json, trade_history.json） |

---

## 採用8銘柄 × バックテスト結果（同一エンジン検証済み）

| 銘柄 | ロジック | RR | OOS PF | Sharpe | MDD | Kelly | OOS/IS | 月次+ |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **USDCAD** | v80 KMID+KLOW+Body | 3.0 | **3.32** | 5.70 | 12.5% | 0.504 | 1.60 | 10/10 |
| **XAUUSD** | Logic-A GOLD tol=0.20 | 2.5 | **3.13** | 3.85 | 8.7% | 0.511 | 1.20 | 9/10 |
| **EURUSD** | v80 KMID+KLOW+Body | 3.0 | **2.61** | 5.70 | 23.5% | 0.434 | 1.17 | 10/10 |
| **AUDUSD** | v80 KMID+KLOW+Body | 3.0 | **2.52** | 5.90 | 16.7% | 0.408 | 1.27 | 10/10 |
| **GBPUSD** | v80 KMID+KLOW+Body | 3.0 | **2.24** | 6.39 | 20.0% | 0.394 | 0.85 | 10/10 |
| **USDCHF** | v80 KMID+KLOW+Body | 3.0 | **2.23** | 8.75 | 11.9% | 0.378 | 1.04 | 9/9 |
| **NZDUSD** | Logic-A GOLD tol=0.20 | 3.0 | **2.11** | 3.35 | 14.6% | 0.337 | 0.73 | 9/10 |
| **USDJPY** | Logic-C オーパーツ (v77) | 3.0 | **1.83** | 5.70 | 20.0% | 0.300 | 0.79 | 10/10 |

> 全銘柄、`backtest_final_optimized.py` で **現行2.5R / 現行3.0R / v80 3.0R** を同一条件比較し、最強ロジックを採用。

---

## 戦略概要 — YAGAMI改

### コンセプト

やがみメソッドに基づく**二番底・二番天井パターン**の自動検出。
マルチタイムフレーム（4時間足・1時間足・1分足）を組み合わせ、高確率なエントリーポイントを機械的に判定する。

### 戦略進化の系譜

```
v77 オーパーツYAGAMI（KMID+KLOW発見）
  ↓ 足し算の美学
v79 GOLDYAGAMI（日足EMA20+ADX+Streak をカテゴリ別追加）
  ↓ 引き算の美学（第二弾）
v80 Purified（指標予測力分析 → 不安定フィルター除去、安定2指標のみ残す）
```

### v80 Purified — 「引き算の美学」の結論

指標予測力分析（19,054件OOS）により、安定して効くフィルターだけを残した統一ロジック。

```
安定フィルター（採用）:
  ✅ KMID     — 4H足 実体方向一致（r=+0.122, 77%期間一貫）
  ✅ KLOW     — 4H足 下ヒゲ比率 < 0.15%
  ✅ 4Hボディ比率 — 4H足の実体/全体比率 ≥ 0.3（十字線除外）

不安定フィルター（除去）:
  ❌ 日足EMA方向一致 — 逆効果（match WR 26.9% < no-match 30.1%）
  ❌ ADX≥20         — 期間で効果反転
  ❌ Streak≥4       — ほぼ無予測力
  ❌ EMA距離        — 弱すぎて有意でない
```

### 銘柄別ロジック選択（同一エンジン検証）

全8銘柄で **現行2.5R / 現行3.0R / v80 3.0R** を同一バックテストエンジンで比較し、
OOS PFが最も高く、OOS/IS比≥0.70をクリアしたロジックを採用。

| ロジック | フィルター | エントリー | RR | 対象銘柄 |
|---|---|---|---|---|
| **v80 統一** | KMID + KLOW + 4Hボディ比率≥0.3 | E0（即時2分以内） | 3.0 | EURUSD, GBPUSD, USDCAD, USDCHF, AUDUSD |
| **Logic-A** | KMID + KLOW + 日足EMA20方向一致 + EMA距離 | E2（スパイク除外） | 2.5/3.0 | XAUUSD, NZDUSD |
| **Logic-C** | KMID + KLOW + 1H確認足方向 | E0（即時） | 3.0 | USDJPY |

### エントリー条件（全ロジック共通 AND）

| 条件 | 内容 |
|------|------|
| トレンド一致 | 4H EMA20より上→ロングのみ、下→ショートのみ |
| パターン成立 | 1H足 二番底/二番天井の安値/高値が ATR×tol_factor 以内 |
| **KMID** | 直前4H足の実体方向がエントリー方向と一致 |
| **KLOW** | 直前4H足の下ヒゲ比率 < 0.15%（`< 0.0015`） |
| **SL** | パターンの安値/高値 ± ATR×0.15 |
| **TP** | リスク幅 × RR倍（2.5R or 3.0R） |
| **半利確** | 1R到達でポジション50%決済 → SLをBE移動 |

### v80専用フィルター

| フィルター | 内容 | 効果 |
|---|---|---|
| 4Hボディ比率 ≥ 0.3 | 十字線（方向性の弱い足）を除外 | 低品質トレードのみ排除、PF維持・MDD改善 |

### Logic-A（GOLD）専用フィルター

| フィルター | 内容 | 対象銘柄 |
|---|---|---|
| 日足EMA20方向一致 | 日足レベルのトレンド整合性 | XAUUSD, NZDUSD |
| EMA距離 ≥ ATR×1.0 | トレンドの強さ確認 | XAUUSD, NZDUSD |
| tol_factor=0.20 | パターン許容幅を厳格化（MDD削減） | XAUUSD, NZDUSD |

---

## デプロイ

### 前提条件

1. `gcloud` CLI がインストール・認証済み
2. `deploy/.env` に環境変数を設定済み
3. GCPプロジェクトが作成済み

### 手順

```bash
cd ~/sena3fx

# 1. デプロイ（ビルド・Cloud Run・Scheduler 全自動）
bash deploy/deploy_gcp.sh

# 2. ヘルスチェック
curl https://<SERVICE_URL>/health

# 3. ステータス確認
curl https://<SERVICE_URL>/status
```

### 環境変数（deploy/.env）

```
BROKER=oanda
OANDA_TOKEN=<OANDA APIトークン>
OANDA_ACCOUNT=<OANDA アカウントID>
DISCORD_WEBHOOK=<Discord Webhook URL>
GCS_BUCKET=sena3fx-paper-trading
GCP_PROJECT=aiyagami
```

---

## Cloud Run エンドポイント

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/run` | POST | 1サイクル実行（シグナル判定・注文） |
| `/health` | GET | ヘルスチェック |
| `/report` | POST | 定時レポートをDiscordへ送信 |
| `/status` | GET | 現在のオープンポジション・動的リスク一覧 |
| `/notify_test` | POST | Discord通知テスト |
| `/weekly_feedback` | POST | 週次GCSログ記録 |
| `/feedback_history` | GET | フィードバック履歴取得 |
| `/test_trade` | POST | テスト取引（最小ロット買い→即決済） |
| `/debug_broker` | GET | ブローカー接続診断 |

---

## ディレクトリ構成

```
sena3fx/
├── README.md
├── CLAUDE.md                        # AI開発ガイド（Claude Code用）
├── TRADING_MEMO.md                  # 本番運用ルール書
│
├── cloud_run/                       # ★ Cloud Run 本番コード
│   ├── main.py                      #   FastAPI エンドポイント（全8銘柄）
│   ├── Dockerfile
│   ├── requirements.txt
│   └── strategies/
│       ├── yagami_mtf_v79.py        #   v80 / Logic-A / Logic-B 本番用
│       ├── yagami_mtf_v77.py        #   Logic-C 本番用（USDJPY）
│       └── yagami_mtf_v78.py        #   参照用（過学習チェック済）
│
├── strategies/
│   └── current/
│       ├── yagami_mtf_v79.py        # cloud_run/strategies/ と同期必須
│       └── yagami_mtf_v78.py        # 参照用
│
├── utils/
│   └── risk_manager.py              # RiskManager / SYMBOL_CONFIG
│
├── scripts/                         # バックテスト・データ取得
│   ├── backtest_final_optimized.py  # ★ 最終バックテスト（現行vs v80 3.0R比較）
│   ├── backtest_v80_purified.py     # v80フィルター5種×RR2種テスト
│   ├── backtest_all_symbols.py      # 全15銘柄×3ロジック統合BT
│   ├── analyze_indicator_predictive_power.py  # 指標予測力分析
│   ├── fetch_*.py                   # データ取得（OANDA API）
│   └── generate_htf_from_1m.py      # 1分足→上位足生成
│
├── data/                            # バックテスト用CSVデータ
│   ├── ohlc/                        # 1分足（大文字銘柄名）
│   └── {symbol}_{tf}.csv            # IS/OOS分割データ
│
├── results/                         # バックテスト結果（CSV）
├── deploy/                          # デプロイスクリプト
│   ├── deploy_gcp.sh               # Cloud Run 全自動デプロイ
│   └── .env                        # 環境変数（.gitignore対象）
├── docs/                            # ドキュメント・やがみ氏PDF教材
└── tests/                           # テスト
```

---

## バックテスト実行

```bash
cd ~/sena3fx

# 最終比較バックテスト（現行2.5R vs 現行3.0R vs v80 3.0R × 8銘柄）
python scripts/backtest_final_optimized.py

# v80フィルター精査（5バリアント × 2RR × 7銘柄）
python scripts/backtest_v80_purified.py

# 全15銘柄×3ロジック統合バックテスト
python scripts/backtest_all_symbols.py
```

---

## 注意事項

- `gcp-key.json` / `deploy/.env` は `.gitignore` 対象。コミット禁止
- `strategies/current/` と `cloud_run/strategies/` は常に同一内容を保つこと
- USDJPYに1分足データは存在しない（15m/1h/4hのみ）
- 詳細な運用ルールは `TRADING_MEMO.md` を参照
