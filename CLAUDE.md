# CLAUDE.md

sena3fx — **YAGAMI改** MTF自動取引エージェント

## プロジェクト概要

やがみメソッドをベースに再設計した **YAGAMI改** 戦略の自動化システム。
Cloud Run上でペーパートレードを24時間稼働させ、PDCAサイクルで継続的に戦略を改善する。

> **YAGAMI改** = やがみメソッドのエッセンス（二番底・二番天井 + MTFアライメント）を
> 定量・計量分析で再設計した次世代実装。旧v77以前とは別物として位置づける。

---

## 戦略命名系譜（YAGAMI家系図）

| 名称 | バージョン | 別名 | 概要 |
|---|---|---|---|
| **オーパーツYAGAMI** | v77 | v77ピュア / Logic-C | やがみメソッドの定量実装初号機。KMID+KLOW+4H EMA20+1H二番底天井。「どこから来たのか分からないほど強い」ことからオーパーツと命名。現行ロジック全ての土台 |
| **GOLDYAGAMI** | v79/v80 | Goldロジック / Logic-A | オーパーツYAGAMIに「引き算の美学（ADX/セッション除去）+ 足し算の効果（日足EMA20+EMA距離）」を施した現行主力ロジック。定量・計量分析で過学習なしを確認済み |
| **ニュージーランドYAGAMI** | — | NZ-YAGAMI | GOLDYAGAMIをNZDUSD専用に特化。アダプティブリスク2→3%（勝ち+0.5%/負け-0.5%）＋手数料モデル（5USD/lot片道）込み。`backtest_nzdusd_3pct.py` で実装 |

### GOLDYAGAMI（v79/v80）の確定フィルター

```
ベース（全銘柄共通）:
  ✅ KMID     — 4H文脈足 実体方向一致
  ✅ KLOW     — 4H文脈足 下ヒゲ比率 < 0.15%
  ✅ EMA距離  — 4H close が EMA20 から ATR×1.0 以上

カテゴリ別:
  FX銘柄   : Streak≥4（直近4本の4H足が同方向）+ E1エントリー
  XAUUSD   : 日足EMA20方向一致（1d_trend）+ E2エントリー（スパイク除外）

除去確定（過学習リスク）:
  ❌ ADX / セッション時間 / ATR拡張 / 確認足方向チェック
```

---

## Commands

| Command | Description |
|---------|-------------|
| `pip install -r requirements.txt` | 依存パッケージインストール（numpy, pandas, hmmlearn） |
| `python -m pytest tests/` | テスト実行 |
| `bash deploy/deploy_gcp.sh` | Cloud Run デプロイ（GCP） |
| `cd cloud_run && uvicorn main:app --reload` | Cloud Run ローカル起動 |
| `python scripts/fetch_all_ohlc.py` | 全銘柄OHLCデータ取得（OANDA API） |
| `python scripts/fetch_fx_robust.py` | FXデータ取得（リトライ付き） |
| `python scripts/generate_htf_from_1m.py` | 1分足から上位足（15m/1h/4h）生成 |
| `python scripts/backtest_final_optimized.py` | 最終採用銘柄バックテスト |
| `python scripts/backtest_all_symbols.py` | 全15銘柄×3ロジック統合バックテスト |

---

- **戦略名**: **YAGAMI改**（2026/3/9〜。旧バージョン名: v79/v78/v77…はアーカイブ済み）
- **現行ファイル**: `yagami_mtf_v79.py`（FX/METALS）/ `yagami_mtf_v77.py`（USDJPY本番用、cloud_run保持）
- **メイン銘柄**: USDJPY / XAUUSD
- **対象銘柄**: USDJPY / EURJPY / GBPJPY / XAUUSD / SPX500 / US30 / NAS100 他
- **データソース**: OANDA API（リアルタイム）/ ローカルCSV（バックテスト用）
- **ブランチ**: `main`
- **リモート**: GitHub `saltstarpx/sena3fx`
- **作業ディレクトリ**: `/home/user/sena3fx`（`/home/ubuntu/` ではない）

---

## 現在の戦略: YAGAMI改（FX + METALS カテゴリ別フィルター）

### 概要

4時間足EMA20によるトレンドフィルター + 1時間足の二番底・二番天井パターン検出。
エントリーは足更新後2分以内の1分足始値（成行）。半利確あり（1R到達でポジション半分決済・SLをBEへ）。

- **XAUUSD（貴金属）**: `yagami_mtf_v79.py`（v79A: 日足EMA20方向一致）
- **EURUSD/GBPUSD/AUDUSD（FX）**: `yagami_mtf_v79.py`（v79BC: ADX≥20 + Streak≥4）
- **USDJPY**: `yagami_mtf_v77.py`（1m足なし、v79バックテスト対象外）

### v79の追加フィルター（v77からの変更点）

定量分析（OOS全銘柄カイ二乗検定）・カテゴリ別過学習チェックに基づく改善。

```python
ADX_THRESHOLD_DEFAULT = 20   # v79B: 強トレンドの業界標準基準値（データ非依存）
STREAK_MIN_DEFAULT    = 4    # v79C: 直近4本の4H足が同方向（固定値）
# v79A: use_1d_trend=True → 日足EMA20方向一致（パラメータなし）
```

| フィルター | 内容 | 対象カテゴリ | 効果（OOS） |
|------|------|------|------|
| **v79A** | 日足EMA20 方向一致（MTFアライメント強化） | METALS | XAUUSD PF 2.03→2.16（+6.4%） |
| **v79B** | 4H ADX ≥ 20（レンジ相場排除） | FX | avg PF 1.82→1.90（+4.4%） |
| **v79C** | 直近4本の4H足が同方向（トレンド一貫性） | FX | GBPUSD PF 2.17達成 |
| **v79BC** | B+C 組合せ | FX | avg PF 1.82→**1.98**（GBPUSD 2.17） |

### 定量・計量分析の主要発見（OOS期間）

| 発見 | 銘柄 | 詳細 | 統計的有意性 |
|------|------|------|------|
| 夜間帯（UTC20-24）の勝率低下 | XAUUSD | WR 21.3% vs 基準32.3% | p=0.014 ** |
| 低ボラティリティ時の勝率向上 | GBPUSD | WR 40.4% vs 基準28.7% | p=0.0006 *** |
| ADX15-20帯（半トレンド）でEURUSD勝率向上 | EURUSD | WR 32.1% vs 基準27.7% | p=0.013 ** |
| EMA距離が大きいほど勝率向上傾向 | XAUUSD | EMA遠2-3ATR: +5.7% | p=0.078 * |

### v79バックテスト結果（OOS: 2025/03〜2026/02）

| カテゴリ | バリアント | 銘柄 | v77 PF | v79 PF | 変化 | IS/OOS乖離 |
|------|------|------|:---:|:---:|:---:|:---:|
| FX | v79BC | EURUSD | 1.78 | **1.87** | +5.1% | IS-0.02/OOS+0.10 |
| FX | v79BC | GBPUSD | 1.73 | **2.17** | +25.4% | — |
| FX | v79BC | AUDUSD | 1.95 | 1.90 | -2.6% | — |
| FX avg | v79BC | — | 1.82 | **1.98** | +8.8% | 過学習なし |
| METALS | v79A | XAUUSD | 2.03 | **2.16** | +6.4% | IS-0.52/OOS+0.13 ✅ |
| INDICES | — | US30/SPX/NAS | 1.08 | — | — | 採用不可 |

### v77バックテスト結果（USDJPY, 2025/1-12, spread=0.4pips）

| 指標 | v76 | **v77** |
|------|:---:|:---:|
| トレード数 | 373回 | **327回**（-12%） |
| 勝率 | 56.6% | **76.1%** |
| PF | 2.17 | **4.96** |
| 総損益 | +8,227pips | **+12,551pips** |
| MDD | 460.9pips | **222.6pips** |
| 月次シャープ | 5.57 | **10.47** |
| ケリー基準 | 0.305 | **0.608** |
| 最大連敗 | 5回 | **3回** |
| プラス月 | 12/12 | **12/12** |

### 過去のバックテスト詳細

カテゴリ別改善検証・3戦略比較・v77過学習検証の詳細結果は `results/` 配下のCSVファイルを参照。

**主要な結論:**
- セッションフィルター（UTC7-22）はFXで有効だがXAUUSDには逆効果（v78の過学習を確認）
- v77が全銘柄で最高性能。F1+F3の時間フィルターは逆効果
- v77過学習検証: IS/OOS分割・ウォークフォワード・ブートストラップ・閾値感度・全期間 → 5段階全PASS
- 指数（US30/SPX500/NAS100）は全バリアントでPF<1.5、採用不可

---

## ディレクトリ構成

```
sena3fx/
├── CLAUDE.md                        # AI開発ガイド（このファイル）
├── TRADING_MEMO.md                  # 本番トレード方針メモ
├── README.md
├── requirements.txt                 # 依存パッケージ（numpy, pandas, hmmlearn）
│
├── strategies/
│   ├── current/
│   │   ├── yagami_mtf_v79.py        # ★ 現行戦略 v79（FX/METALS カテゴリ別フィルター）
│   │   └── yagami_mtf_v78.py        # 参照用v78（過学習チェック済）
│   └── archive/                     # v1〜v77（履歴・アーカイブ済）
│
├── cloud_run/                       # Cloud Run本番コード
│   ├── main.py                      # FastAPI エンドポイント（全銘柄統合）
│   ├── Dockerfile
│   ├── requirements.txt             # Cloud Run用依存（fastapi, uvicorn, gcs等）
│   ├── gcp-key.json                 # GCPキー（.gitignore対象）
│   └── strategies/
│       ├── yagami_mtf_v79.py        # 本番用（YAGAMI改 FX/METALS）
│       ├── yagami_mtf_v78.py        # 参照用（過学習チェック済）
│       └── yagami_mtf_v77.py        # USDJPY本番用（cloud_runのみ保持）
│
├── deploy/                          # デプロイ関連
│   ├── deploy_gcp.sh               # Cloud Run デプロイスクリプト
│   └── union_kelly_monitor.service  # systemdサービス定義
│
├── data/                            # バックテスト用CSVデータ（IS/OOS分割）
│   ├── {symbol}_is_*.csv            # 各銘柄IS（小文字: audusd, eurusd等）
│   ├── {symbol}_oos_*.csv           # 各銘柄OOS
│   └── ohlc/                        # 全期間1ファイル（大文字: AUDUSD_1m.csv等）
│
├── scripts/                         # データ取得・バックテストスクリプト
│   ├── backtest_all_symbols.py      # 全15銘柄×3ロジック統合バックテスト
│   ├── backtest_final_optimized.py  # 最終採用銘柄バックテスト
│   ├── backtest_logic_comparison.py # ロジック比較（A/B/C）
│   ├── backtest_portfolio_integration.py # ポートフォリオ統合分析
│   ├── fetch_data.py                # データ取得（OANDA API）
│   ├── fetch_all_ohlc.py            # 全銘柄OHLC取得
│   ├── fetch_fx_robust.py           # FXデータ取得（リトライ付き）
│   ├── generate_htf_from_1m.py      # 1mから上位足生成
│   ├── main_loop.py                 # メインループ
│   └── （他 fetch_*.py 多数）       # 各種データ取得スクリプト
│
├── utils/
│   └── risk_manager.py              # ★ AdaptiveRiskManager / SYMBOL_CONFIG
│
├── backtest/
│   └── archive/                     # VectorBT等（アーカイブ済）
│
├── tools/                           # Windows用MT5ボット管理
│   ├── setup.bat / start_bot.bat    # セットアップ・起動
│   └── stop_bot.bat                 # 停止
│
├── reports/                         # PDCAサイクルレポート・ダッシュボード
│   └── dashboard.html
│
├── trade_logs/                      # ペーパートレード実績ログ
│
├── results/                         # バックテスト結果（PNG・CSV）
│   ├── backtest_final_optimized.csv # 最終採用銘柄結果
│   ├── approved_universe_v77.json   # 採用銘柄リスト
│   └── ...
│
├── docs/
│   ├── strategy_development_log_v60_v76.md
│   └── learning_materials/          # やがみ氏PDF教材7冊
│
└── tests/
    └── test_oanda_env_compat.py
```

---

## データ仕様（重要）

### ⚠️ データには2系統ある

| 系統 | パス | 命名規則 | 特徴 |
|---|---|---|---|
| **IS/OOS分割** | `data/` | `{symbol}_is_*.csv` / `{symbol}_oos_*.csv`（小文字） | 期間分割済み、スクリプト主流 |
| **全期間1ファイル** | `data/ohlc/` | `{SYMBOL}_*.csv`（大文字） | manusが整備、全期間連続 |

### data/ の主要ファイル

| 銘柄 | IS期間 | OOS期間 | 利用可能TF |
|---|---|---|---|
| USDJPY | 2024/7〜2025/2 | 2025/3〜2026/2 | 15m / 1h / 4h（1mなし） |
| XAUUSD | 2025/1〜2025/2 | 2025/3〜2026/2 | 15m / 1h / 4h / 全期間1m(409,615行) |
| AUDUSD / EURUSD / GBPUSD | 同上 | 同上 | 15m / 1h / 4h / 全期間1m |
| EURJPY / GBPJPY | 同上 | 同上 | 15m / 1h / 4h（1mなし） |
| NAS100 / SPX500 / US30 | 同上 | 同上 | 15m / 1h / 4h / 1m(IS+OOS分) |

カラム: `timestamp, open, high, low, close, volume`
タイムゾーン: UTC（インデックス後 `tz_localize('UTC')` が必要な場合あり）

**「データがない」「代替データで構成」という判断は誤り。全ファイルが存在する。**

---

## utils/risk_manager.py（必須インポート）

バックテストスクリプトでは必ずこのモジュールを使用すること。

```python
from utils.risk_manager import SYMBOL_CONFIG, AdaptiveRiskManager
```

### SYMBOL_CONFIG（主要銘柄）

| 銘柄 | pip | spread | quote_type | 口座 |
|---|---|---|---|---|
| USDJPY | 0.01 | 0.0pips | A（JPY建て） | raw_spread |
| EURUSD | 0.0001 | 0.0pips | B（USD建て） | raw_spread |
| GBPUSD | 0.0001 | 0.1pips | B | raw_spread |
| AUDUSD | 0.0001 | 0.0pips | B | raw_spread |
| EURJPY | 0.01 | 2.4pips | A | standard |
| GBPJPY | 0.01 | 2.2pips | A | standard |
| US30 | 1.0 | 0.8pips | D（指数） | zero |
| SPX500 | 0.1 | 0.1pips | D | zero |
| NAS100 | 1.0 | 8.3pips | D | zero |
| XAUUSD | 0.01 | 5.2pips | B | raw_spread |
| XAGUSD | 0.001 | 2.6pips | B | raw_spread |

---

## やがみメソッド（戦略の根拠）

### 基本構造

1. **4時間足でトレンド方向を確認**（EMA20との位置関係）
2. **1時間足で二番底・二番天井パターンを検出**
3. **1分足でエントリー**（足更新後2分以内の始値）

### v77のエントリー条件（全条件AND）

| 条件 | 内容 |
|------|------|
| トレンド一致 | 4H EMA20より上→ロングのみ、下→ショートのみ |
| パターン成立 | 二番底（ロング）/ 二番天井（ショート）の安値/高値が ATR×0.3 以内 |
| 確認足 | パターン形成後の足が方向一致の実体（陽線/陰線） |
| **KMID** | **直前4H足の実体方向がエントリー方向と一致** |
| **KLOW** | **直前4H足の下ヒゲ比率 < 0.15%（`< 0.0015`）** |
| リスク幅 | SLまでの距離が ATR×2（1H基準）以内 |

### SL/TP設定

- **SL**: 二番底/天井の安値・高値から ATR×0.15 外側
- **TP**: SLリスク幅 × 2.5倍（RR=2.5）
- **半利確**: 1R到達でポジション半分決済 → SLをBEへ移動

---

## バックテスト実行方法

```bash
cd /home/user/sena3fx  # ← /home/ubuntu/ ではない

# 最終採用銘柄バックテスト（7銘柄）
python scripts/backtest_final_optimized.py

# 全15銘柄×3ロジック統合バックテスト
python scripts/backtest_all_symbols.py

# ロジック比較（Logic-A/B/C）
python scripts/backtest_logic_comparison.py

# ポートフォリオ統合分析
python scripts/backtest_portfolio_integration.py
```

### バックテスト合格基準（YAGAMI改）

| 指標 | 基準 |
|------|------|
| プロフィットファクター | ≥ 3.0 |
| 勝率 | ≥ 65% |
| MDD（ピーク比） | ≤ 20% |
| ケリー基準 | ≥ 0.45 |
| プラス月 | ≥ 90% |
| OOS PF ≥ IS PF × 0.7 | （過学習チェック） |

---

## Cloud Run エンドポイント

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/run` | POST | 1サイクル実行（シグナル判定・注文） |
| `/health` | GET | ヘルスチェック |
| `/report` | POST | 定時レポートをDiscordへ送信 |
| `/status` | GET | 現在のオープンポジション・動的リスク一覧 |
| `/notify_test` | POST | Discord通知テスト（採用銘柄一覧送信） |
| `/weekly_feedback` | POST | 週次GCSログ記録 |
| `/feedback_history` | GET | フィードバック履歴取得 |
| `/test_trade` | POST | テスト取引（GBPUSD最小ロット買い→即決済） |
| `/debug_broker` | GET | ブローカー接続診断（価格取得・口座情報） |

---

## 注意事項

- `strategies/current/yagami_mtf_v79.py` と `cloud_run/strategies/yagami_mtf_v79.py` は**常に同一内容**を保つこと
- `strategies/current/yagami_mtf_v78.py` と `cloud_run/strategies/yagami_mtf_v78.py` は**常に同一内容**を保つこと
- v77は `cloud_run/strategies/yagami_mtf_v77.py` にのみ存在（`strategies/current/` にはない）
- `data/` のCSVファイルはバックテスト専用。ペーパートレードはOANDA APIからリアルタイム取得
- `gcp-key.json` は `.gitignore` 対象。コミットしないこと
- **USDJPYに1分足データは存在しない**（15m/1h/4hのみ）。バックテストは15mで代用
- `data/ohlc/` の銘柄名は大文字（`AUDUSD_1m.csv`）、`data/` の銘柄名は小文字（`audusd_1m.csv`）
- やがみPDF教材（`docs/learning_materials/`）が戦略のソースオブトゥルース

---

## バージョン履歴

| バージョン | 採用日 | 主な変更 | PF（OOS） |
|---|---|---|:---:|
| **v79** | 2026/3/9 | 定量・計量分析による MDD対策+トレンドフォロー強化（カテゴリ別フィルター） | FX avg **1.98** / XAUUSD **2.16** |
| v78 | 2026/3/9 | XAUUSD専用4改善（過学習の可能性あり、参照用に保持） | 2.50（XAUUSD、UTC5-21データ依存） |
| **v77** | 2026/3/7 | KMID+KLOWフィルター追加（qlib Alpha158系） | **4.96**（USDJPY） |
| v76 | 2026/2 | スプレッド計算バグ修正（チャートレベル固定） | 2.39 |
| v75 | 2026/1 | スプレッド計算バグあり（非推奨） | — |
| v60〜v74 | 2025 | 各種パラメータ最適化 | — |

詳細: `docs/strategy_development_log_v60_v76.md`

---

## 現在の採用銘柄・戦略（2026/3/9時点）

| 銘柄 | 戦略 | PF（OOS） | MDD | Kelly | 月次プラス |
|---|---|:---:|:---:|:---:|:---:|
| **USDJPY** | v77 | **4.96** | 222.6pips | 0.608 | 12/12 |
| **XAUUSD** | **v79A（Goldロジック確定）** | **2.16** | （改善） | 改善 | 12/12 |

次の候補: GBPUSD（Goldロジック適用、PF=1.86 ← FXカテゴリ最高）

### ⭐ 全銘柄共通ロジック確定（2026/3/9）

定量・計量分析 + ロジック比較バックテスト（`scripts/backtest_logic_comparison.py`）により、
**全銘柄にGoldロジック（v79A: 日足EMA20方向一致 + E2エントリー）を適用する**ことが確定。

#### ロジック比較結果（OOS: 2025-06〜2026-02）

| 銘柄 | Logic-A PF（Goldロジック） | Logic-B PF（ADX+Streak） | 推奨 |
|---|:---:|:---:|:---:|
| EURUSD | **1.73** | 1.61 | **Logic-A** |
| GBPUSD | **1.86** | 1.66 | **Logic-A** |
| AUDUSD | 1.98 | **2.05** | Logic-B（僅差） |
| NAS100 | **1.27** | 1.13 | Logic-A（指数は採用不可） |
| SPX500 | 1.94 | **2.04** | Logic-B（指数は採用不可） |
| US30 | 1.41 | **1.51** | Logic-B（指数は採用不可） |
| **FX avg** | **1.85** | 1.77 | **Logic-A（+0.08）** |

> **⚠️ 指数（NAS100/SPX500/US30）はどちらのロジックでも採用基準（PF≥2.0）未達。引き続き採用不可。**

#### Goldロジック（全銘柄統一）の設定

```python
# 全銘柄（XAUUSD / EURUSD / GBPUSD / AUDUSD）: Goldロジック統一
# 日足EMA20方向一致 + E2エントリー（スパイク除外、2-3分以内）
from scripts.backtest_logic_comparison import generate_signals
sigs = generate_signals(data_1m, data_15m, data_4h,
                        spread=spread_price,
                        logic="A",          # ← Goldロジック固定
                        atr_1m_d=atr_dict,
                        m1c=m1_cache)
```

#### Goldロジックの詳細

| 項目 | 設定 |
|---|---|
| トレンドフィルター① | 4H EMA20（クローズ > EMA20 → Long方向のみ） |
| トレンドフィルター② | **日足EMA20方向一致**（クローズ > 日足EMA20） |
| KMIDフィルター | 直前4H足の実体方向がエントリー方向と一致 |
| KLOWフィルター | 直前4H足の下ヒゲ比率 < 0.15% |
| EMA距離フィルター | 4H終値とEMA20の距離 ≥ ATR×1.0 |
| パターン | 1H足二番底/二番天井（ATR×0.30以内） |
| エントリー | **E2方式**: スパイク除外（レンジ > ATR×2.0の足をスキップ）、2-3分以内 |
| SL | 二番底/天井の安値・高値 ± ATR×0.15 |
| TP | リスク幅 × 2.5倍（RR=2.5） |
| 半利確 | 1R到達でポジション50%決済 → SLをBEへ |

### カテゴリ別推奨バリアント（2026/3/9 最終確定）

| カテゴリ | 銘柄 | ロジック | avg OOS PF | 備考 |
|---|---|:---:|:---:|---|
| 貴金属 | **XAUUSD** | **Goldロジック** | **2.16** | 確定採用。直近3ヶ月(12-02)PF=2.70 |
| FX | EURUSD/GBPUSD/AUDUSD | **Goldロジック** | **1.85** | GBPUSD最高(1.86)、FX全銘柄で優位 |
| 指数 | US30/SPX500/NAS100 | — | <2.0 | 全ロジックで採用基準未達 |

### v78の過学習に関する注記

v78D（UTC5-21、実体最小値、許容幅縮小）はXAUUSD OOSデータ上でPF2.50を記録したが、
カテゴリ別検証（経済的根拠ベースのUTC7-22）では**v77以下（PF1.86）**に低下した。
v79A（日足EMA20）は同様の改善効果を過学習なしで実現する（IS-0.52/OOS+0.13）。

---

## セッション引き継ぎログ

### 2026-03-11〜12 セッション完了内容

- 全15銘柄×3ロジック統合バックテスト完了（`scripts/backtest_all_symbols.py`）
- 最終採用7銘柄確定（`results/backtest_final_optimized.csv`）
- ポートフォリオ統合分析完了（PF=1.97、Sharpe=7.32、MDD=12.98%）
- 本番運用ルール書（`TRADING_MEMO.md`）整備
- 採用基準: OOS PF≥1.30 / Sharpe≥2.0 / OOS/IS≥0.70 / 月次+≥70% / MDD≤25%
- リスクステージング: Phase1(0.5%)→Phase2(1.0%)→Phase3(1.5-2.0%)

#### ⚠️ 未解決課題

- バックテスト合格基準はPF≥3.0（単体）だが、ポートフォリオ採用基準はPF≥1.30（分散前提）
- MT5本番ボット（`tools/` にWindows用バッチあり）の本番稼働は未確認
