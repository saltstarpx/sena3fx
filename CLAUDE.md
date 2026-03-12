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

### カテゴリ別改善バックテスト結果（2026/3/9実施）

`scripts/backtest_category_improvements.py` による過学習対策付きカテゴリ別検証。

#### 設計方針（過学習防止）
- **セッション時間はOOSデータ非依存**で設定（経済的根拠ベース）
  - FX/貴金属: UTC 7-22（London+NY主要セッション）
  - 指数: UTC 14-22（NYSE通常取引時間）
- **カテゴリ内同一パラメータ**使用（銘柄ごとにチューニングしない）
- **カテゴリPASS基準**: 過半数銘柄でOOS PFが改善すること

#### FXカテゴリ（EURUSD / GBPUSD / AUDUSD、UTC 7-22）

| バリアント | EURUSD | GBPUSD | AUDUSD | カテゴリ avg PF | 判定 |
|------|:---:|:---:|:---:|:---:|:---:|
| v77 | 1.78 | 1.73 | 1.95 | 1.82 | baseline |
| +A (1H KLOW) | 1.77 | 1.73 | 1.95 | 1.82 | ❌ 0/3改善 |
| **+B (セッションフィルター)** | **1.82** | **2.01** | 1.94 | **1.92** | **✅ 2/3改善** |
| +C (実体最小値) | 1.62 | 1.85 | 1.63 | 1.70 | ❌ 1/3改善 |
| +D (許容幅縮小) | 1.60 | 1.73 | 1.53 | 1.62 | ❌ 0/3改善 |

**FX推奨: v77 + 改善B（セッションフィルター UTC7-22）のみ**
過学習チェック: IS+0.11 / OOS+0.10 → 乖離小、過学習なし

#### 貴金属カテゴリ（XAUUSD、UTC 7-22）

| バリアント | XAUUSD OOS PF | 判定 |
|------|:---:|:---:|
| **v77** | **2.03** | **✅ best** |
| +A | 2.00 | ❌ |
| +B (UTC7-22) | 1.86 | ❌ |
| +C | 1.86 | ❌ |
| +D | 1.98 | ❌ |

**⚠️ 重要発見**: 経済的根拠ベースのUTC 7-22ではXAUUSDにセッションフィルターは逆効果。
旧v78でUTC 5-21が有効だったのはXAUUSD OOSデータへの**過学習**であった可能性が高い。
**METALS推奨: v77（改善なし）**

#### 指数カテゴリ（US30 / SPX500 / NAS100、UTC 14-22）

| バリアント | US30 | SPX500 | NAS100 | avg PF | 判定 |
|------|:---:|:---:|:---:|:---:|:---:|
| v77 | 0.93 | 0.92 | 0.89 | 0.91 | PF<1.0 |
| +B | 1.10 | 1.02 | 1.26 | 1.13 | ✅ 3/3改善 **⚠️過学習疑い** |

IS改善+0.95に対してOOS改善+0.21 → IS/OOS乖離大、過学習フラグ。
avg PF=1.13はいずれも採用基準（PF≥2.0）に遠く未達。
**指数推奨: 現戦略では採用不可（全バリアントでPF<1.5）**

### 3戦略比較バックテスト結果（2026/3/8実施, 7銘柄）

v77 / F1+F3（UTC5-15時フィルター+4H&1Hパターン）/ Hybrid（F1+F3構造+v77式KLOW）を比較。

| 銘柄 | v77 PF | F1+F3 PF | Hybrid PF | 推奨 | 合否 |
|------|:---:|:---:|:---:|:---:|:---:|
| **XAUUSD** | **4.57** | 3.66 | 3.16 | **v77** | ✅全PASS |
| SPX500 | 2.96 | 2.46 | 2.17 | v77 | ❌PF不足 |
| GBPUSD | 2.61 | 2.26 | 2.39 | v77 | ❌PF不足 |
| US30 | 2.49 | 2.22 | 2.03 | v77 | ❌PF不足 |
| AUDUSD | 2.33 | 2.14 | 1.92 | v77 | ❌PF不足 |
| EURUSD | 2.28 | 1.69 | 1.69 | v77 | ❌PF不足 |
| NAS100 | 2.13 | 2.01 | 1.55 | v77 | ❌PF不足 |

**結論**: v77が常に最高性能。F1+F3の時間フィルターはこのデータセットでは逆効果。

### v77過学習検証（5段階、全PASS）

| 検証 | 結果 | 詳細 |
|------|:---:|------|
| IS/OOS分割 | **PASS** | OOSの方が効果大（PF改善 IS+1.7 → OOS+3.3） |
| ウォークフォワード | **PASS** | 12/12月（100%）でPF改善 |
| ブートストラップ | **PASS** | p=0.0000（PF差・勝率差とも統計的有意） |
| 閾値感度 | **PASS** | KLOW 0.0005〜0.005全範囲でPF>3.5 |
| 全期間（20ヶ月） | **PASS** | PF 5.26, 20/20月プラス |

---

## ディレクトリ構成

```
sena3fx/
├── CLAUDE.md                        # AI開発ガイド（このファイル）
├── TRADING_MEMO.md                  # 本番トレード方針メモ
├── README.md
│
├── strategies/
│   ├── current/
│   │   ├── yagami_mtf_v79.py        # ★ 現行戦略 v79（FX/METALS カテゴリ別フィルター）
│   │   ├── yagami_mtf_v78.py        # 参照用v78（XAUUSD専用過学習の可能性あり）
│   │   └── yagami_mtf_v78.py        # 参照用（過学習チェック済）
│   └── archive/                     # v1〜v77（履歴・アーカイブ済）
│                                    # ※ v76, v76_improved, v77 はアーカイブ済み
│
├── cloud_run/                       # Cloud Run本番コード
│   ├── main.py                      # FastAPI エンドポイント（USDJPY=v77, FX/METALS=v79）
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── gcp-key.json                 # GCPキー（.gitignore対象）
│   └── strategies/
│       ├── yagami_mtf_v79.py        # 本番用（YAGAMI改 FX/METALS）
│       ├── yagami_mtf_v78.py        # 参照用（過学習チェック済）
│       ├── yagami_mtf_v77.py        # USDJPY本番用（cloud_runのみ保持）
│       └── archive/                 # v76 アーカイブ済み
│
├── data/                            # バックテスト用CSVデータ（IS/OOS分割）
│   ├── usdjpy_is_15m.csv            # IS 15分足（2024/7〜2025/2, 16,629行）
│   ├── usdjpy_is_1h.csv             # IS 1時間足
│   ├── usdjpy_is_4h.csv             # IS 4時間足（1,042行）
│   ├── usdjpy_oos_15m.csv           # OOS 15分足（2025/3〜2026/2, 24,814行）
│   ├── usdjpy_oos_1h.csv            # OOS 1時間足
│   ├── usdjpy_oos_4h.csv            # OOS 4時間足（1,553行）
│   ├── xauusd_is_15m.csv / _4h.csv  # XAUUSD IS（2025/1〜2025/2）
│   ├── xauusd_oos_15m.csv / _4h.csv # XAUUSD OOS（2025/3〜2026/2）
│   ├── xauusd_1m.csv                # XAUUSD 全期間1分足（409,615行）
│   ├── {symbol}_is_*.csv            # 各銘柄IS（audusd/eurusd/gbpusd/eurjpy/gbpjpy/nas100/spx500/us30等）
│   ├── {symbol}_oos_*.csv           # 各銘柄OOS
│   └── ohlc/                        # 全期間1ファイル（大文字銘柄名, manusが整備）
│       ├── AUDUSD_1m.csv / _15m.csv / _4h.csv
│       ├── EURUSD_1m.csv / _15m.csv / _4h.csv
│       ├── GBPUSD_1m.csv / _15m.csv / _4h.csv
│       ├── NAS100_1m.csv / _15m.csv / _4h.csv
│       ├── SPX500_1m.csv / _15m.csv / _4h.csv
│       ├── US30_1m.csv / _15m.csv / _4h.csv
│       ├── XAUUSD_1m.csv / _15m.csv / _4h.csv
│       ├── USDJPY_1h.csv / _4h.csv  # ※USDJPYに1mなし
│       └── README.md
│
├── scripts/                             # ★ アクティブスクリプトのみ（バックテスト系はarchiveへ移動済）
│   ├── fetch_data.py                    # データ取得（OANDA API）
│   ├── fetch_all_ohlc.py               # 全銘柄OHLC取得
│   ├── fetch_oos_data.py               # OOSデータ取得
│   ├── fetch_fx_robust.py              # FXデータ取得（リトライ付き）
│   ├── generate_htf_from_1m.py         # 1mから上位足生成
│   ├── main_loop.py                    # メインループ
│   └── archive/                        # 旧バックテスト・分析スクリプト全件
│
├── utils/
│   ├── risk_manager.py              # ★ AdaptiveRiskManager / SYMBOL_CONFIG
│   └── position_manager.py
│
├── backtest/
│   └── vectorbt_runner.py           # VectorBT統合
│
├── tradingview/
│   ├── yagami_v77_kmid_klow.pine    # TradingView Pineスクリプト
│   └── README_yagami_v77.md
│
├── trade_logs/                      # ペーパートレード実績ログ
│   ├── paper_state.json
│   ├── performance_log.csv
│   └── ...
│
├── results/                         # バックテスト結果（PNG・CSV）
│   ├── strategy_comparison.png      # 3戦略比較チャート
│   ├── strategy_comparison_summary.csv
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

> **⚠️ 旧バックテストスクリプト（backtest_*.py 等）は `scripts/archive/` に移動済み。**
> 新しい YAGAMI改 用バックテストはここに追記していく。

```bash
cd /home/user/sena3fx  # ← /home/ubuntu/ ではない

# 新バックテスト（YAGAMI改）
# ← ここに追加予定
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
| `/report` | GET | 定時レポートをDiscordへ送信 |
| `/health` | GET | ヘルスチェック |
| `/status` | GET | 現在のオープンポジション一覧 |
| `/notify_test` | GET | Discord通知テスト |

---

## 注意事項

- `strategies/current/yagami_mtf_v79.py` と `cloud_run/strategies/yagami_mtf_v79.py` は**常に同一内容**を保つこと
- `strategies/current/yagami_mtf_v77.py` と `cloud_run/strategies/yagami_mtf_v77.py` は**常に同一内容**を保つこと
- `strategies/current/yagami_mtf_v78.py` と `cloud_run/strategies/yagami_mtf_v78.py` は**常に同一内容**を保つこと
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
