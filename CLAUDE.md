# CLAUDE.md

sena3fx — やがみメソッド MTF自動取引エージェント

## プロジェクト概要

やがみメソッドに基づくMTF（マルチタイムフレーム）二番底・二番天井戦略の自動化システム。
Cloud Run上でペーパートレードを24時間稼働させ、PDCAサイクルで継続的に戦略を改善する。

- **現在の戦略バージョン**: **v78**（2026/3/9 正式採用、XAUUSD専用改善）
- **メイン銘柄**: USDJPY（バックテスト基準）/ XAUUSD（唯一の全基準PASS）
- **対象銘柄**: USDJPY / EURJPY / GBPJPY / XAUUSD / SPX500 / US30 / NAS100 他
- **データソース**: OANDA API（リアルタイム）/ ローカルCSV（バックテスト用）
- **ブランチ**: `main`
- **リモート**: GitHub `saltstarpx/sena3fx`
- **作業ディレクトリ**: `/home/user/sena3fx`（`/home/ubuntu/` ではない）

---

## 現在の戦略: v78（XAUUSD専用）/ v77（その他銘柄）

### 概要

4時間足EMA20によるトレンドフィルター + 1時間足の二番底・二番天井パターン検出。
エントリーは足更新後2分以内の1分足始値（成行）。半利確あり（1R到達でポジション半分決済・SLをBEへ）。

- **XAUUSD**: `yagami_mtf_v78.py`（4改善適用、utc_end=21）
- **USDJPY / その他FX**: `yagami_mtf_v77.py`（改善C・Dで逆効果のため据え置き）

### v78の追加フィルター（v77からの変更点）

```python
PATTERN_TOLERANCE = 0.2      # [改善D] 二番底/天井の許容幅 ATR×0.2（v77: ×0.3）
BODY_MIN_FACTOR   = 0.2      # [改善C] 確認足の実体サイズ最小値 ATR×0.2
UTC_START_DEFAULT = 5        # [改善B] 時間帯フィルター 開始（UTC）
UTC_END_DEFAULT   = 21       # [改善B] 時間帯フィルター 終了（UTC, XAUUSD基準）
```

| 改善 | 内容 | 効果（XAUUSD OOS） |
|------|------|------|
| **改善A** | 1H確認足にもKLOWチェックを追加 | 実装のみ（単体では微減） |
| **改善B** | 時間帯フィルター UTC5〜21時 | PF 2.00→2.20, MDD 15.6%→12.5% |
| **改善C** | 確認足の実体サイズ ≥ ATR×0.2 | PF 2.20→2.37, MDD 12.5%→10.6% |
| **改善D** | パターン許容幅を ATR×0.3→ATR×0.2 に縮小 | PF 2.37→2.50, MDD 10.6%→6.9% |

### v78バックテスト結果（XAUUSD OOS: 2025/03〜2026/02, spread=5.2pips）

| 指標 | v77 | **v78D（全改善）** |
|------|:---:|:---:|
| トレード数（OOS） | 555件 | **180件**（-68%） |
| PF | 2.03 | **2.50** (+23%) |
| MDD | 15.6% | **6.9%** (-56%) |
| Kelly | -0.012 | **+0.043**（正転） |
| 月次プラス | 12/12 | **12/12** |

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
v78改善はXAUUSD専用（FX通貨ペアではC・Dが逆効果）。

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
│   │   ├── yagami_mtf_v78.py        # ★ 現行戦略 v78（XAUUSD専用、4改善適用）
│   │   ├── yagami_mtf_v77.py        # 現行戦略 v77（USDJPY等、v78非適用銘柄）
│   │   └── yagami_mtf_v76_improved.py
│   └── archive/                     # v1〜v76（履歴）
│
├── cloud_run/                       # Cloud Run本番コード
│   ├── main.py                      # FastAPI エンドポイント（v77使用）
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── gcp-key.json                 # GCPキー（.gitignore対象）
│   └── strategies/
│       ├── yagami_mtf_v78.py        # 本番用v78（current/と同一、XAUUSD用）
│       ├── yagami_mtf_v77.py        # 本番用v77（current/と同一）
│       └── archive/
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
├── scripts/
│   ├── backtest_strategy_comparison.py  # ★ v77/F1+F3/Hybrid 3戦略比較（最新）
│   ├── backtest_is_oos_f1f3.py          # F1+F3フィルター IS/OOS比較
│   ├── backtest_7sym_vmax.py            # 7銘柄バックテスト
│   ├── backtest_symbol_selection.py     # 銘柄選定バックテスト
│   ├── backtest_v78_improvements.py     # ★ v78改善段階検証（v77→v78A/B/C/D）
│   ├── backtest_v77_correct.py          # v77正式バックテスト
│   ├── backtest_v77_bugfix.py           # v77バグ修正検証（修正前後比較）
│   ├── fetch_data.py / fetch_all_ohlc.py / fetch_oos_data.py  # データ取得
│   ├── generate_htf_from_1m.py          # 1mから上位足生成
│   └── ...（その他分析スクリプト多数）
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

```bash
cd /home/user/sena3fx  # ← /home/ubuntu/ ではない

# 3戦略比較（v77/F1+F3/Hybrid × 全銘柄）
python3.11 scripts/backtest_strategy_comparison.py

# F1+F3フィルター IS/OOS比較
python3.11 scripts/backtest_is_oos_f1f3.py

# 7銘柄バックテスト
python3.11 scripts/backtest_7sym_vmax.py
```

### バックテスト実行コマンド追加

```bash
# v78改善段階検証（XAUUSD OOS で v77→v78A/B/C/D を比較）
python3.11 scripts/backtest_v78_improvements.py

# v77バグ修正前後比較（Bug①②の影響を確認）
python3.11 scripts/backtest_v77_bugfix.py
```

### バックテスト合格基準（v77）

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

- `strategies/current/yagami_mtf_v78.py` と `cloud_run/strategies/yagami_mtf_v78.py` は**常に同一内容**を保つこと
- `strategies/current/yagami_mtf_v77.py` と `cloud_run/strategies/yagami_mtf_v77.py` は**常に同一内容**を保つこと
- `data/` のCSVファイルはバックテスト専用。ペーパートレードはOANDA APIからリアルタイム取得
- `gcp-key.json` は `.gitignore` 対象。コミットしないこと
- **USDJPYに1分足データは存在しない**（15m/1h/4hのみ）。バックテストは15mで代用
- `data/ohlc/` の銘柄名は大文字（`AUDUSD_1m.csv`）、`data/` の銘柄名は小文字（`audusd_1m.csv`）
- やがみPDF教材（`docs/learning_materials/`）が戦略のソースオブトゥルース

---

## バージョン履歴

| バージョン | 採用日 | 主な変更 | PF（OOS） |
|---|---|---|:---:|
| **v78** | 2026/3/9 | XAUUSD専用4改善（時間帯・実体サイズ・許容幅・1HKLOW） | **2.50**（XAUUSD） |
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
| **XAUUSD** | **v78** | **2.50** | 6.9% | +0.043 | 12/12 |

次の候補: SPX500（v77, PF=2.96 ← あと0.04でPASS）

### v78推奨設定（銘柄別）

| 銘柄 | バージョン | utc_end | 理由 |
|---|---|:---:|---|
| XAUUSD | **v78** | 21 | 全4改善でPF2.50達成 |
| GBPUSD | v77 | — | v78B(utc_end=20)単体でPF1.95、採用基準未達 |
| AUDUSD / EURUSD | v77 | — | v78C・DでPF低下、v77据え置き |
| 指数（US30/SPX500/NAS100） | v77 | — | v77・v78共にPF<1.0、採用不可 |
