# CLAUDE.md

sena3fx — USDJPY自律型自動取引エージェント

## プロジェクト概要

USDJPYを対象とした、やがみメソッドに基づくMTF（マルチタイムフレーム）二番底・二番天井戦略の自動化システム。
Cloud Run上でペーパートレードを24時間稼働させ、PDCAサイクルで継続的に戦略を改善する。

- **現在の戦略バージョン**: **v77**（2026/3/7 正式採用）
- **対象通貨ペア**: USDJPY / EURJPY / GBPJPY（メイン: USDJPY）
- **データソース**: OANDA API（リアルタイム）/ ローカルCSV（バックテスト用）
- **ブランチ**: `main`
- **リモート**: GitHub `saltstarpx/sena3fx`

---

## 現在の戦略: v77

### 概要

4時間足EMA20によるトレンドフィルター + 1時間足/4時間足の二番底・二番天井パターン検出。
エントリーは足更新後2分以内の1分足始値（成行）。半利確あり（1R到達でポジション半分決済・SLをBEへ）。

### v77の追加フィルター（v76からの変更点）

**KMID（実体方向一致）+ KLOW（下ヒゲ小）フィルター**

```python
# ロングなら直前4H足が陽線、ショートなら陰線
kmid_ok = (dir == 1 and prev_4h.close > prev_4h.open) or \
          (dir == -1 and prev_4h.close < prev_4h.open)

# 下ヒゲが小さい（モメンタム純度が高い）
klow_ok = (min(prev_4h.open, prev_4h.close) - prev_4h.low) / prev_4h.open < 0.0015
```

- **KMID**: 逆方向の4H足でのエントリー（勝率32〜38%の養分トレード183本）を除外
- **KLOW**: 下ヒゲが大きい足（買い圧力が混在）でのエントリーを除外
- **根拠**: qlib Alpha158系57ファクタースクリーニングで最高IC（情報係数）を記録

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

### 過学習検証（5段階、全PASS）

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
├── CLAUDE.md                    # AI開発ガイド（このファイル）
├── README.md
├── .gitignore
│
├── strategies/
│   ├── current/
│   │   └── yagami_mtf_v77.py   # ★ 現行戦略（本番・バックテスト共通）
│   └── archive/
│       ├── yagami_mtf_v76.py   # v76（スプレッド計算修正版）
│       ├── yagami_mtf_v75.py   # v75（スプレッドバグあり）
│       └── ...                  # v1〜v74（学習データ）
│
├── cloud_run/                   # Cloud Run本番コード
│   ├── main.py                  # FastAPI エンドポイント（v77使用）
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── gcp-key.json             # GCPキー（.gitignore対象）
│   └── strategies/
│       ├── yagami_mtf_v77.py   # 本番用v77（current/と同一）
│       └── archive/
│           └── yagami_mtf_v76.py
│
├── data/                        # バックテスト用CSVデータ（品質チェック済み）
│   ├── usdjpy_is_1m.csv         # IS 1分足（2024/7〜2025/2, 248,731行）
│   ├── usdjpy_is_15m.csv        # IS 15分足（2024/7〜2025/2, 16,629行）
│   ├── usdjpy_is_4h.csv         # IS 4時間足（2024/7〜2025/2, 1,042行）
│   ├── usdjpy_oos_1m.csv        # OOS 1分足（2025/3〜2026/2, 371,480行）
│   ├── usdjpy_oos_15m.csv       # OOS 15分足（2025/3〜2026/2, 24,814行）
│   ├── usdjpy_oos_4h.csv        # OOS 4時間足（2025/3〜2026/2, 1,553行）
│   └── archive/                 # 旧版・バグあり（USDJPY_15m.csv等）
│
├── scripts/
│   ├── backtest_full_oanda.py   # v76/v77バックテストエンジン
│   ├── backtest_v76_full_period.py # IS+OOS全期間統合バックテスト
│   ├── check_ohlc_quality.py    # データ品質チェック（定期実行推奨）
│   ├── fix_ohlc_data.py         # データ修正スクリプト
│   ├── compare_v75_v76.py       # v75 vs v76 比較
│   ├── detailed_analysis_v76.py # v76詳細分析
│   └── fetch_data.py            # OANDAデータ取得
│
├── docs/
│   ├── v76_backtest_analysis_report.md  # v76詳細分析レポート
│   ├── strategy_development_log_v60_v76.md # 開発議事録
│   ├── data_quality_bugs.md     # データバグ記録（27カラム・月末重複）
│   ├── bug_impact_analysis_report.md   # バグ影響分析
│   ├── prompts/                 # Claude Code向けプロンプト
│   └── learning_materials/      # やがみ氏PDF教材7冊
│
└── results/                     # バックテスト結果（PNG・CSV）
```

---

## データ仕様（重要）

バックテストスクリプトでは必ず以下のパスを使用すること。
**「データがない」「代替データで構成」という判断は誤り。全ファイルが存在する。**

| ファイル | 行数 | 期間 | カラム |
|---|:---:|---|---|
| `data/usdjpy_is_1m.csv` | 248,731 | 2024/7/1〜2025/2/28 | timestamp,open,high,low,close,volume |
| `data/usdjpy_is_15m.csv` | 16,629 | 2024/7/1〜2025/2/28 | 同上 |
| `data/usdjpy_is_4h.csv` | 1,042 | 2024/7/1〜2025/2/28 | 同上 |
| `data/usdjpy_oos_1m.csv` | 371,480 | 2025/3/3〜2026/2/27 | 同上 |
| `data/usdjpy_oos_15m.csv` | 24,814 | 2025/3/3〜2026/2/27 | 同上 |
| `data/usdjpy_oos_4h.csv` | 1,553 | 2025/3/3〜2026/2/27 | 同上 |

タイムゾーン: UTC（`pd.read_csv(..., parse_dates=['timestamp'], index_col='timestamp')` 後に `tz_localize('UTC')` が必要な場合あり）

---

## やがみメソッド（戦略の根拠）

### 基本構造

1. **4時間足でトレンド方向を確認**（EMA20との位置関係）
2. **1時間足/4時間足で二番底・二番天井パターンを検出**
3. **1分足でエントリー**（足更新後2分以内の始値）

### v77のエントリー条件（全条件AND）

| 条件 | 内容 |
|------|------|
| トレンド一致 | 4H EMA20より上→ロングのみ、下→ショートのみ |
| パターン成立 | 二番底（ロング）/ 二番天井（ショート）の安値/高値が ATR×0.3 以内 |
| 確認足 | パターン形成後の足が方向一致の実体（陽線/陰線） |
| **KMID** | **直前4H足の実体方向がエントリー方向と一致** |
| **KLOW** | **直前4H足の下ヒゲ比率 < 0.15%** |
| リスク幅 | SLまでの距離が ATR×3（4H）または ATR×2（1H）以内 |

### SL/TP設定

- **SL**: 二番底/天井の安値・高値から ATR×0.15 外側
- **TP**: SLリスク幅 × 2.5倍（チャートレベル基準）
- **半利確**: 1R到達でポジション半分決済 → SLをBEへ移動

---

## バックテスト実行方法

```bash
cd /home/ubuntu/sena3fx

# IS+OOS全期間統合バックテスト（v76/v77対応）
python3.11 scripts/backtest_v76_full_period.py

# データ品質チェック（新規データ追加後は必ず実行）
python3.11 scripts/check_ohlc_quality.py
```

### バックテスト合格基準（v77）

| 指標 | 基準 |
|------|------|
| プロフィットファクター | ≥ 3.0 |
| 勝率 | ≥ 65% |
| 最大ドローダウン | ≤ 300pips |
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

- `strategies/current/yagami_mtf_v77.py` と `cloud_run/strategies/yagami_mtf_v77.py` は**常に同一内容**を保つこと
- `data/` のCSVファイルはバックテスト専用。ペーパートレードはOANDA APIからリアルタイム取得
- `gcp-key.json` は `.gitignore` 対象。コミットしないこと
- 新規データ追加後は必ず `scripts/check_ohlc_quality.py` を実行すること
- やがみPDF教材（`docs/learning_materials/`）が戦略のソースオブトゥルース

---

## バージョン履歴

| バージョン | 採用日 | 主な変更 | PF（OOS） |
|---|---|---|:---:|
| **v77** | 2026/3/7 | KMID+KLOWフィルター追加（qlib Alpha158系） | **4.96** |
| v76 | 2026/2 | スプレッド計算バグ修正（チャートレベル固定） | 2.39 |
| v75 | 2026/1 | スプレッド計算バグあり（非推奨） | — |
| v60〜v74 | 2025 | 各種パラメータ最適化 | — |

詳細: `docs/strategy_development_log_v60_v76.md`
