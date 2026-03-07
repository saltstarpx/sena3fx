# Claude Code向け バックテストスクリプト作成指示書

> **重要**: このドキュメントはClaude Codeがバックテストスクリプトを作成・修正する際に必ず参照すること。
> データファイルは**すべて存在する**。「データがない」「代替データで構成する」などの判断は誤りであるため、必ずこのドキュメントのパスを使用すること。

---

## 1. データファイル一覧（確認済み・全ファイル実在）

### メインバックテスト用データ（`/home/ubuntu/sena3fx/data/`）

| 変数名 | ファイルパス | 行数 | 期間 | カラム |
|--------|-------------|------|------|--------|
| `is_1m` | `/home/ubuntu/sena3fx/data/usdjpy_is_1m.csv` | 248,731行 | 2024-07-01 〜 2025-02-28 | timestamp, open, high, low, close, volume |
| `is_15m` | `/home/ubuntu/sena3fx/data/usdjpy_is_15m.csv` | 16,629行 | 2024-07-01 〜 2025-02-28 | timestamp, open, high, low, close, volume |
| `is_4h` | `/home/ubuntu/sena3fx/data/usdjpy_is_4h.csv` | 1,042行 | 2024-07-01 〜 2025-02-28 | timestamp, open, high, low, close, volume |
| `oos_1m` | `/home/ubuntu/sena3fx/data/usdjpy_oos_1m.csv` | 371,480行 | 2025-03-03 〜 2026-02-27 | timestamp, open, high, low, close, volume |
| `oos_15m` | `/home/ubuntu/sena3fx/data/usdjpy_oos_15m.csv` | 24,814行 | 2025-03-03 〜 2026-02-27 | timestamp, open, high, low, close, volume |
| `oos_4h` | `/home/ubuntu/sena3fx/data/usdjpy_oos_4h.csv` | 1,553行 | 2025-03-03 〜 2026-02-27 | timestamp, open, high, low, close, volume |

### 他通貨ペアデータ（`/home/ubuntu/sena3fx/data/`）

| 変数名 | ファイルパス | 行数 | 期間 | カラム |
|--------|-------------|------|------|--------|
| `eurjpy_4h` | `/home/ubuntu/sena3fx/data/eurjpy_4h.csv` | 1,805行 | 2025-01-01 〜 2026-02-27 | timestamp, open, high, low, close, volume |
| `gbpjpy_4h` | `/home/ubuntu/sena3fx/data/gbpjpy_4h.csv` | 1,805行 | 2025-01-01 〜 2026-02-27 | timestamp, open, high, low, close, volume |

### 最新データ（2026年分 / `/home/ubuntu/sena3fx/data/ohlc/`）

| ファイルパス | 行数 | 期間 | カラム |
|-------------|------|------|--------|
| `/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Q1.csv` | 82,076行 | 2026-01-01 〜 2026-02-27 | timestamp, open, high, low, close |
| `/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Jan.csv` | — | 2026年1月分 | timestamp, open, high, low, close |
| `/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Feb.csv` | — | 2026年2月分 | timestamp, open, high, low, close |
| `/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m.csv` | 6,649行 | 2026-02-26 〜 2026-03-05 | timestamp, open, high, low, close, spread, tick_count |

---

## 2. データ読み込みの正しい実装

```python
import pandas as pd

DATA = "/home/ubuntu/sena3fx/data"

def load(path):
    """OANDAデータ読み込み共通関数。タイムスタンプはUTC aware。"""
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

# IS期間（インサンプル: 2024/7〜2025/2）
is_1m  = load(f"{DATA}/usdjpy_is_1m.csv")    # 1分足
is_15m = load(f"{DATA}/usdjpy_is_15m.csv")   # 15分足
is_4h  = load(f"{DATA}/usdjpy_is_4h.csv")    # 4時間足

# OOS期間（アウトオブサンプル: 2025/3〜2026/2）
oos_1m  = load(f"{DATA}/usdjpy_oos_1m.csv")  # 1分足
oos_15m = load(f"{DATA}/usdjpy_oos_15m.csv") # 15分足
oos_4h  = load(f"{DATA}/usdjpy_oos_4h.csv")  # 4時間足
```

> **注意**: `pd.read_csv` の `parse_dates=["timestamp"]` と `pd.to_datetime(..., utc=True)` を必ず使用すること。タイムゾーンを統一しないとデータの結合・比較が正しく動作しない。

---

## 3. 戦略ファイルのパス

```python
import sys
sys.path.insert(0, "/home/ubuntu/sena3fx/strategies/current")
import yagami_mtf_v76 as strategy

# または直接インポート
from strategies.current.yagami_mtf_v76 import generate_signals
```

| ファイル | パス |
|---------|------|
| 現行戦略（v76） | `/home/ubuntu/sena3fx/strategies/current/yagami_mtf_v76.py` |
| Cloud Run用（本番） | `/home/ubuntu/sena3fx/cloud_run/strategies/yagami_mtf_v76.py` |
| 過去バージョン | `/home/ubuntu/sena3fx/strategies/archive/yagami_mtf_v*.py` |

---

## 4. バックテストエンジンの正しい実装

v76の `generate_signals()` が返すシグナル辞書の構造：

```python
{
    "time":    pd.Timestamp,   # エントリー時刻（UTC aware）
    "dir":     int,            # 1=ロング, -1=ショート
    "ep":      float,          # 実際の約定価格（スプレッド込み）
    "sl":      float,          # ストップロスレベル（チャートレベル）
    "tp":      float,          # テイクプロフィットレベル（チャートレベル）
    "risk":    float,          # リスク幅（チャートレベル、スプレッドなし）
    "spread":  float,          # スプレッド（参照用）
    "tf":      str,            # "4h" or "1h"
    "pattern": str,            # "double_bottom" or "double_top"
}
```

### 半利確ロジック（必須）

```python
# 半利確ライン = チャートレベルのEP + risk（スプレッドなし）
raw_ep = pos["ep"] - pos["spread"] * pos["dir"]
half_tp = raw_ep + pos["risk"] * pos["dir"]

# 半利確到達時: 半分決済 → SLをBEへ移動
if not pos["half_closed"]:
    if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
        pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
        pos["sl"] = raw_ep  # SLをBE（ブレークイーブン）へ
        pos["half_closed"] = True
```

### 損益計算（pips単位）

```python
# SL到達時の損益
sl_pnl = (pos["sl"] - pos["ep"]) * 100 * pos["dir"]

# TP到達時の損益
tp_pnl = (pos["tp"] - pos["ep"]) * 100 * pos["dir"]

# 半利確あり場合の合計
total_pnl = pos.get("half_pnl", 0) + sl_pnl  # or tp_pnl
```

---

## 5. 出力先ディレクトリ

```python
RESULTS = "/home/ubuntu/sena3fx/results"
# 例: f"{RESULTS}/v76_is_oos_backtest.png"
# 例: f"{RESULTS}/v76_is_oos_summary.csv"
```

---

## 6. 既存の参照スクリプト

以下のスクリプトが既に動作確認済みのため、参考にすること：

| スクリプト | 内容 |
|-----------|------|
| `/home/ubuntu/sena3fx/scripts/backtest_full_oanda.py` | IS+OOS統合バックテスト（v76） |
| `/home/ubuntu/sena3fx/scripts/compare_v75_v76.py` | v75 vs v76 スプレッド別比較 |
| `/home/ubuntu/sena3fx/scripts/analyze_bug_impact.py` | データバグ修正の影響分析 |
| `/home/ubuntu/sena3fx/scripts/check_ohlc_quality.py` | OHLCデータ品質チェック |

---

## 7. よくある誤りと対処法

| 誤り | 正しい対処 |
|------|-----------|
| 「1分足・15分足データがない」 | `data/usdjpy_is_1m.csv` 等が存在する。このドキュメントのパスを使用すること |
| 「1h→15分代替で構成」 | 不要。15分足は `usdjpy_is_15m.csv` / `usdjpy_oos_15m.csv` に実在する |
| タイムゾーンエラー | `pd.to_datetime(..., utc=True)` を必ず使用する |
| `strategies/yagami_mtf_v76` が見つからない | `sys.path.insert(0, ".../strategies/current")` を追加する |
| カラム名エラー | 全ファイル共通: `timestamp, open, high, low, close, volume`（volumeなしのファイルもある） |

---

## 8. 標準的なバックテスト統計指標

新しいバックテストスクリプトでは以下の指標を必ず出力すること：

- **トレード数**・**勝率**・**PF（プロフィットファクター）**
- **総損益（pips）**・**平均利益**・**平均損失**
- **ケリー基準**（`勝率 - (1-勝率) / (平均利益/平均損失)`）
- **最大ドローダウン（pips）**
- **月次シャープレシオ**（`月次平均損益 / 月次標準偏差 × √12`）
- **t検定 p値**（`scipy.stats.ttest_1samp(pnl_list, 0)`）
- **プラス月数 / 総月数**

---

*最終更新: 2026-03-07*
*作成者: Manus AI（sena3fxプロジェクト管理）*
