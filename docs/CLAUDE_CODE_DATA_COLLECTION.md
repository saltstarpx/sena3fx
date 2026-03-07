# ローソク足データ収集 — Claude Code 手順書

## 目的

OANDAデモAPIから指定ペアの1分足データ（3ヶ月分）を取得し、  
`/home/ubuntu/sena3fx/data/` に CSV として保存する。

---

## 取得仕様

| 項目 | 値 |
|------|-----|
| API | OANDA fxTrade Practice REST API v3 |
| エンドポイント | `GET /v3/instruments/{instrument}/candles` |
| 認証 | Bearer トークン（下記参照） |
| 取得期間 | **2025-12-01 00:00:00 UTC 〜 2026-02-28 23:59:00 UTC** |
| タイムフレーム | **M1（1分足）** |
| 1リクエスト上限 | **5000本**（OANDAの仕様） |
| 価格タイプ | `M`（Mid価格） |

### 認証情報

```
OANDA_TOKEN  = "b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467"
BASE_URL     = "https://api-fxpractice.oanda.com"
HEADERS      = {"Authorization": "Bearer {OANDA_TOKEN}", "Content-Type": "application/json"}
```

---

## 取得対象ペア（3ペア）

既存データ（`/home/ubuntu/sena3fx/data/`）にない以下3ペアを取得する。

| ペア名 | OANDA instrument | 保存ファイル名 |
|--------|-----------------|--------------|
| CADJPY | `CAD_JPY` | `cadjpy_1m.csv` |
| CHFJPY | `CHF_JPY` | `chfjpy_1m.csv` |
| NZDUSD | `NZD_USD` | `nzdusd_1m.csv` |

> 既存ファイル（eurusd, gbpusd, audusd, usdjpy, eurjpy, gbpjpy, xauusd）は取得不要。

---

## 出力CSVフォーマット

```csv
timestamp,open,high,low,close,volume
2025-12-01 00:00:00,1.03550,1.03560,1.03540,1.03555,12
2025-12-01 00:01:00,1.03555,1.03570,1.03550,1.03565,8
...
```

- `timestamp` はタイムゾーンなし（UTC）の文字列: `YYYY-MM-DD HH:MM:SS`
- `open/high/low/close` は float（小数点以下5桁）
- `volume` は int
- ヘッダー行あり
- 時系列昇順（古い順）

---

## 実装手順

### Step 1: Pythonスクリプトを作成

`/home/ubuntu/sena3fx/scripts/fetch_candles_for_backtest.py` として以下を実装する。

```python
"""
fetch_candles_for_backtest.py
OANDA APIから1分足データを取得してCSVに保存する
"""
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import os

OANDA_TOKEN = "b3c7db048d5b6d1ac77e4263bd8bfb8b-1222fbcaf7d9ffe642692a226f7e7467"
BASE_URL    = "https://api-fxpractice.oanda.com"
HEADERS     = {"Authorization": f"Bearer {OANDA_TOKEN}", "Content-Type": "application/json"}
DATA_DIR    = "/home/ubuntu/sena3fx/data"

# 取得対象
TARGETS = [
    {"pair": "CADJPY", "instrument": "CAD_JPY", "filename": "cadjpy_1m.csv"},
    {"pair": "CHFJPY", "instrument": "CHF_JPY", "filename": "chfjpy_1m.csv"},
    {"pair": "NZDUSD", "instrument": "NZD_USD", "filename": "nzdusd_1m.csv"},
]

START = datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
END   = datetime(2026, 2, 28, 23, 59, 0, tzinfo=timezone.utc)
COUNT_PER_REQ = 5000  # OANDAの1リクエスト上限


def fetch_candles(instrument, from_dt, to_dt):
    """指定期間の1分足を全て取得して DataFrame で返す"""
    all_rows = []
    current = from_dt

    while current < to_dt:
        params = {
            "granularity": "M1",
            "price": "M",
            "from": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": COUNT_PER_REQ,
        }
        try:
            r = requests.get(
                f"{BASE_URL}/v3/instruments/{instrument}/candles",
                headers=HEADERS, params=params, timeout=30
            )
            if r.status_code != 200:
                print(f"  ERROR {r.status_code}: {r.text[:100]}")
                break

            candles = r.json().get("candles", [])
            if not candles:
                break

            for c in candles:
                if not c.get("complete", True):
                    continue
                ts = c["time"][:19].replace("T", " ")  # "2025-12-01T00:00:00"
                m  = c["mid"]
                all_rows.append({
                    "timestamp": ts,
                    "open":  float(m["o"]),
                    "high":  float(m["h"]),
                    "low":   float(m["l"]),
                    "close": float(m["c"]),
                    "volume": int(c.get("volume", 0)),
                })

            # 次のリクエストの開始時刻
            last_ts = candles[-1]["time"]
            last_dt = datetime.strptime(last_ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            current = last_dt + timedelta(minutes=1)

            print(f"  取得中: {last_dt.strftime('%Y-%m-%d %H:%M')} | 累計{len(all_rows)}本", end="\r")
            time.sleep(0.3)  # レート制限対策

        except Exception as e:
            print(f"  例外: {e}")
            time.sleep(5)
            continue

    print()
    return pd.DataFrame(all_rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for target in TARGETS:
        pair       = target["pair"]
        instrument = target["instrument"]
        filename   = target["filename"]
        out_path   = os.path.join(DATA_DIR, filename)

        print(f"\n[{pair}] 取得開始: {START.date()} → {END.date()}")

        df = fetch_candles(instrument, START, END)

        if df.empty:
            print(f"[{pair}] データなし、スキップ")
            continue

        # 重複除去・ソート
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        df.to_csv(out_path, index=False)
        print(f"[{pair}] 保存完了: {len(df)}行 → {out_path}")
        print(f"  期間: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
```

### Step 2: スクリプトを実行

```bash
cd /home/ubuntu/sena3fx
python3.11 scripts/fetch_candles_for_backtest.py
```

所要時間の目安: **各ペア約3〜5分**（合計10〜15分）

### Step 3: 取得結果を確認

```bash
for f in cadjpy_1m.csv chfjpy_1m.csv nzdusd_1m.csv; do
  echo "$f: $(wc -l < /home/ubuntu/sena3fx/data/$f)行"
  head -2 /home/ubuntu/sena3fx/data/$f
done
```

期待値: 各ペア **約 120,000〜130,000 行**（3ヶ月 × 1440本/日）

### Step 4: GitHubにコミット

```bash
cd /home/ubuntu/sena3fx
git add data/cadjpy_1m.csv data/chfjpy_1m.csv data/nzdusd_1m.csv
git commit -m "data: add 3-month 1m candles for CADJPY/CHFJPY/NZDUSD"
git push origin main
```

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| `401 Unauthorized` | `OANDA_TOKEN` が正しいか確認 |
| `400 Bad Request` | `from` パラメータのフォーマットを確認（`Z`サフィックス必須） |
| データが途中で止まる | スクリプトを再実行（既存ファイルがあれば上書きされる） |
| `volume` が全部0 | 正常（週末・祝日はvolume=0になることがある） |

---

## 完了後の連絡

取得完了したら Manus 側に「データ取得完了」と伝えてください。  
Manus がバックテストスクリプトに追加して即座に実行します。

---

*作成: sena3fx プロジェクト | 2026-03-07*
