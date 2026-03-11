"""
1分足 Ask+Bid → 中値OHLCV変換スクリプト
入力: data/tick_1min/{SYMBOL}_1Min_{Ask|Bid}_2024.01.01_2026.02.28.csv.gz
出力: data/ohlc/{SYMBOL}_1m.csv (LFSなし)
タイムゾーン: EET(UTC+2) → UTC変換
"""
import gzip
import pandas as pd
from pathlib import Path

SYMBOLS = ["AUDJPY", "CADJPY", "EURGBP", "EURJPY", "GBPJPY", "XAGUSD"]
REPO = Path(__file__).parent.parent
TICK_DIR = REPO / "data" / "tick_1min"
OHLC_DIR = REPO / "data" / "ohlc"
OHLC_DIR.mkdir(parents=True, exist_ok=True)

# EETはUTC+2（夏時間なし固定でForex Testerは通常EET=UTC+2）
EET_OFFSET = pd.Timedelta(hours=2)

for sym in SYMBOLS:
    ask_path = TICK_DIR / f"{sym}_1Min_Ask_2024.01.01_2026.02.28.csv.gz"
    bid_path = TICK_DIR / f"{sym}_1Min_Bid_2024.01.01_2026.02.28.csv.gz"

    print(f"Processing {sym}...")

    ask = pd.read_csv(ask_path, compression="gzip",
                      names=["ts","open","high","low","close","volume"],
                      header=0, dtype={"volume": float})
    bid = pd.read_csv(bid_path, compression="gzip",
                      names=["ts","open","high","low","close","volume"],
                      header=0, dtype={"volume": float})

    # タイムスタンプ変換: EET → UTC (naive)
    ask["ts"] = pd.to_datetime(ask["ts"], format="%Y.%m.%d %H:%M:%S") - EET_OFFSET
    bid["ts"] = pd.to_datetime(bid["ts"], format="%Y.%m.%d %H:%M:%S") - EET_OFFSET

    ask = ask.set_index("ts")
    bid = bid.set_index("ts")

    # 共通インデックスで中値計算
    idx = ask.index.union(bid.index)
    ask = ask.reindex(idx)
    bid = bid.reindex(idx)

    mid = pd.DataFrame(index=idx)
    mid["open"]   = (ask["open"]  + bid["open"])  / 2
    mid["high"]   = (ask["high"]  + bid["high"])  / 2
    mid["low"]    = (ask["low"]   + bid["low"])   / 2
    mid["close"]  = (ask["close"] + bid["close"]) / 2
    mid["volume"] = ask["volume"].fillna(0) + bid["volume"].fillna(0)

    # NaN行（片方しかないバー）は削除
    mid = mid.dropna(subset=["open","high","low","close"])

    mid.index.name = "timestamp"
    mid = mid.reset_index()
    mid["timestamp"] = mid["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    out_path = OHLC_DIR / f"{sym}_1m.csv"
    mid.to_csv(out_path, index=False, float_format="%.5f")
    print(f"  -> {out_path} ({len(mid):,} rows)")

print("Done.")
