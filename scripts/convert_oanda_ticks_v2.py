"""
OANDAティックデータ -> OHLC変換スクリプト（月次処理版）
月ごとにOHLCを生成してから結合することでメモリを節約する
"""
import os
import gc
import glob
import pandas as pd

TICK_DIR = "/home/ubuntu/sena3fx/data/oanda_ticks"
OUT_DIR  = "/home/ubuntu/sena3fx/data"
PAIR     = "usdjpy_oos"

TIMEFRAMES = {
    "1m":  "1min",
    "15m": "15min",
    "1h":  "1h",
    "4h":  "4h",
}

files = sorted(glob.glob(f"{TICK_DIR}/ticks_USDJPY-oj5k_*.csv"))
print(f"対象ファイル数: {len(files)}")

monthly_ohlc = {tf: [] for tf in TIMEFRAMES}

for f in files:
    fname = os.path.basename(f)
    print(f"\n処理中: {fname}", flush=True)

    chunks = []
    for chunk in pd.read_csv(
        f,
        sep="\t",
        names=["date", "time", "bid", "ask", "last", "volume"],
        skiprows=1,
        dtype={"bid": float, "ask": float},
        chunksize=500_000,
    ):
        chunk["timestamp"] = pd.to_datetime(
            chunk["date"] + " " + chunk["time"],
            format="%Y.%m.%d %H:%M:%S.%f",
            errors="coerce",
            utc=True,
        )
        chunk = chunk.dropna(subset=["timestamp", "bid", "ask"])
        chunk["mid"] = (chunk["bid"] + chunk["ask"]) / 2
        chunks.append(chunk[["timestamp", "mid"]].set_index("timestamp"))

    df_month = pd.concat(chunks).sort_index()
    del chunks
    gc.collect()

    print(f"  ティック数: {len(df_month):,}")

    for tf_key, rule in TIMEFRAMES.items():
        ohlc = df_month["mid"].resample(rule).ohlc()
        ohlc.columns = ["open", "high", "low", "close"]
        ohlc["volume"] = df_month["mid"].resample(rule).count()
        ohlc = ohlc.dropna(subset=["open"])
        monthly_ohlc[tf_key].append(ohlc)
        print(f"  {tf_key}: {len(ohlc)}本", flush=True)

    del df_month
    gc.collect()

print("\n結合・保存中...")
for tf_key in TIMEFRAMES:
    combined = pd.concat(monthly_ohlc[tf_key]).sort_index()
    combined = combined["2025-03-01":"2026-02-28"]
    # 週末（土曜=5, 日曜=6）を除去（FX取引時間に合わせる）
    combined = combined[combined.index.dayofweek < 5]
    combined.index.name = "timestamp"
    out_path = f"{OUT_DIR}/{PAIR}_{tf_key}.csv"
    combined.to_csv(out_path)
    print(f"  {tf_key}: {len(combined)}行 -> {out_path}")

print("\n変換完了")
