"""
OANDAティックデータ → OHLC変換スクリプト
形式: <DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>
midprice = (bid + ask) / 2 でOHLCを生成
"""
import os
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

# 全ファイルを読み込んで結合
files = sorted(glob.glob(f"{TICK_DIR}/ticks_USDJPY-oj5k_*.csv"))
print(f"対象ファイル数: {len(files)}")

dfs = []
for f in files:
    print(f"  読み込み中: {os.path.basename(f)}", flush=True)
    df = pd.read_csv(
        f,
        sep="\t",
        names=["date", "time", "bid", "ask", "last", "volume"],
        skiprows=1,
        dtype={"bid": float, "ask": float},
    )
    # タイムスタンプ生成（OANDAはサーバー時間=UTC+0 or EST?）
    # ファイルの日付を確認: 2025.02.28 17:00:00.045 → MT5はUTCベース
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y.%m.%d %H:%M:%S.%f",
        errors="coerce",
        utc=True,
    )
    df = df.dropna(subset=["timestamp", "bid", "ask"])
    df["mid"] = (df["bid"] + df["ask"]) / 2
    dfs.append(df[["timestamp", "mid", "bid", "ask"]])

all_ticks = pd.concat(dfs, ignore_index=True)
all_ticks = all_ticks.sort_values("timestamp").drop_duplicates("timestamp")
all_ticks = all_ticks.set_index("timestamp")

# OOS期間のみ（2025年3月〜2026年2月）
all_ticks = all_ticks["2025-03-01":"2026-02-28"]
print(f"\n総ティック数: {len(all_ticks):,}")
print(f"期間: {all_ticks.index[0]} 〜 {all_ticks.index[-1]}")

# 各時間足にリサンプリング
for tf_key, rule in TIMEFRAMES.items():
    out_path = f"{OUT_DIR}/{PAIR}_{tf_key}.csv"
    print(f"\nリサンプリング: {tf_key} ({rule})", flush=True)
    
    ohlc = all_ticks["mid"].resample(rule).ohlc()
    ohlc.columns = ["open", "high", "low", "close"]
    
    # volume（ティック数）
    ohlc["volume"] = all_ticks["mid"].resample(rule).count()
    
    # 欠損行（週末・祝日）を除去
    ohlc = ohlc.dropna(subset=["open"])
    
    ohlc.index.name = "timestamp"
    ohlc.to_csv(out_path)
    print(f"  保存: {len(ohlc)}行 → {out_path}")

print("\n変換完了")
