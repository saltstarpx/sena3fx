"""
OANDAダウンロードティックデータ（タブ区切りCSV）を
OHLC形式（1分足・15分足・4時間足）に変換してバックテスト用CSVとして保存する。

入力フォーマット（タブ区切り）:
  <DATE>  <TIME>  <BID>  <ASK>  <LAST>  <VOLUME>
  2025.01.31  16:59:59.564  44884.8  44887.3  (空)  (空)

価格: (BID + ASK) / 2 のミッドポイントを使用
タイムスタンプ: UTC（OANDAティックデータはUTC基準）

対象銘柄:
  US30  → us30_is_*.csv / us30_oos_*.csv
  US500 → spx500_is_*.csv / spx500_oos_*.csv  (既存ファイル名に合わせる)

IS期間:  2024-07-01 〜 2025-02-28
OOS期間: 2025-03-03 〜 2026-02-27
"""
import os
import sys
import zipfile
from io import StringIO

import pandas as pd

# ── 設定 ──────────────────────────────────────────────
UPLOAD_DIR = "/home/ubuntu/upload"
DATA_DIR   = "/home/ubuntu/sena3fx/data"
os.makedirs(DATA_DIR, exist_ok=True)

IS_START  = pd.Timestamp("2024-07-01", tz="UTC")
IS_END    = pd.Timestamp("2025-03-01", tz="UTC")   # 2025-02-28末まで
OOS_START = pd.Timestamp("2025-03-03", tz="UTC")
OOS_END   = pd.Timestamp("2026-02-28", tz="UTC")

# ZIPファイル名 → (銘柄prefix, OANDAファイル名prefix)
INSTRUMENTS = {
    "US30":  "us30",
    "US500": "spx500",
}

TIMEFRAMES = {
    "1m":  "1min",
    "15m": "15min",
    "4h":  "240min",
}


# ── ティックデータ読み込み ─────────────────────────────
def load_ticks_from_zip(zip_path: str) -> pd.DataFrame:
    """ZIPからティックCSVを読み込んでDataFrameを返す"""
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        csv_names = [n for n in names if n.endswith(".csv")]
        if not csv_names:
            print(f"  [WARN] CSVなし: {zip_path}")
            return pd.DataFrame()
        csv_name = csv_names[0]
        with z.open(csv_name) as f:
            content = f.read().decode("utf-8", errors="replace")

    df = pd.read_csv(
        StringIO(content),
        sep="\t",
        header=0,
        names=["date", "time", "bid", "ask", "last", "volume"],
        dtype={"bid": float, "ask": float},
        na_values=["", " "],
        on_bad_lines="skip",
    )

    # BID/ASKが両方有効な行のみ使用
    df = df.dropna(subset=["bid", "ask"])
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]

    # ミッドポイント価格
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # タイムスタンプ変換: "2025.01.31" + "16:59:59.564" → UTC
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y.%m.%d %H:%M:%S.%f",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df[["timestamp", "mid"]]


# ── OHLC集約 ──────────────────────────────────────────
def resample_to_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """ミッドポイント価格をOHLCにリサンプリングする"""
    df = df.set_index("timestamp")
    ohlc = df["mid"].resample(rule).agg(
        open="first",
        high="max",
        low="min",
        close="last",
    )
    vol = df["mid"].resample(rule).count().rename("volume")
    ohlc = ohlc.join(vol).dropna(subset=["open"])
    ohlc = ohlc[ohlc["volume"] > 0]
    ohlc = ohlc.reset_index()
    ohlc["volume"] = ohlc["volume"].astype(int)
    return ohlc


# ── 品質チェック ───────────────────────────────────────
def quality_check(df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"  [{label}] 品質チェック ({len(df):,}行)")

    dup = df["timestamp"].duplicated().sum()
    if dup > 0:
        print(f"    [修正] タイムスタンプ重複 {dup}件 → 除去")
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

    df = df.sort_values("timestamp").reset_index(drop=True)

    ohlc_errors = (
        (df["high"] < df["low"]).sum() +
        (df["open"] > df["high"]).sum() +
        (df["open"] < df["low"]).sum() +
        (df["close"] > df["high"]).sum() +
        (df["close"] < df["low"]).sum()
    )
    if ohlc_errors > 0:
        print(f"    [ERROR] OHLC整合性エラー: {ohlc_errors}件")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex) or len(df.columns) > 7:
        print(f"    [ERROR] カラム数異常: {len(df.columns)}個")
        sys.exit(1)

    print(f"    [OK] {len(df):,}行 | {df['timestamp'].min()} 〜 {df['timestamp'].max()}")
    return df


# ── メイン ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ティックデータ → OHLC変換スクリプト")
    print("=" * 60)

    for instrument_key, file_prefix in INSTRUMENTS.items():
        print(f"\n{'='*60}")
        print(f"銘柄: {instrument_key} → {file_prefix}")
        print(f"{'='*60}")

        # 対象ZIPを収集してソート
        zip_files = sorted([
            os.path.join(UPLOAD_DIR, f)
            for f in os.listdir(UPLOAD_DIR)
            if f.startswith(f"ticks_{instrument_key}_") and f.endswith(".zip")
        ])

        if not zip_files:
            print(f"  [WARN] ZIPファイルが見つかりません: {instrument_key}")
            continue

        print(f"  対象ZIPファイル: {len(zip_files)}個")

        # 全ZIPを読み込んで結合
        all_ticks = []
        for zf in zip_files:
            print(f"  読み込み中: {os.path.basename(zf)}")
            ticks = load_ticks_from_zip(zf)
            if not ticks.empty:
                all_ticks.append(ticks)
                print(f"    → {len(ticks):,}ティック")

        if not all_ticks:
            print(f"  [ERROR] データなし: {instrument_key}")
            continue

        df_all = pd.concat(all_ticks, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["timestamp"], keep="first")
        df_all = df_all.sort_values("timestamp").reset_index(drop=True)
        print(f"\n  全期間合計: {len(df_all):,}ティック")
        print(f"  期間: {df_all['timestamp'].min()} 〜 {df_all['timestamp'].max()}")

        # IS / OOS に分割
        periods = {
            "is":  (IS_START,  IS_END),
            "oos": (OOS_START, OOS_END),
        }

        for period_label, (p_start, p_end) in periods.items():
            mask = (df_all["timestamp"] >= p_start) & (df_all["timestamp"] < p_end)
            df_period = df_all[mask].copy()

            if df_period.empty:
                print(f"\n  [{period_label.upper()}] データなし（期間外）")
                continue

            print(f"\n  [{period_label.upper()}] {len(df_period):,}ティック | {df_period['timestamp'].min()} 〜 {df_period['timestamp'].max()}")

            # 各時間軸にリサンプリング
            for tf_label, rule in TIMEFRAMES.items():
                ohlc = resample_to_ohlc(df_period, rule)
                ohlc = quality_check(ohlc, f"{file_prefix.upper()}-{period_label.upper()}-{tf_label}")

                fname = f"{file_prefix}_{period_label}_{tf_label}.csv"
                out   = os.path.join(DATA_DIR, fname)
                ohlc.to_csv(out, index=False)
                print(f"    保存完了: {fname} ({len(ohlc):,}行)")

    print("\n" + "=" * 60)
    print("全変換・保存完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
