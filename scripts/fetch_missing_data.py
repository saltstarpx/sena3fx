"""
不足データの取得スクリプト
===========================
1. XAGUSD 1m: OANDA API から IS/OOS 期間を取得
2. 1h 不足銘柄（NAS100, US30, SPX500）: 15m から作成してキャッシュ

使い方:
    export OANDA_API_KEY="your_token"
    python3 scripts/fetch_missing_data.py

OANDA_API_KEY が設定されていない場合も動作します（15mフォールバック）。
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

IS_START  = datetime(2024, 7, 1, tzinfo=timezone.utc)
IS_END    = datetime(2025, 3, 31, 23, 59, tzinfo=timezone.utc)
OOS_START = datetime(2025, 4, 1, tzinfo=timezone.utc)
OOS_END   = datetime(2026, 2, 27, 23, 59, tzinfo=timezone.utc)

# ── OANDA API ────────────────────────────────────────────────────────────────
def fetch_oanda_chunk(instrument, granularity, start_dt, end_dt, api_key,
                      account_type="practice", max_count=5000):
    """OANDA v20 から1チャンク（最大5000本）取得"""
    import json
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError, URLError

    host = ("api-fxpractice.oanda.com" if account_type == "practice"
            else "api-fxtrade.oanda.com")

    from_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    to_str   = end_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    url = (f"https://{host}/v3/instruments/{instrument}/candles"
           f"?granularity={granularity}&price=M&from={from_str}&to={to_str}"
           f"&count={max_count}")

    try:
        req = Request(url, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        with urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  HTTP {e.code}: {body[:200]}")
        return None
    except URLError as e:
        print(f"  接続エラー: {e.reason}")
        return None

    candles = raw.get("candles", [])
    if not candles:
        return None

    rows = []
    for c in candles:
        if not c.get("complete", True):
            continue
        ts = datetime.strptime(c["time"][:19], "%Y-%m-%dT%H:%M:%S")
        ts = ts.replace(tzinfo=timezone.utc)
        if "mid" in c:
            o  = float(c["mid"]["o"])
            h  = float(c["mid"]["h"])
            l  = float(c["mid"]["l"])
            cl = float(c["mid"]["c"])
        elif "bid" in c and "ask" in c:
            o  = (float(c["bid"]["o"]) + float(c["ask"]["o"])) / 2
            h  = (float(c["bid"]["h"]) + float(c["ask"]["h"])) / 2
            l  = (float(c["bid"]["l"]) + float(c["ask"]["l"])) / 2
            cl = (float(c["bid"]["c"]) + float(c["ask"]["c"])) / 2
        else:
            continue
        rows.append({"timestamp": ts, "open": o, "high": h, "low": l,
                     "close": cl, "volume": int(c.get("volume", 0))})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def fetch_oanda_period(instrument, granularity, start_dt, end_dt,
                       api_key, account_type="practice", chunk_days=3):
    """
    OANDA APIから期間を分割して全データ取得する。
    M1は1日あたり1440本なのでchunk_days=3で約4320本/リクエスト。
    """
    bars_per_day = {
        "M1": 1440, "M5": 288, "M15": 96,
        "M30": 48, "H1": 24, "H4": 6, "D": 1
    }.get(granularity, 24)

    max_per_chunk = 4500  # 5000 未満で余裕を持たせる
    chunk_duration = timedelta(days=max(1, max_per_chunk // bars_per_day))

    all_dfs = []
    cursor  = start_dt
    total   = 0

    print(f"  {instrument} {granularity}: {start_dt.date()} 〜 {end_dt.date()} を取得中...")

    while cursor < end_dt:
        chunk_end = min(cursor + chunk_duration, end_dt)
        df = fetch_oanda_chunk(instrument, granularity, cursor, chunk_end,
                                api_key, account_type)
        if df is None or len(df) == 0:
            # リトライ（バックオフ）
            time.sleep(2)
            df = fetch_oanda_chunk(instrument, granularity, cursor, chunk_end,
                                    api_key, account_type)
            if df is None or len(df) == 0:
                print(f"  警告: {cursor.date()} 〜 {chunk_end.date()} のデータ取得失敗")
                cursor = chunk_end + timedelta(seconds=1)
                continue

        all_dfs.append(df)
        total += len(df)
        cursor = df.index[-1] + timedelta(seconds=1)

        if len(all_dfs) % 10 == 0:
            print(f"    進捗: {cursor.date()} | 累計 {total:,} 本")

        time.sleep(0.3)  # レートリミット対策

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    print(f"  完了: {len(combined):,} 本")
    return combined


def load_csv_utc(path):
    """CSV をロードして UTC DatetimeIndex に統一"""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.rename(columns={ts_col: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "close"])


def save_split_csv(df, sym, period, tf, start, end):
    """IS/OOS スプリットCSVとして保存"""
    sliced = df[(df.index >= start) & (df.index <= end)].copy()
    if len(sliced) == 0:
        print(f"  警告: {sym} {period} {tf} - スライス後データ空")
        return False
    path = os.path.join(DATA_DIR, f"{sym}_{period}_{tf}.csv")
    sliced.to_csv(path)
    print(f"  保存: {path} ({len(sliced):,} 行)")
    return True


# ── メイン ──────────────────────────────────────────────────────────────────
def main():
    api_key      = os.environ.get("OANDA_API_KEY")
    account_type = os.environ.get("OANDA_ACCOUNT", "practice")

    print("=" * 65)
    print("不足データ取得・作成スクリプト")
    print("=" * 65)
    if api_key:
        print(f"OANDA APIキー: 設定済み (account={account_type})")
    else:
        print("OANDA APIキー: 未設定 → API取得をスキップ（15mフォールバック使用）")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. XAGUSD 1m: 完全欠如 → OANDA APIから取得
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1] XAGUSD 1m データ確認...")
    xag_1m_is_path  = os.path.join(DATA_DIR, "xagusd_is_1m.csv")
    xag_1m_oos_path = os.path.join(DATA_DIR, "xagusd_oos_1m.csv")

    xag_is_ok  = os.path.exists(xag_1m_is_path)
    xag_oos_ok = os.path.exists(xag_1m_oos_path)

    if xag_is_ok and xag_oos_ok:
        print("  XAGUSD IS/OOS 1m: 既存ファイルあり → スキップ")
    elif api_key:
        print("  XAGUSD 1m が不足 → OANDA API から取得します")
        periods_to_fetch = []
        if not xag_is_ok:
            periods_to_fetch.append(("is", IS_START, IS_END))
        if not xag_oos_ok:
            periods_to_fetch.append(("oos", OOS_START, OOS_END))

        for period_name, p_start, p_end in periods_to_fetch:
            print(f"\n  XAGUSD {period_name.upper()} 1m 取得中...")
            df = fetch_oanda_period(
                "XAG_USD", "M1", p_start, p_end,
                api_key=api_key, account_type=account_type
            )
            if df is not None and len(df) > 0:
                save_split_csv(df, "xagusd", period_name, "1m", p_start, p_end)
            else:
                print(f"  XAGUSD {period_name.upper()} 1m: 取得失敗 → 15mフォールバック使用")
    else:
        print("  APIキー未設定 → XAGUSD は 15m データでシミュレーション")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. NAS100 IS/OOS 15m: 全期間ファイルをスライスしてキャッシュ
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2] NAS100 IS/OOS 15m 作成...")
    nas_15m_full = load_csv_utc(os.path.join(DATA_DIR, "nas100_15m.csv"))
    if nas_15m_full is not None:
        for period_name, p_start, p_end in [
            ("is", IS_START, IS_END), ("oos", OOS_START, OOS_END)
        ]:
            path = os.path.join(DATA_DIR, f"nas100_{period_name}_15m.csv")
            if not os.path.exists(path):
                save_split_csv(nas_15m_full, "nas100", period_name, "15m",
                                p_start, p_end)
            else:
                print(f"  nas100_{period_name}_15m.csv: 既存 → スキップ")
    else:
        print("  nas100_15m.csv が見つかりません")
        if api_key:
            print("  OANDA API から NAS100 15m を取得します...")
            for period_name, p_start, p_end in [
                ("is", IS_START, IS_END), ("oos", OOS_START, OOS_END)
            ]:
                df = fetch_oanda_period(
                    "NAS100_USD", "M15", p_start, p_end,
                    api_key=api_key, account_type=account_type
                )
                if df is not None:
                    save_split_csv(df, "nas100", period_name, "15m", p_start, p_end)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. NAS100 IS/OOS 1m: 全期間ファイルをスライスしてキャッシュ
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3] NAS100 IS/OOS 1m 作成...")
    nas_1m_full = load_csv_utc(os.path.join(DATA_DIR, "nas100_1m.csv"))
    if nas_1m_full is not None:
        for period_name, p_start, p_end in [
            ("is", IS_START, IS_END), ("oos", OOS_START, OOS_END)
        ]:
            path = os.path.join(DATA_DIR, f"nas100_{period_name}_1m.csv")
            if not os.path.exists(path):
                save_split_csv(nas_1m_full, "nas100", period_name, "1m", p_start, p_end)
            else:
                print(f"  nas100_{period_name}_1m.csv: 既存 → スキップ")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. US30 IS/OOS 1m: 全期間ファイルをスライスしてキャッシュ
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4] US30 IS/OOS 1m 作成...")
    us30_1m_full = load_csv_utc(os.path.join(DATA_DIR, "us30_1m.csv"))
    if us30_1m_full is not None:
        for period_name, p_start, p_end in [
            ("is", IS_START, IS_END), ("oos", OOS_START, OOS_END)
        ]:
            path = os.path.join(DATA_DIR, f"us30_{period_name}_1m.csv")
            if not os.path.exists(path):
                save_split_csv(us30_1m_full, "us30", period_name, "1m", p_start, p_end)
            else:
                print(f"  us30_{period_name}_1m.csv: 既存 → スキップ")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. SPX500 IS/OOS 1m: 全期間ファイルをスライスしてキャッシュ
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5] SPX500 IS/OOS 1m 作成...")
    spx_1m_full = load_csv_utc(os.path.join(DATA_DIR, "spx500_1m.csv"))
    if spx_1m_full is not None:
        for period_name, p_start, p_end in [
            ("is", IS_START, IS_END), ("oos", OOS_START, OOS_END)
        ]:
            path = os.path.join(DATA_DIR, f"spx500_{period_name}_1m.csv")
            if not os.path.exists(path):
                save_split_csv(spx_1m_full, "spx500", period_name, "1m", p_start, p_end)
            else:
                print(f"  spx500_{period_name}_1m.csv: 既存 → スキップ")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. 1h 相当データ作成: 15m → 1h リサンプル（NAS100, US30, SPX500）
    #    generate_signals_f1f3 が resample("1h") するため 15m を渡せば十分だが
    #    念のため 1h キャッシュも作成してロード高速化
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[6] 1h キャッシュ作成（15m → 1h リサンプル）...")
    for sym, full_sym in [("nas100", "NAS100"), ("us30", "US30"), ("spx500", "SPX500")]:
        for period_name, p_start, p_end in [
            ("is", IS_START, IS_END), ("oos", OOS_START, OOS_END)
        ]:
            out_path = os.path.join(DATA_DIR, f"{sym}_{period_name}_1h.csv")
            if os.path.exists(out_path):
                print(f"  {sym}_{period_name}_1h.csv: 既存 → スキップ")
                continue

            # IS/OOS 15m ファイルから作成
            src_path = os.path.join(DATA_DIR, f"{sym}_{period_name}_15m.csv")
            if not os.path.exists(src_path):
                # 全期間ファイルからスライス
                src_path = os.path.join(DATA_DIR, f"{sym}_15m.csv")

            src = load_csv_utc(src_path) if os.path.exists(src_path) else None
            if src is None:
                print(f"  {sym}_{period_name}_1h.csv: 元データなし → スキップ")
                continue

            sliced = src[(src.index >= p_start) & (src.index <= p_end)]
            vol_col = "volume" if "volume" in sliced.columns else None
            agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
            if vol_col:
                agg["volume"] = "sum"
            resampled = sliced.resample("1h").agg(agg).dropna(subset=["open", "close"])
            resampled.to_csv(out_path)
            print(f"  保存: {out_path} ({len(resampled):,} 行, 15m→1h リサンプル)")

    print("\n完了: データ準備終了")
    print("次のステップ: python3 scripts/backtest_is_oos_metals_indices.py")


if __name__ == "__main__":
    main()
