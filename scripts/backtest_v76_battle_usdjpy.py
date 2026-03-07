"""
v76 vs v76改 USDJPY精密対決バックテスト
=========================================
データ: OANDA正規 15m/4h (IS+OOS結合)
期間: 2025/1/1 ~ 2025/12/31
スプレッド: 0.4 pips
資金: 300万円
半利確ロジック: あり（参照: backtest_full_oanda.py）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)

SPREAD = 0.4
INIT_CAPITAL = 3_000_000
START = "2025-01-01"
END = "2025-12-31"


def load(path):
    """OANDAデータ読み込み"""
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def run_backtest(data_15m, data_4h, gen_signals, spread=0.4, rr_ratio=2.5, label=""):
    """
    15分足バーごとにシミュレーション。半利確ロジック付き。
    generate_signals の data_1m 引数には data_15m を渡す（1mデータなしのため）。
    """
    print(f"\n  {label} シグナル生成中...", flush=True)
    signals = gen_signals(data_15m, data_15m, data_4h,
                          spread_pips=spread, rr_ratio=rr_ratio)
    sig_map = {s["time"]: s for s in signals}
    print(f"    シグナル数: {len(signals)}")

    trades = []
    pos = None

    start_ts = pd.Timestamp(START, tz="UTC")
    end_ts = pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=7)  # 年末ポジ決済猶予

    for i in range(len(data_15m)):
        bar = data_15m.iloc[i]
        t = bar.name

        if t > end_ts:
            # 年末超え: 強制決済
            if pos is not None:
                close_pnl = (bar["close"] - pos["ep"]) * 100 * pos["dir"]
                total = pos.get("half_pnl", 0) + close_pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": t,
                    "dir": pos["dir"], "pnl": total,
                    "result": "win" if total > 0 else "loss",
                    "exit_type": "FORCED",
                    "month": pos["entry_time"].strftime("%Y-%m"),
                })
                pos = None
            break

        # ── ポジション管理 ──
        if pos is not None:
            d = pos["dir"]
            raw_ep = pos["ep"] - pos["spread"] * d
            half_tp = raw_ep + pos["risk"] * d

            # 半利確チェック
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"] = raw_ep  # SLをBEへ
                    pos["half_closed"] = True

            # SLチェック
            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                sl_pnl = (pos["sl"] - pos["ep"]) * 100 * d
                total = pos.get("half_pnl", 0) + sl_pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": t,
                    "dir": d, "pnl": total,
                    "result": "win" if total > 0 else "loss",
                    "exit_type": "SL" if not pos["half_closed"] else "HALF+SL",
                    "month": pos["entry_time"].strftime("%Y-%m"),
                })
                pos = None
                continue

            # TPチェック
            if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                tp_pnl = (pos["tp"] - pos["ep"]) * 100 * d
                total = pos.get("half_pnl", 0) + tp_pnl
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": t,
                    "dir": d, "pnl": total,
                    "result": "win" if total > 0 else "loss",
                    "exit_type": "TP" if not pos["half_closed"] else "HALF+TP",
                    "month": pos["entry_time"].strftime("%Y-%m"),
                })
                pos = None
                continue

        # ── エントリー（期間内のみ） ──
        if pos is None and t in sig_map:
            entry_time = t
            if entry_time >= start_ts and entry_time <= pd.Timestamp(END, tz="UTC"):
                pos = {**sig_map[t], "entry_time": entry_time, "half_closed": False}

    df_trades = pd.DataFrame(trades)
    print(f"    トレード数: {len(df_trades)}")
    return df_trades


def calc_stats(df, label):
    """統計指標を計算・表示"""
    if df.empty:
        print(f"  {label}: トレードなし")
        return {}

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")
    wr = len(wins) / len(df) * 100
    avg_w = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_l = losses["pnl"].mean() if len(losses) > 0 else 0
    kelly = wr / 100 - (1 - wr / 100) / (abs(avg_w) / abs(avg_l)) if avg_l != 0 else 0

    # t検定
    t_stat, p_val = stats.ttest_1samp(df["pnl"], 0) if len(df) >= 2 else (0, 1)

    # 月次
    monthly = df.groupby("month")["pnl"].sum()
    plus_months = (monthly > 0).sum()
    total_months = len(monthly)

    # MDD (pips)
    cumsum = df["pnl"].cumsum()
    peak = cumsum.cummax()
    mdd = (peak - cumsum).max()

    # 月次シャープ
    m_mean = monthly.mean()
    m_std = monthly.std()
    sharpe = (m_mean / m_std * np.sqrt(12)) if m_std > 0 else 0

    # 円換算（1ロット=10万通貨、USDJPY: 1pip=0.01→1pip=1000円）
    pnl_jpy = df["pnl"].sum() * 1000  # pips × 1000円/pip
    ret_pct = pnl_jpy / INIT_CAPITAL * 100

    return {
        "label": label,
        "trades": len(df),
        "wins": len(wins),
        "losses": len(losses),
        "wr": wr,
        "pf": pf,
        "total_pnl": df["pnl"].sum(),
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "kelly": kelly,
        "t_stat": t_stat,
        "p_value": p_val,
        "plus_months": f"{plus_months}/{total_months}",
        "mdd_pips": mdd,
        "sharpe": sharpe,
        "pnl_jpy": pnl_jpy,
        "ret_pct": ret_pct,
        "monthly": monthly,
    }


def print_comparison(s_v76, s_imp):
    """2戦略比較テーブル"""
    print(f"\n{'='*72}")
    print(f"  USDJPY v76 vs v76改 精密対決結果")
    print(f"  期間: {START} ~ {END} / スプレッド: {SPREAD}pips / 資金: {INIT_CAPITAL:,}円")
    print(f"{'='*72}")
    print(f"  {'指標':<18s} {'v76':>18s} {'v76改':>18s} {'勝者':>8s}")
    print(f"  {'─'*62}")

    rows = [
        ("トレード数", "trades", "d", None),
        ("勝率 (%)", "wr", ".1f", True),
        ("PF", "pf", ".3f", True),
        ("総損益 (pips)", "total_pnl", ".1f", True),
        ("平均利益 (pips)", "avg_win", ".1f", True),
        ("平均損失 (pips)", "avg_loss", ".1f", False),  # 負値だがabs小さい方が良い
        ("ケリー基準", "kelly", ".3f", True),
        ("MDD (pips)", "mdd_pips", ".1f", False),
        ("月次シャープ", "sharpe", ".3f", True),
        ("t検定 p値", "p_value", ".4f", False),
        ("プラス月", "plus_months", "s", None),
        ("損益 (円)", "pnl_jpy", ",.0f", True),
        ("リターン (%)", "ret_pct", ".2f", True),
    ]

    score = {"v76": 0, "imp": 0}
    for label, key, fmt, higher_better in rows:
        v = s_v76.get(key, 0)
        i = s_imp.get(key, 0)

        if higher_better is not None:
            if key == "avg_loss":
                # 損失: abs()が小さい方が良い
                winner = "imp" if abs(i) < abs(v) else "v76"
            elif higher_better:
                winner = "imp" if i > v else "v76"
            else:
                winner = "imp" if i < v else "v76"
            score[winner] += 1
            mark = "◀" if winner == "v76" else "▶"
        else:
            mark = ""

        if fmt == "s":
            print(f"  {label:<18s} {str(v):>18s} {str(i):>18s} {mark:>8s}")
        else:
            print(f"  {label:<18s} {v:>18{fmt}} {i:>18{fmt}} {mark:>8s}")

    print(f"\n  {'─'*62}")
    print(f"  スコア: v76 {score['v76']} - {score['imp']} v76改")
    if score["imp"] > score["v76"]:
        print(f"  >>> v76改 の勝利! ({score['imp']}-{score['v76']})")
    elif score["v76"] > score["imp"]:
        print(f"  >>> v76 の勝利! ({score['v76']}-{score['imp']})")
    else:
        print(f"  >>> 引き分け! ({score['v76']}-{score['imp']})")

    return score


def main():
    from strategies.current.yagami_mtf_v76 import generate_signals as gen_v76
    from strategies.current.yagami_mtf_v76_improved import generate_signals as gen_improved

    # ── データ読み込み（IS+OOSを結合して2025年分を抽出） ──
    print("=" * 72)
    print("  v76 vs v76改 USDJPY精密対決バックテスト")
    print(f"  期間: {START} ~ {END}")
    print(f"  スプレッド: {SPREAD}pips / 資金: {INIT_CAPITAL:,}円")
    print(f"  半利確ロジック: ON")
    print("=" * 72)

    print("\n  データ読み込み中...")
    is_15m = load(f"{DATA}/usdjpy_is_15m.csv")
    is_4h = load(f"{DATA}/usdjpy_is_4h.csv")
    oos_15m = load(f"{DATA}/usdjpy_oos_15m.csv")
    oos_4h = load(f"{DATA}/usdjpy_oos_4h.csv")

    # IS+OOS結合
    data_15m = pd.concat([is_15m, oos_15m]).sort_index()
    data_15m = data_15m[~data_15m.index.duplicated(keep="first")]
    data_4h = pd.concat([is_4h, oos_4h]).sort_index()
    data_4h = data_4h[~data_4h.index.duplicated(keep="first")]

    print(f"  15m: {len(data_15m)}本 ({data_15m.index[0]} ~ {data_15m.index[-1]})")
    print(f"  4h:  {len(data_4h)}本 ({data_4h.index[0]} ~ {data_4h.index[-1]})")

    # ── v76 バックテスト ──
    df_v76 = run_backtest(data_15m, data_4h, gen_v76,
                          spread=SPREAD, rr_ratio=2.5, label="v76")
    s_v76 = calc_stats(df_v76, "v76")

    # ── v76改 バックテスト ──
    df_imp = run_backtest(data_15m, data_4h, gen_improved,
                          spread=SPREAD, rr_ratio=3.0, label="v76改")
    s_imp = calc_stats(df_imp, "v76改")

    # ── 比較 ──
    score = print_comparison(s_v76, s_imp)

    # ── 月次詳細 ──
    if "monthly" in s_v76 and "monthly" in s_imp:
        print(f"\n  月次損益 (pips)")
        print(f"  {'月':>10s} {'v76':>12s} {'v76改':>12s}")
        print(f"  {'─'*36}")
        all_months = sorted(set(list(s_v76["monthly"].index) + list(s_imp["monthly"].index)))
        for m in all_months:
            if not m.startswith("2025"):
                continue
            v = s_v76["monthly"].get(m, 0)
            i = s_imp["monthly"].get(m, 0)
            print(f"  {m:>10s} {v:>+12.1f} {i:>+12.1f}")

    # ── 決済タイプ分析 ──
    print(f"\n  決済タイプ内訳")
    for label, df in [("v76", df_v76), ("v76改", df_imp)]:
        if df.empty:
            continue
        print(f"  {label}:")
        for et, cnt in df["exit_type"].value_counts().items():
            avg = df[df["exit_type"] == et]["pnl"].mean()
            print(f"    {et:<10s}: {cnt:>4d}回  平均{avg:>+.1f}pips")

    # ── CSV保存 ──
    df_v76.to_csv(f"{RESULTS}/v76_battle_usdjpy_v76_trades.csv", index=False)
    df_imp.to_csv(f"{RESULTS}/v76_battle_usdjpy_improved_trades.csv", index=False)

    # サマリーCSV
    summary = pd.DataFrame([
        {k: v for k, v in s_v76.items() if k != "monthly"},
        {k: v for k, v in s_imp.items() if k != "monthly"},
    ])
    summary.to_csv(f"{RESULTS}/v76_battle_usdjpy_summary.csv", index=False)
    print(f"\n  結果保存: {RESULTS}/v76_battle_usdjpy_*.csv")


if __name__ == "__main__":
    main()
