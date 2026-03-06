"""
EURJPY・GBPJPY OHLC変換 + v76バックテスト 一括スクリプト
期間: 2025年1月〜2026年2月（14ヶ月）
スプレッド: EURJPY=1.1pips, GBPJPY=1.5pips（OANDAの実スプレッド参考値）
"""
import sys, os, gc, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "/home/ubuntu/sena3fx/strategies")
import yagami_mtf_v76 as v76

DATA    = "/home/ubuntu/sena3fx/data"
RESULTS = "/home/ubuntu/sena3fx/results"
TIMEFRAMES = {"1m": "1min", "15m": "15min", "4h": "4h"}

# ---- OHLC変換 ----
def convert_ticks(tick_dir, pair_key, start, end, spread_label):
    files = sorted(glob.glob(f"{tick_dir}/*.csv"))
    print(f"\n=== {pair_key} OHLC変換 ({len(files)}ファイル) ===")
    monthly = {tf: [] for tf in TIMEFRAMES}

    for f in files:
        print(f"  処理中: {os.path.basename(f)}", flush=True)
        chunks = []
        for chunk in pd.read_csv(
            f, sep="\t",
            names=["date", "time", "bid", "ask", "last", "volume"],
            skiprows=1, dtype={"bid": float, "ask": float},
            chunksize=500_000,
        ):
            chunk["timestamp"] = pd.to_datetime(
                chunk["date"] + " " + chunk["time"],
                format="%Y.%m.%d %H:%M:%S.%f", errors="coerce", utc=True)
            chunk = chunk.dropna(subset=["timestamp", "bid", "ask"])
            chunk["mid"] = (chunk["bid"] + chunk["ask"]) / 2
            chunks.append(chunk[["timestamp", "mid"]].set_index("timestamp"))
        df_m = pd.concat(chunks).sort_index()
        del chunks; gc.collect()

        for tf_key, rule in TIMEFRAMES.items():
            ohlc = df_m["mid"].resample(rule).ohlc()
            ohlc.columns = ["open", "high", "low", "close"]
            ohlc["volume"] = df_m["mid"].resample(rule).count()
            ohlc = ohlc.dropna(subset=["open"])
            ohlc = ohlc[ohlc.index.dayofweek < 5]
            monthly[tf_key].append(ohlc)
        del df_m; gc.collect()

    result = {}
    for tf_key in TIMEFRAMES:
        combined = pd.concat(monthly[tf_key]).sort_index()
        combined = combined[start:end]
        combined.index.name = "timestamp"
        out_path = f"{DATA}/{pair_key}_{tf_key}.csv"
        combined.to_csv(out_path)
        result[tf_key] = combined
        print(f"  {tf_key}: {len(combined)}行 -> {out_path}")
    return result

# ---- バックテストエンジン ----
def run_backtest(data_1m, data_15m, data_4h, spread, label, pair):
    print(f"\n{label} シグナル生成中...", flush=True)
    signals = v76.generate_signals(data_1m, data_15m, data_4h, spread_pips=spread)
    sig_map = {s["time"]: s for s in signals}
    print(f"  シグナル数: {len(signals)}")

    trades = []
    pos = None
    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t   = bar.name
        if pos is not None:
            d      = pos["dir"]
            raw_ep = pos["ep"] - pos["spread"] * d
            half_tp = raw_ep + pos["risk"] * d
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"] = raw_ep
                    pos["half_closed"] = True
            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                sl_pnl = (pos["sl"] - pos["ep"]) * 100 * d
                total  = pos.get("half_pnl", 0) + sl_pnl
                trades.append({"entry_time": pos["entry_time"], "exit_time": t,
                    "dir": d, "pnl": total, "result": "win" if total > 0 else "loss",
                    "exit_type": "SL" if not pos["half_closed"] else "HALF+SL",
                    "month": pos["entry_time"].strftime("%Y-%m"), "pair": pair})
                pos = None; continue
            if (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                tp_pnl = (pos["tp"] - pos["ep"]) * 100 * d
                total  = pos.get("half_pnl", 0) + tp_pnl
                trades.append({"entry_time": pos["entry_time"], "exit_time": t,
                    "dir": d, "pnl": total, "result": "win" if total > 0 else "loss",
                    "exit_type": "TP" if not pos["half_closed"] else "HALF+TP",
                    "month": pos["entry_time"].strftime("%Y-%m"), "pair": pair})
                pos = None; continue
        if pos is None and t in sig_map:
            pos = {**sig_map[t], "entry_time": t, "half_closed": False}

    df_t = pd.DataFrame(trades)
    print(f"  トレード数: {len(df_t)}")
    return df_t

def calc_stats(df, label):
    if df.empty:
        print(f"{label}: トレードなし"); return {}
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
    wr     = len(wins) / len(df) * 100
    avg_w  = wins["pnl"].mean()   if len(wins)   > 0 else 0
    avg_l  = losses["pnl"].mean() if len(losses) > 0 else 0
    kelly  = wr/100 - (1 - wr/100) / (abs(avg_w) / abs(avg_l)) if avg_l != 0 else 0
    t_stat, p_val = stats.ttest_1samp(df["pnl"], 0)
    monthly      = df.groupby("month")["pnl"].sum()
    plus_months  = (monthly > 0).sum()
    total_months = len(monthly)
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"  トレード数:  {len(df)}回")
    print(f"  勝率:        {wr:.1f}%")
    print(f"  PF:          {pf:.2f}")
    print(f"  総損益:      {df['pnl'].sum():+.1f}pips")
    print(f"  平均利益:    {avg_w:+.1f}pips")
    print(f"  平均損失:    {avg_l:+.1f}pips")
    print(f"  ケリー基準:  {kelly:.3f}")
    print(f"  t統計量:     {t_stat:.4f}")
    print(f"  p値:         {p_val:.4f}  {'★p<0.05 統計的有意' if p_val < 0.05 else '(有意差なし)'}")
    print(f"  プラス月:    {plus_months}/{total_months}ヶ月")
    return dict(label=label, trades=len(df), win_rate=wr, pf=pf,
                total_pnl=df["pnl"].sum(), avg_win=avg_w, avg_loss=avg_l,
                kelly=kelly, t_stat=t_stat, p_value=p_val,
                plus_months=f"{plus_months}/{total_months}", monthly=monthly)

# ---- EURJPY ----
eur_data = convert_ticks(
    "/home/ubuntu/sena3fx/data/oanda_ticks_eurjpy",
    "eurjpy", "2025-01-01", "2026-02-28", "EURJPY")
# EURJPYスプレッド: 1.1pips
df_eur = run_backtest(eur_data["1m"], eur_data["15m"], eur_data["4h"],
                      spread=1.1, label="EURJPY", pair="EURJPY")

# ---- GBPJPY ----
gbp_data = convert_ticks(
    "/home/ubuntu/sena3fx/data/oanda_ticks_gbpjpy",
    "gbpjpy", "2025-01-01", "2026-02-28", "GBPJPY")
# GBPJPYスプレッド: 1.5pips
df_gbp = run_backtest(gbp_data["1m"], gbp_data["15m"], gbp_data["4h"],
                      spread=1.5, label="GBPJPY", pair="GBPJPY")

# ---- 各ペア統計 ----
s_eur = calc_stats(df_eur, "EURJPY (2025年1月〜2026年2月)")
s_gbp = calc_stats(df_gbp, "GBPJPY (2025年1月〜2026年2月)")

# ---- USDJPY読み込み（既存） ----
df_usdjpy = pd.read_csv(f"{RESULTS}/v76_all_oanda_trades.csv", parse_dates=["entry_time"])
df_usdjpy["pair"] = "USDJPY"
s_usd = calc_stats(df_usdjpy, "USDJPY (2024年7月〜2026年2月)")

# ---- 全ペア統合 ----
df_all3 = pd.concat([df_usdjpy, df_eur, df_gbp], ignore_index=True)
s_all3 = calc_stats(df_all3, "全ペア統合 (USDJPY+EURJPY+GBPJPY)")

# ---- チャート生成 ----
plt.rcParams["font.family"] = "Noto Sans CJK JP"
fig, axes = plt.subplots(4, 1, figsize=(14, 22))
fig.patch.set_facecolor("#0d0d1a")

def bar_chart(ax, monthly, title):
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=45, fontsize=8)
    ax.axhline(0, color="white", linewidth=0.6)
    ax.set_title(title, fontsize=10, color="white")
    ax.set_ylabel("損益 (pips)", color="white")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")

if "monthly" in s_eur:
    bar_chart(axes[0], s_eur["monthly"],
              f"EURJPY月別損益  {s_eur['trades']}回 / PF {s_eur['pf']:.2f} / 勝率{s_eur['win_rate']:.1f}% / p={s_eur['p_value']:.4f}")
if "monthly" in s_gbp:
    bar_chart(axes[1], s_gbp["monthly"],
              f"GBPJPY月別損益  {s_gbp['trades']}回 / PF {s_gbp['pf']:.2f} / 勝率{s_gbp['win_rate']:.1f}% / p={s_gbp['p_value']:.4f}")

# USDJPYエクイティカーブ
df_usd_s = df_usdjpy.sort_values("entry_time")
axes[2].plot(range(len(df_usd_s)), df_usd_s["pnl"].cumsum().values, color="#3498db", linewidth=1.2, label="USDJPY")
axes[2].set_title(f"USDJPY エクイティカーブ  {s_usd['trades']}回 / PF {s_usd['pf']:.2f} / p={s_usd['p_value']:.4f}", fontsize=10, color="white")
axes[2].set_facecolor("#1a1a2e"); axes[2].tick_params(colors="white")
axes[2].set_ylabel("累積損益 (pips)", color="white")
axes[2].axhline(0, color="white", linewidth=0.5, linestyle="--")

# 全ペア統合エクイティカーブ
df_all3_s = df_all3.sort_values("entry_time")
for pair, color in [("USDJPY", "#3498db"), ("EURJPY", "#2ecc71"), ("GBPJPY", "#e74c3c")]:
    sub = df_all3_s[df_all3_s["pair"] == pair]["pnl"].cumsum()
    axes[3].plot(range(len(sub)), sub.values, color=color, linewidth=1.0, label=pair, alpha=0.8)
axes[3].set_title(
    f"全ペア統合  {s_all3['trades']}回 / PF {s_all3['pf']:.2f} / 勝率{s_all3['win_rate']:.1f}% / "
    f"p={s_all3['p_value']:.4f} {'★統計的有意' if s_all3['p_value'] < 0.05 else ''}",
    fontsize=10, color="white")
axes[3].legend(facecolor="#1a1a2e", labelcolor="white")
axes[3].set_facecolor("#1a1a2e"); axes[3].tick_params(colors="white")
axes[3].set_xlabel("トレード番号", color="white")
axes[3].set_ylabel("累積損益 (pips)", color="white")
axes[3].axhline(0, color="white", linewidth=0.5, linestyle="--")

plt.tight_layout(pad=2.0)
out_img = f"{RESULTS}/v76_multipair_validation.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.close()
print(f"\nチャート保存: {out_img}")

df_eur.to_csv(f"{RESULTS}/v76_eurjpy_trades.csv", index=False)
df_gbp.to_csv(f"{RESULTS}/v76_gbpjpy_trades.csv", index=False)
df_all3.to_csv(f"{RESULTS}/v76_all3pairs_trades.csv", index=False)
print("トレードログ保存完了")
