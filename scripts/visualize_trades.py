"""
visualize_trades.py
===================
バックテスト取引をローソク足チャート上に可視化
- エントリー: 青↑(Long) / 赤↓(Short)
- SL: 赤破線 + ×マーク
- TP: 緑破線 + ★マーク
- 半利確(1R): オレンジ破線
- BE決済: 黄色●マーク
出力: results/trade_charts/ に銘柄別PNG
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import SYMBOL_CONFIG

# ── backtest_portfolio_680k からインポート ──
from scripts.backtest_portfolio_680k import (
    load_1m, calc_atr, generate_signals, TARGETS,
    A1_EMA_DIST_MIN, MAX_LOOKAHEAD, RR_RATIO, HALF_R,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "trade_charts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 15分足にリサンプル（見やすさ重視） ──
def resample_15m(d1m):
    r = d1m.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["open", "close"])
    return r

# ── ローソク足描画 ──
def draw_candles(ax, df, width_minutes=10):
    """df: DatetimeIndex, OHLC columns"""
    w = timedelta(minutes=width_minutes)
    for i in range(len(df)):
        t = df.index[i]
        o, h, l, c = df.iloc[i][["open", "high", "low", "close"]]
        color = "#26a69a" if c >= o else "#ef5350"  # 緑=陽線, 赤=陰線
        # ヒゲ
        ax.plot([t, t], [l, h], color=color, linewidth=0.8, zorder=1)
        # 実体
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < (h - l) * 0.001:
            body_height = (h - l) * 0.005  # 十字線の最小幅
        rect = plt.Rectangle((t - w/2, body_bottom), w, body_height,
                              facecolor=color, edgecolor=color, linewidth=0.5, zorder=2)
        ax.add_patch(rect)

# ── トレードのexit情報を詳細に取得 ──
def trace_trade(d1m, sig):
    """1分足でトレードを追跡し、exit bar index・種別・半利確barを返す"""
    m1t = d1m.index
    m1h = d1m["high"].values
    m1l = d1m["low"].values
    sp = m1t.searchsorted(sig["time"], side="right")
    if sp >= len(m1t):
        return None

    ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
    risk = sig["risk"]; d = sig["dir"]
    half_price = ep + d * risk * HALF_R
    lim = min(len(m1h) - sp, MAX_LOOKAHEAD)

    result = {
        "entry_bar": sp, "entry_time": sig["time"], "entry_price": ep,
        "sl": sl, "tp": tp, "half_price": half_price, "dir": d,
        "exit_bar": None, "exit_time": None, "exit_price": None,
        "exit_type": None, "half_bar": None, "half_time": None,
    }

    for i in range(lim):
        h = m1h[sp + i]; lo = m1l[sp + i]
        if d == 1:
            if lo <= sl:
                result.update(exit_bar=sp+i, exit_time=m1t[sp+i], exit_price=sl, exit_type="SL")
                return result
            if h >= tp:
                result.update(exit_bar=sp+i, exit_time=m1t[sp+i], exit_price=tp, exit_type="TP")
                return result
            if h >= half_price:
                result["half_bar"] = sp + i
                result["half_time"] = m1t[sp + i]
                for j in range(i+1, lim):
                    if m1l[sp+j] <= ep:
                        result.update(exit_bar=sp+j, exit_time=m1t[sp+j], exit_price=ep, exit_type="BE")
                        return result
                    if m1h[sp+j] >= tp:
                        result.update(exit_bar=sp+j, exit_time=m1t[sp+j], exit_price=tp, exit_type="TP(半利確後)")
                        return result
                # タイムアウト
                result.update(exit_type="timeout")
                return result
        else:
            if h >= sl:
                result.update(exit_bar=sp+i, exit_time=m1t[sp+i], exit_price=sl, exit_type="SL")
                return result
            if lo <= tp:
                result.update(exit_bar=sp+i, exit_time=m1t[sp+i], exit_price=tp, exit_type="TP")
                return result
            if lo <= half_price:
                result["half_bar"] = sp + i
                result["half_time"] = m1t[sp + i]
                for j in range(i+1, lim):
                    if m1h[sp+j] >= ep:
                        result.update(exit_bar=sp+j, exit_time=m1t[sp+j], exit_price=ep, exit_type="BE")
                        return result
                    if m1l[sp+j] <= tp:
                        result.update(exit_bar=sp+j, exit_time=m1t[sp+j], exit_price=tp, exit_type="TP(半利確後)")
                        return result
                result.update(exit_type="timeout")
                return result
    return None

# ── 1トレード可視化 ──
def plot_trade(d1m, trade_info, sym, trade_idx, tf="15min"):
    """1つのトレードをローソク足チャートで可視化"""
    entry_time = trade_info["entry_time"]
    exit_time = trade_info["exit_time"]
    if exit_time is None:
        return None

    # 前後にパディング（エントリー前2h、exit後1h）
    pad_before = timedelta(hours=3)
    pad_after = timedelta(hours=2)
    start = entry_time - pad_before
    end = exit_time + pad_after

    # 15分足にリサンプル
    chunk = d1m[start:end]
    if len(chunk) < 10:
        return None

    if tf == "15min":
        candles = resample_15m(chunk)
        candle_width = 10
    else:
        candles = chunk
        candle_width = 0.6

    if len(candles) < 5:
        return None

    # チャート作成
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    draw_candles(ax, candles, width_minutes=candle_width)

    ep = trade_info["entry_price"]
    sl = trade_info["sl"]
    tp = trade_info["tp"]
    hp = trade_info["half_price"]
    d = trade_info["dir"]
    exit_price = trade_info["exit_price"]
    exit_type = trade_info["exit_type"]

    # SL/TP/半利確 水平線
    ax.axhline(y=sl, color="#ff4444", linestyle="--", linewidth=1.2, alpha=0.8, label=f"SL: {sl:.5g}")
    ax.axhline(y=tp, color="#44ff44", linestyle="--", linewidth=1.2, alpha=0.8, label=f"TP: {tp:.5g}")
    ax.axhline(y=hp, color="#ffaa00", linestyle=":", linewidth=1.0, alpha=0.6, label=f"1R (半利確): {hp:.5g}")
    ax.axhline(y=ep, color="#4488ff", linestyle="-", linewidth=0.8, alpha=0.5, label=f"Entry: {ep:.5g}")

    # SL/TPゾーン塗りつぶし
    if d == 1:
        ax.axhspan(sl, ep, alpha=0.05, color="red")
        ax.axhspan(ep, tp, alpha=0.05, color="green")
    else:
        ax.axhspan(ep, sl, alpha=0.05, color="red")
        ax.axhspan(tp, ep, alpha=0.05, color="green")

    # エントリー矢印
    arrow_size = 18
    if d == 1:
        ax.annotate("", xy=(entry_time, ep),
                    xytext=(entry_time, ep - (tp - sl) * 0.08),
                    arrowprops=dict(arrowstyle="-|>", color="#4488ff", lw=2.5),
                    zorder=10)
        ax.plot(entry_time, ep, marker="^", color="#4488ff", markersize=arrow_size,
                zorder=11, markeredgecolor="white", markeredgewidth=1.0)
    else:
        ax.annotate("", xy=(entry_time, ep),
                    xytext=(entry_time, ep + (tp - sl) * 0.08),
                    arrowprops=dict(arrowstyle="-|>", color="#ff4488", lw=2.5),
                    zorder=10)
        ax.plot(entry_time, ep, marker="v", color="#ff4488", markersize=arrow_size,
                zorder=11, markeredgecolor="white", markeredgewidth=1.0)

    # Exit マーカー
    if exit_time is not None and exit_price is not None:
        if "SL" in exit_type:
            ax.plot(exit_time, exit_price, marker="X", color="#ff4444",
                    markersize=16, zorder=11, markeredgecolor="white", markeredgewidth=1.0)
        elif "TP" in exit_type:
            ax.plot(exit_time, exit_price, marker="*", color="#44ff44",
                    markersize=20, zorder=11, markeredgecolor="white", markeredgewidth=0.8)
        elif "BE" in exit_type:
            ax.plot(exit_time, exit_price, marker="o", color="#ffdd00",
                    markersize=14, zorder=11, markeredgecolor="white", markeredgewidth=1.0)

    # 半利確ポイント
    if trade_info["half_time"] is not None:
        ax.plot(trade_info["half_time"], hp, marker="D", color="#ffaa00",
                markersize=10, zorder=11, markeredgecolor="white", markeredgewidth=0.8)

    # 保持時間を計算
    hold_td = exit_time - entry_time
    hold_hours = hold_td.total_seconds() / 3600
    if hold_hours < 1:
        hold_str = f"{hold_td.total_seconds()/60:.0f}分"
    elif hold_hours < 24:
        hold_str = f"{hold_hours:.1f}時間"
    else:
        hold_str = f"{hold_hours/24:.1f}日"

    # 損益 pips 計算
    cfg = SYMBOL_CONFIG.get(sym, {})
    pip = cfg.get("pip", 0.0001)
    if d == 1:
        pnl_pips = (exit_price - ep) / pip
    else:
        pnl_pips = (ep - exit_price) / pip
    # 半利確分の概算（半分はhalf_priceで決済）
    if trade_info["half_time"] is not None:
        half_pnl = (hp - ep) * d / pip * 0.5
        remain_pnl = pnl_pips * 0.5
        total_pips = half_pnl + remain_pnl
    else:
        total_pips = pnl_pips

    pnl_color = "#44ff44" if total_pips > 0 else "#ff4444"

    # タイトル
    dir_str = "LONG ↑" if d == 1 else "SHORT ↓"
    title = (f"{sym}  {dir_str}  |  {exit_type}  |  "
             f"{total_pips:+.1f} pips  |  保持: {hold_str}")
    ax.set_title(title, fontsize=14, color="white", fontweight="bold", pad=12)

    # 右側にPnL表示
    ax.text(0.98, 0.95, f"{total_pips:+.1f} pips",
            transform=ax.transAxes, fontsize=16, fontweight="bold",
            color=pnl_color, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor=pnl_color, alpha=0.8))

    # 時間情報
    ax.text(0.02, 0.95,
            f"Entry: {entry_time.strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"Exit:  {exit_time.strftime('%Y-%m-%d %H:%M')} UTC",
            transform=ax.transAxes, fontsize=9, color="#aaaaaa",
            ha="left", va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.8))

    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^" if d==1 else "v",
               color="w", markerfacecolor="#4488ff" if d==1 else "#ff4488",
               markersize=10, label="Entry", linestyle="None"),
        Line2D([0], [0], color="#ff4444", linestyle="--", label=f"SL ({sl:.5g})"),
        Line2D([0], [0], color="#44ff44", linestyle="--", label=f"TP ({tp:.5g})"),
        Line2D([0], [0], color="#ffaa00", linestyle=":", label=f"1R半利確 ({hp:.5g})"),
        Line2D([0], [0], marker="X", color="#ff4444", markersize=8, label="SL決済", linestyle="None"),
        Line2D([0], [0], marker="*", color="#44ff44", markersize=10, label="TP決済", linestyle="None"),
        Line2D([0], [0], marker="o", color="#ffdd00", markersize=8, label="BE決済", linestyle="None"),
        Line2D([0], [0], marker="D", color="#ffaa00", markersize=6, label="半利確ポイント", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
              facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", ncol=2)

    # 軸の設定
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.tick_params(colors="#888888", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")
    ax.grid(True, alpha=0.15, color="#555")
    fig.autofmt_xdate(rotation=30)

    # Y軸ラベル
    ax.set_ylabel("Price", color="#888888", fontsize=10)

    plt.tight_layout()
    fname = f"{sym}_trade_{trade_idx:03d}_{exit_type.replace('(','').replace(')','')}.png"
    fpath = os.path.join(OUT_DIR, fname)
    fig.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fpath

# ── メイン ──
def main():
    print(f"\n{'='*70}")
    print(f"  トレード可視化 — ローソク足 + エントリー/SL/TP")
    print(f"{'='*70}")

    # 各銘柄から代表的なトレードを選択して可視化
    # SL, TP, BE, TP(半利確後) を各1つずつ + ランダム数件
    SAMPLES_PER_TYPE = 2   # 各exit種別から2件ずつ
    RANDOM_EXTRA = 2       # ランダム追加

    total_charts = 0

    for tgt in TARGETS:
        sym = tgt["sym"]; logic = tgt["logic"]
        print(f"\n  {sym} ({logic}) ... ", end="", flush=True)

        d1m = load_1m(sym)
        if d1m is None:
            print("データなし"); continue

        d4h = d1m.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])

        cfg = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d = calc_atr(d1m, 10).to_dict()
        m1c = {"idx": d1m.index, "opens": d1m["open"].values,
               "closes": d1m["close"].values,
               "highs": d1m["high"].values, "lows": d1m["low"].values}

        edm = tgt.get("ema_dist_min", A1_EMA_DIST_MIN)
        hbr = tgt.get("h4_body_ratio_min", 0.0)
        sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c,
                                ema_dist_min=edm, h4_body_ratio_min=hbr)
        print(f"{len(sigs)}シグナル → ", end="", flush=True)

        # 全トレードのexit情報を取得
        trades_by_type = {"SL": [], "TP": [], "BE": [], "TP(半利確後)": []}
        all_valid = []

        for idx, sig in enumerate(sigs):
            info = trace_trade(d1m, sig)
            if info is None or info["exit_type"] is None or info["exit_time"] is None:
                continue
            all_valid.append((idx, info))
            et = info["exit_type"]
            if et in trades_by_type:
                trades_by_type[et].append((idx, info))

        # 各種別からサンプリング
        selected = []
        for etype, trades in trades_by_type.items():
            if not trades:
                continue
            # 保持時間の中央値付近を選択（典型的な例）
            trades_sorted = sorted(trades, key=lambda x: (x[1]["exit_time"] - x[1]["entry_time"]).total_seconds())
            mid = len(trades_sorted) // 2
            indices = list(range(max(0, mid-1), min(len(trades_sorted), mid + SAMPLES_PER_TYPE - 1 + 1)))
            for i in indices[:SAMPLES_PER_TYPE]:
                selected.append(trades_sorted[i])

        # ランダム追加（重複回避）
        sel_ids = {s[0] for s in selected}
        remaining = [(i, info) for i, info in all_valid if i not in sel_ids]
        if remaining:
            np.random.seed(42)
            extra_idx = np.random.choice(len(remaining), size=min(RANDOM_EXTRA, len(remaining)), replace=False)
            for ei in extra_idx:
                selected.append(remaining[ei])

        # チャート生成
        sym_charts = 0
        for trade_idx, info in selected:
            fpath = plot_trade(d1m, info, sym, trade_idx)
            if fpath:
                sym_charts += 1
                total_charts += 1

        print(f"{sym_charts}チャート生成")

    print(f"\n{'='*70}")
    print(f"  合計 {total_charts} チャートを {OUT_DIR}/ に出力")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
