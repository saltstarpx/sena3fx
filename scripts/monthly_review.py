"""
monthly_review.py
==================
本番取引ログの月次定量レビュースクリプト

【使い方】
  python scripts/monthly_review.py                   # 当月レビュー
  python scripts/monthly_review.py --month 2026-03   # 特定月
  python scripts/monthly_review.py --all             # 全期間サマリー
  python scripts/monthly_review.py --csv path/to/trades.csv  # ローカルCSV指定

【入力】
  GCS: logs/paper_trades.csv
  ローカル: trade_logs/paper_trades.csv（フォールバック）

【出力】
  results/monthly_review_{YYYY-MM}.png
  results/monthly_review_{YYYY-MM}.txt
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(ROOT, "results")
LOG_DIR  = os.path.join(ROOT, "trade_logs")
BASELINE = os.path.join(ROOT, "results", "backtest_baseline.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ── GCS経由でCSVを読む（オプション） ──────────────────────────────
def load_trades_gcs():
    """GCS logs/paper_trades.csv を pandas DataFrameで返す。失敗時はNone。"""
    try:
        from google.cloud import storage
        import io
        bucket_name = os.environ.get("GCS_BUCKET", "")
        if not bucket_name: return None
        client  = storage.Client()
        bucket  = client.bucket(bucket_name)
        blob    = bucket.blob("logs/paper_trades.csv")
        content = blob.download_as_text()
        return pd.read_csv(io.StringIO(content))
    except Exception:
        return None

def load_trades(csv_path=None):
    """取引ログをロード（GCS → ローカル → 指定パスの優先順）"""
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = load_trades_gcs()
        if df is None:
            local = os.path.join(LOG_DIR, "paper_trades.csv")
            if not os.path.exists(local):
                print("  [WARN] 取引ログが見つかりません。サンプルデータで動作確認します。")
                return _make_sample_df()
            df = pd.read_csv(local)

    # 型変換
    for tc in ["entry_time", "exit_time"]:
        if tc in df.columns:
            df[tc] = pd.to_datetime(df[tc], errors="coerce", utc=True)

    # 派生カラム
    if "entry_time" in df.columns:
        df["entry_hour"] = df["entry_time"].dt.hour
        df["entry_dow"]  = df["entry_time"].dt.weekday   # 0=Mon
        df["month"]      = df["entry_time"].dt.to_period("M")

    if "pnl" in df.columns:
        df["win"] = (df["pnl"] > 0).astype(int)

    return df

def _make_sample_df():
    """動作確認用サンプルデータ（実取引なし時）"""
    rng = np.random.default_rng(42)
    n   = 60
    pairs = ["XAUUSD","SPX500","GBPUSD","AUDUSD","NZDUSD"]
    times = pd.date_range("2026-03-01", periods=n, freq="12h", tz="UTC")
    return pd.DataFrame({
        "trade_id":   [f"T{i:04d}" for i in range(n)],
        "pair":       rng.choice(pairs, n),
        "dir":        rng.choice([1,-1], n),
        "ep":         rng.uniform(1.0, 2000.0, n),
        "sl":         rng.uniform(0.9, 1999.0, n),
        "tp":         rng.uniform(1.1, 2001.0, n),
        "exit_price": rng.uniform(0.9, 2001.0, n),
        "exit_type":  rng.choice(["TP","SL","BE"], n, p=[0.45,0.35,0.20]),
        "pnl":        rng.normal(5, 20, n),
        "strategy":   "gold_logic",
        "entry_time": times,
        "exit_time":  times + pd.Timedelta(hours=rng.integers(1,24,n).mean()),
        "entry_hour": times.hour,
        "entry_dow":  times.weekday,
        "month":      times.to_period("M"),
        "win":        (rng.normal(5, 20, n) > 0).astype(int),
        "risk_pips":  rng.uniform(5, 30, n),
    })

# ── 統計ヘルパー ──────────────────────────────────────────────────
def pf(df_sub):
    gw = df_sub[df_sub["pnl"]>0]["pnl"].sum()
    gl = abs(df_sub[df_sub["pnl"]<0]["pnl"].sum())
    return gw/gl if gl>0 else float("inf")

def kelly(df_sub):
    wr  = df_sub["win"].mean() if len(df_sub)>0 else 0
    avg_w = df_sub[df_sub["pnl"]>0]["pnl"].mean() if (df_sub["pnl"]>0).any() else 0
    avg_l = abs(df_sub[df_sub["pnl"]<0]["pnl"].mean()) if (df_sub["pnl"]<0).any() else 1
    return wr - (1-wr)/(avg_w/avg_l) if avg_l>0 else 0

def mdd(df_sub):
    eq = df_sub["pnl"].cumsum()
    pk = eq.cummax()
    dd = (pk - eq) / (pk.abs() + 1e-9) * 100
    return dd.max()

def chi2_wr(sub_wins, sub_n, total_wins, total_n):
    if sub_n < 5: return 1.0
    exp_w = total_wins/total_n * sub_n
    exp_l = (total_n-total_wins)/total_n * sub_n
    if exp_l < 1 or exp_w < 1: return 1.0
    chi2, p = stats.chisquare([sub_wins, sub_n-sub_wins], [exp_w, exp_l])
    return p

# ── 月次レビュー本体 ──────────────────────────────────────────────
def review_month(df_all, target_month, baseline):
    df = df_all[df_all["month"] == target_month].copy()
    print(f"\n{'='*68}")
    print(f"  月次レビュー: {target_month}  ({len(df)}トレード)")
    print(f"{'='*68}")

    if len(df) == 0:
        print("  取引なし")
        return None, None

    syms = baseline.get("symbols", {})
    report_lines = [f"【月次レビュー】{target_month}", ""]

    # ── 1. 銘柄別成績 ────────────────────────────────────────────
    print(f"\n  ■ 銘柄別成績")
    print(f"  {'銘柄':8} {'N':>5} {'WR':>6} {'PF':>6} {'MDD':>6} {'Kelly':>7} "
          f"{'PnL':>8} | {'期待WR':>7} {'期待PF':>7} {'乖離':>6}")
    print(f"  {'-'*75}")

    sym_stats = []
    for sym, grp in df.groupby("pair"):
        if len(grp) < 3: continue
        n      = len(grp)
        wr_act = grp["win"].mean()
        pf_act = pf(grp)
        mdd_act= mdd(grp)
        k_act  = kelly(grp)
        pnl_sum= grp["pnl"].sum()

        base = syms.get(sym, {})
        wr_exp = base.get("expected_wr", 0) / 100
        pf_exp = base.get("expected_pf", 0)
        wr_gap = wr_act - wr_exp
        pf_ratio = pf_act / pf_exp if pf_exp > 0 else 0

        # ドリフトアラート
        drift_wr  = abs(wr_gap) > baseline["portfolio"].get("drift_alert_wr_gap", 0.08)
        drift_pf  = pf_ratio < baseline["portfolio"].get("drift_alert_pf_ratio", 0.7)
        alert = "⚠️" if (drift_wr or drift_pf) else "✅"

        pf_s = f"{pf_act:.2f}" if pf_act < 99 else "∞"
        print(f"  {sym:8} {n:>5} {wr_act*100:>5.1f}% {pf_s:>6} {mdd_act:>5.1f}% "
              f"{k_act:>6.3f} {pnl_sum:>+8.1f}p | "
              f"{wr_exp*100:>6.1f}% {pf_exp:>7.2f} "
              f"{'↑' if wr_gap>0 else '↓'}{abs(wr_gap)*100:>4.1f}pp {alert}")

        sym_stats.append({
            "sym": sym, "n": n, "wr": wr_act, "pf": pf_act, "mdd": mdd_act,
            "kelly": k_act, "pnl": pnl_sum, "wr_exp": wr_exp, "pf_exp": pf_exp,
            "wr_gap": wr_gap, "pf_ratio": pf_ratio, "alert": alert
        })
        report_lines.append(
            f"{sym}: n={n} WR={wr_act*100:.1f}%(期待{wr_exp*100:.1f}%) "
            f"PF={pf_s}(期待{pf_exp:.2f}) PnL={pnl_sum:+.1f}p {alert}"
        )

    # ── 2. ポートフォリオ合計 ────────────────────────────────────
    total_pnl  = df["pnl"].sum()
    total_wr   = df["win"].mean()
    total_pf   = pf(df)
    total_mdd  = mdd(df)
    total_k    = kelly(df)
    print(f"  {'[合計]':8} {len(df):>5} {total_wr*100:>5.1f}% "
          f"{total_pf:>6.2f} {total_mdd:>5.1f}% {total_k:>6.3f} {total_pnl:>+8.1f}p")
    report_lines += ["",
        f"【合計】n={len(df)} WR={total_wr*100:.1f}% PF={total_pf:.2f} "
        f"MDD={total_mdd:.1f}% Kelly={total_k:.3f} PnL={total_pnl:+.1f}p"]

    # ── 3. 時間帯別WR（全銘柄合計） ─────────────────────────────
    print(f"\n  ■ UTC時間帯別 勝率（有意差のみ）")
    bad_hours = []; good_hours = []
    if "entry_hour" in df.columns:
        total_w = df["win"].sum(); total_n = len(df)
        for h in range(24):
            sub = df[df["entry_hour"]==h]
            if len(sub) < 5: continue
            p_val = chi2_wr(sub["win"].sum(), len(sub), total_w, total_n)
            wr_h  = sub["win"].mean()
            gap   = wr_h - total_wr
            if p_val < 0.1:
                marker = "❌" if gap < -0.05 else ("✅" if gap > 0.05 else "")
                if marker:
                    print(f"    UTC {h:02d}: WR={wr_h*100:.1f}% (基準{total_wr*100:.1f}%, "
                          f"gap={gap*100:+.1f}pp, p={p_val:.3f}) {marker}")
                    if gap < -0.05: bad_hours.append(h)
                    else: good_hours.append(h)

    # ── 4. 方向別分析 ──────────────────────────────────────────
    print(f"\n  ■ 方向別（Long/Short）")
    if "dir" in df.columns:
        for d, lbl in [(1,"Long"),(-1,"Short")]:
            sub = df[df["dir"]==d]
            if len(sub) < 3: continue
            wr_d = sub["win"].mean()
            pf_d = pf(sub)
            print(f"    {lbl:6}: n={len(sub):>4} WR={wr_d*100:.1f}% PF={pf_d:.2f}")

    # ── 5. バックテスト vs 本番 比較サマリー ────────────────────
    print(f"\n  ■ バックテスト vs 本番 乖離チェック")
    alerts = [s for s in sym_stats if s["alert"] == "⚠️"]
    if alerts:
        for a in alerts:
            print(f"    ⚠️ {a['sym']}: "
                  f"WR乖離 {a['wr_gap']*100:+.1f}pp / "
                  f"PF比率 {a['pf_ratio']:.2f}x")
        report_lines.append("\n【要注意銘柄】")
        for a in alerts:
            report_lines.append(f"⚠️ {a['sym']}: WR{a['wr_gap']*100:+.1f}pp PF比{a['pf_ratio']:.2f}x")
    else:
        print(f"    ✅ 全銘柄 乖離なし（期待値±8pp以内）")
        report_lines.append("✅ 全銘柄 乖離なし")

    report_lines.append(f"\n【時間帯アラート】")
    if bad_hours:
        report_lines.append(f"⚠️ 勝率低下UTC時間: {sorted(bad_hours)}")
    if good_hours:
        report_lines.append(f"✅ 勝率上昇UTC時間: {sorted(good_hours)}")
    if not bad_hours and not good_hours:
        report_lines.append("特記なし")

    return sym_stats, "\n".join(report_lines)

# ── 全期間サマリー ────────────────────────────────────────────────
def review_all(df, baseline):
    print(f"\n{'='*68}")
    print(f"  全期間サマリー ({df['month'].min()} 〜 {df['month'].max()})")
    print(f"{'='*68}")
    print(f"  総トレード数: {len(df)}")
    print(f"  取引期間:     {df['month'].nunique()}ヶ月")

    monthly = df.groupby("month")["pnl"].sum()
    print(f"\n  月次損益 (pip換算):")
    for m, v in monthly.items():
        bar = "█" * int(abs(v) / max(1, monthly.abs().max()) * 20)
        print(f"  {m}:  {v:>+8.1f}p  {'+'*max(0,int(v/5)):20s}" if v>0 else
              f"  {m}:  {v:>+8.1f}p  {'-'*max(0,int(-v/5)):20s}")

    print(f"\n  銘柄別全期間PF:")
    for sym, grp in df.groupby("pair"):
        print(f"    {sym:8}: PF={pf(grp):.2f}  WR={grp['win'].mean()*100:.1f}%  "
              f"n={len(grp)}  累計={grp['pnl'].sum():+.1f}p")

# ── グラフ描画 ────────────────────────────────────────────────────
def plot_monthly(df_month, sym_stats, target_month, baseline):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"月次レビュー {target_month}  |  YAGAMI改 本番トレード分析",
                 color="white", fontsize=14, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

    # ── (0,0) エクイティカーブ ─────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("#16213e")
    if len(df_month) > 0:
        eq = df_month.sort_values("entry_time")["pnl"].cumsum()
        ax0.plot(range(len(eq)), eq.values, color="#2ecc71", linewidth=1.5)
        ax0.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        ax0.fill_between(range(len(eq)), 0, eq.values,
                         where=eq.values>=0, alpha=0.2, color="#2ecc71")
        ax0.fill_between(range(len(eq)), 0, eq.values,
                         where=eq.values<0, alpha=0.2, color="#e74c3c")
    total_pf_val = pf(df_month) if len(df_month)>0 else 0
    pf_s = f"{total_pf_val:.2f}" if total_pf_val < 99 else "∞"
    ax0.set_title(f"月次エクイティ  PF={pf_s}  n={len(df_month)}",
                  color="white", fontsize=9)
    ax0.set_xlabel("トレード番号", color="#aaa", fontsize=7)
    ax0.set_ylabel("累計損益(pip)", color="#aaa", fontsize=7)
    ax0.tick_params(colors="#aaa", labelsize=7)
    for sp in ax0.spines.values(): sp.set_color("#333")

    # ── (0,1) 銘柄別 PF比較（実績 vs 期待値） ───────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor("#16213e")
    if sym_stats:
        syms_  = [s["sym"] for s in sym_stats]
        pf_act = [s["pf"] if s["pf"]<10 else 10 for s in sym_stats]
        pf_exp = [s["pf_exp"] for s in sym_stats]
        x = np.arange(len(syms_)); w = 0.35
        ax1.bar(x-w/2, pf_act, w, label="実績PF", color="#4C9BE8", alpha=0.85)
        ax1.bar(x+w/2, pf_exp, w, label="期待PF", color="#555",    alpha=0.85)
        ax1.axhline(2.0, color="#F5A623", linewidth=0.8, linestyle="--", alpha=0.7)
        ax1.set_xticks(x); ax1.set_xticklabels(syms_, fontsize=7, rotation=20)
        ax1.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
    ax1.set_title("銘柄別PF（実績 vs 期待）", color="white", fontsize=9)
    ax1.tick_params(colors="#aaa", labelsize=7)
    for sp in ax1.spines.values(): sp.set_color("#333")

    # ── (0,2) 銘柄別 WR比較 ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#16213e")
    if sym_stats:
        wr_act = [s["wr"]*100 for s in sym_stats]
        wr_exp = [s["wr_exp"]*100 for s in sym_stats]
        ax2.bar(x-w/2, wr_act, w, label="実績WR%", color="#2ecc71", alpha=0.85)
        ax2.bar(x+w/2, wr_exp, w, label="期待WR%", color="#555",    alpha=0.85)
        ax2.axhline(65, color="#F5A623", linewidth=0.8, linestyle="--", alpha=0.7)
        ax2.set_xticks(x); ax2.set_xticklabels(syms_, fontsize=7, rotation=20)
        ax2.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
    ax2.set_title("銘柄別WR%（実績 vs 期待）", color="white", fontsize=9)
    ax2.tick_params(colors="#aaa", labelsize=7)
    for sp in ax2.spines.values(): sp.set_color("#333")

    # ── (1,0) UTC時間帯別 WR ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#16213e")
    if "entry_hour" in df_month.columns and len(df_month) > 20:
        hour_wr  = df_month.groupby("entry_hour")["win"].mean() * 100
        hour_n   = df_month.groupby("entry_hour")["win"].count()
        base_wr  = df_month["win"].mean() * 100
        colors   = ["#e74c3c" if w < base_wr-5 else
                    "#2ecc71" if w > base_wr+5 else "#4C9BE8"
                    for w in hour_wr]
        ax3.bar(hour_wr.index, hour_wr.values, color=colors, alpha=0.8)
        ax3.axhline(base_wr, color="#fff", linewidth=1, linestyle="--", alpha=0.6)
        for h, n_ in hour_n.items():
            ax3.text(h, hour_wr.get(h, 0)+0.5, str(n_), ha="center",
                     fontsize=5, color="#aaa")
    ax3.set_title("UTC時間帯別WR% (数字=n)", color="white", fontsize=9)
    ax3.set_xlabel("UTC hour", color="#aaa", fontsize=7)
    ax3.set_ylabel("WR%", color="#aaa", fontsize=7)
    ax3.tick_params(colors="#aaa", labelsize=7)
    for sp in ax3.spines.values(): sp.set_color("#333")

    # ── (1,1) 方向別 PnL分布 ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#16213e")
    if "dir" in df_month.columns and len(df_month) > 5:
        long_pnl  = df_month[df_month["dir"]== 1]["pnl"]
        short_pnl = df_month[df_month["dir"]==-1]["pnl"]
        bins = np.linspace(df_month["pnl"].quantile(0.02),
                           df_month["pnl"].quantile(0.98), 30)
        if len(long_pnl)  > 2: ax4.hist(long_pnl,  bins=bins, alpha=0.6,
                                         color="#2ecc71", label=f"Long n={len(long_pnl)}")
        if len(short_pnl) > 2: ax4.hist(short_pnl, bins=bins, alpha=0.6,
                                         color="#e74c3c", label=f"Short n={len(short_pnl)}")
        ax4.axvline(0, color="#fff", linewidth=0.8, linestyle="--")
        ax4.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
    ax4.set_title("Long/Short PnL分布", color="white", fontsize=9)
    ax4.set_xlabel("損益(pip)", color="#aaa", fontsize=7)
    ax4.tick_params(colors="#aaa", labelsize=7)
    for sp in ax4.spines.values(): sp.set_color("#333")

    # ── (1,2) 月次サマリーテキスト ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#16213e")
    ax5.axis("off")
    if sym_stats:
        lines = [f"【{target_month} サマリー】", ""]
        for s in sym_stats:
            pf_s = f"{s['pf']:.2f}" if s['pf']<99 else "∞"
            lines.append(f"{s['sym']:7}: WR={s['wr']*100:.1f}% PF={pf_s} {s['alert']}")
        lines += ["",
            f"合計n={len(df_month)}  PF={pf_s}",
            f"月次PnL: {df_month['pnl'].sum():+.1f}p",
            f"MDD: {mdd(df_month):.1f}%",
        ]
        ax5.text(0.05, 0.95, "\n".join(lines), transform=ax5.transAxes,
                 color="white", fontsize=9, va="top", fontfamily="monospace")

    out_path = os.path.join(OUT_DIR, f"monthly_review_{target_month}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return out_path

# ── エントリーポイント ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="月次レビュースクリプト")
    parser.add_argument("--month",  default=None, help="対象月 YYYY-MM（デフォルト: 当月）")
    parser.add_argument("--all",    action="store_true", help="全期間サマリー")
    parser.add_argument("--csv",    default=None, help="ローカルCSVパス")
    args = parser.parse_args()

    # ベースライン読み込み
    with open(BASELINE, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    # 取引ログ読み込み
    print("\n  取引ログ読み込み中...", end=" ", flush=True)
    df = load_trades(args.csv)
    print(f"{len(df)}トレード")

    if args.all:
        review_all(df, baseline)
        return

    # 対象月決定
    if args.month:
        target = pd.Period(args.month, freq="M")
    else:
        target = pd.Period(datetime.now(timezone.utc), freq="M")

    # 月次レビュー実行
    sym_stats, report_text = review_month(df, target, baseline)

    if report_text is None:
        return

    # グラフ生成
    df_month = df[df["month"] == target].copy()
    print(f"\n  グラフ生成中...", end=" ", flush=True)
    out_png = plot_monthly(df_month, sym_stats, str(target), baseline)
    print(f"→ {out_png}")

    # テキストレポート保存
    out_txt = os.path.join(OUT_DIR, f"monthly_review_{target}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  テキスト → {out_txt}")

    print("\n完了")

if __name__ == "__main__":
    main()
