import pandas as pd
import numpy as np
import os

def compute_stats(csv_path, label):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    wins   = df[df["PnL_pips"] > 0]
    losses = df[df["PnL_pips"] < 0]
    n_w = len(wins); n_l = len(losses)
    total_pnl = df["PnL_pips"].sum()
    entries = df[~df["Position"].str.contains("HALF")] if "Position" in df.columns else df
    n_entries = len(entries)
    avg_win  = wins["PnL_pips"].mean()  if n_w > 0 else 0
    avg_loss = losses["PnL_pips"].mean() if n_l > 0 else 0
    pf = wins["PnL_pips"].sum() / abs(losses["PnL_pips"].sum()) if n_l > 0 else np.inf
    wr = n_w / (n_w + n_l) * 100 if (n_w + n_l) > 0 else 0
    kelly = 0.0
    if avg_loss != 0:
        b = abs(avg_win / avg_loss)
        p = n_w / (n_w + n_l)
        kelly = (b * p - (1 - p)) / b

    return {
        "バージョン": label,
        "総損益(pips)": round(total_pnl, 2),
        "エントリー数": n_entries,
        "勝率(%)": round(wr, 1),
        "PF": round(pf, 2),
        "平均利益(pips)": round(avg_win, 2),
        "平均損失(pips)": round(avg_loss, 2),
        "ケリー基準": round(kelly, 4),
    }

results_dir = "/home/ubuntu/sena3fx/results"
versions = [
    ("trades_v62.csv", "v62"),
    ("trades_v63.csv", "v63"),
    ("trades_v64.csv", "v64"),
    ("trades_v65.csv", "v65"),
    ("trades_v66.csv", "v66"),
    ("trades_v67.csv", "v67"),
    ("trades_v68.csv", "v68"),
]

rows = []
for fname, label in versions:
    s = compute_stats(os.path.join(results_dir, fname), label)
    if s:
        rows.append(s)

df_summary = pd.DataFrame(rows)
print(df_summary.to_string(index=False))
df_summary.to_csv(os.path.join(results_dir, "version_comparison.csv"), index=False)
print(f"\nSaved: {results_dir}/version_comparison.csv")
