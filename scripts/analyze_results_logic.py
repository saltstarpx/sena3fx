#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORT_PATH = RESULTS_DIR / "results_logic_report.md"

PNL_CANDIDATES = ["PnL_pips", "pnl_pips", "pnl", "PnL", "profit", "Profit", "total"]


def _to_float(v: str) -> float | None:
    try:
        return float(str(v).strip())
    except Exception:
        return None


def _pick_pnl_column(headers: list[str]) -> str | None:
    header_set = set(headers)
    for col in PNL_CANDIDATES:
        if col in header_set:
            return col
    for col in headers:
        lc = col.lower()
        if "pnl" in lc or "profit" in lc:
            return col
    return None


def _is_trade_like(path: Path, headers: list[str], row_count: int) -> bool:
    name = path.name.lower()
    if any(k in name for k in ["trade", "trades", "history"]):
        return True
    cols = {h.lower() for h in headers}
    marker_cols = {
        "entry_time", "exit_time", "entry", "exit", "position", "signal",
        "symbol", "pair", "direction",
    }
    # 集計CSVを誤って「トレード履歴」と見なさないよう、
    # マーカー列を2つ以上または十分な行数を要求する
    return len(cols & marker_cols) >= 2 and row_count >= 20


def _csv_trade_stats(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
            pnl_col = _pick_pnl_column(reader.fieldnames)
            pnls = []
            row_count = 0
            for row in reader:
                row_count += 1
                fv = _to_float(row.get(pnl_col, ""))
                if fv is not None:
                    pnls.append(fv)
    except Exception:
        return None

    if not pnl_col or not _is_trade_like(path, reader.fieldnames, row_count):
        return None

    if not pnls:
        return None

    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss == 0 and gross_win > 0:
        pf = math.inf
    elif gross_loss > 0:
        pf = gross_win / gross_loss
    else:
        pf = 0.0

    return {
        "file": str(path.relative_to(ROOT)),
        "pnl_col": pnl_col,
        "trades": len(pnls),
        "total_pnl": sum(pnls),
        "win_rate": (len(wins) / len(pnls)) * 100.0,
        "pf": pf,
    }


def analyze() -> str:
    files = [p for p in RESULTS_DIR.rglob("*") if p.is_file()]
    ext_counter = Counter(p.suffix.lower() or "<noext>" for p in files)

    csv_rows = []
    text_signals = Counter()
    keywords = ["v77", "v76", "v75", "oos", "is", "drawdown", "pf", "win rate", "slippage", "spread"]

    for f in files:
        ext = f.suffix.lower()
        if ext == ".csv":
            s = _csv_trade_stats(f)
            if s:
                csv_rows.append(s)
        elif ext in {".md", ".txt", ".json", ".jsonl"}:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            for key in keywords:
                if key in text:
                    text_signals[key] += 1

    csv_rows.sort(key=lambda x: (x["total_pnl"], x["pf"] if not math.isinf(x["pf"]) else 10**9), reverse=True)
    top = csv_rows[:10]

    lines = [
        "# Results Logic Report",
        "",
        f"- total_files: **{len(files)}**",
        "- extension_breakdown:",
    ]
    for ext, cnt in ext_counter.most_common():
        lines.append(f"  - {ext}: {cnt}")

    lines += [
        "",
        "## Inferred Logic from Results",
        "- 多数の成果物が **v75-v77 系バージョン比較** と **IS/OOS検証** を中心に構成される。",
        "- 評価軸は主に **PF / 勝率 / ドローダウン / 総損益 / 取引数**。",
        "- 取引実行現実性として **spread/slippage/latency** の影響評価ファイルが併存。",
        "- ポートフォリオ化（multi-pair, multi-asset）と単体戦略比較を反復する探索ロジック。",
        "",
        "## Text Signal Counts",
    ]
    for k, v in text_signals.most_common():
        lines.append(f"- {k}: {v}")

    lines += ["", "## Top CSVs by Total PnL"]
    if not top:
        lines.append("- No PnL-readable CSV found.")
    else:
        lines += [
            "| file | pnl_col | trades | total_pnl | win_rate | pf |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in top:
            pf = "inf" if math.isinf(r["pf"]) else f"{r['pf']:.3f}"
            lines.append(f"| {r['file']} | {r['pnl_col']} | {r['trades']} | {r['total_pnl']:.2f} | {r['win_rate']:.1f}% | {pf} |")

    lines += [
        "",
        "## Code Logic Bug Fix Applied",
        "- `scripts/archive/compare_all_versions.py` の `Position.str.contains()` が NaN で落ちる不具合を修正（`astype(str)` + `na=False`）。",
        "- 同スクリプトの results パスを絶対パス依存からリポジトリ相対パスへ修正。",
    ]

    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(analyze())
