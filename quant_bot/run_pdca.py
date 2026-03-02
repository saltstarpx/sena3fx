"""
やがみ5条件 PDCA サイクル — 定量分析スクリプト。

実行フロー:
  [PLAN]  バリアント定義 (entry_threshold / sl_atr_mult / tp_atr_mult / timeframe)
  [DO]    EntryScanner で実データをスキャン → BacktestEngine で評価
  [CHECK] Sharpe / Calmar / WR / PF / MDD を比較表示
  [ACT]   results/ に CSV 保存 + knowledge.json 更新

使い方:
  cd /home/user/sena3fx
  python quant_bot/run_pdca.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quant_bot.entry_engine.scanner import EntryScanner
from quant_bot.backtest.engine import QuantBacktestEngine

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pdca")

# ============================================================
# 定数
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"
TRADE_LOGS_DIR = PROJECT_ROOT / "trade_logs"
KNOWLEDGE_JSON = PROJECT_ROOT / "knowledge.json"
DATA_DIR = PROJECT_ROOT / "data" / "ohlc"

# 合格基準 (Yagami 高勝率)
CRITERIA = {
    "min_profit_factor": 1.8,
    "max_drawdown_pct":  15.0,
    "min_win_rate_pct":  60.0,
    "min_trades":        30,
    "min_sharpe":        1.5,
}

# ============================================================
# データロード (Bug 4 修正: UTC タイムゾーン剥ぎ取り)
# ============================================================

def load_ohlcv(filename: str) -> pd.DataFrame:
    """CSV をロードして UTC タイムゾーンを除去。"""
    path = DATA_DIR / filename
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "datetime"
    return df


# ============================================================
# スキャナー → シグナルシリーズ変換
# ============================================================

def scan_to_signal_series(
    ohlcv_df: pd.DataFrame,
    config: dict,
    instrument: str,
    timeframe: str,
    require_c5: bool = False,
) -> tuple[pd.Series, list[dict]]:
    """
    EntryScanner でスキャンし、BacktestEngine 用のシグナルシリーズに変換。

    Args:
        require_c5: True の場合、C5 (足更新タイミング) が充足したバーのみシグナル出力。
                    「足更新でポジションを取る」やがみメソッドに準拠。

    Returns:
        (signal_series, records):
            signal_series: pd.Series (index=datetime, values='long'/'short'/None)
            records: scanner が yield した全レコードリスト
    """
    scanner = EntryScanner(config)
    records = list(scanner.scan_dataframe(ohlcv_df, instrument, timeframe))

    if not records:
        log.warning(f"シグナルゼロ: {instrument} {timeframe}")
        return pd.Series(dtype=object, name="signal"), []

    # JSONL records → {timestamp: signal} dict
    sig_map: dict[pd.Timestamp, str] = {}
    for r in records:
        if r.get("signal") not in ("LONG", "SHORT"):
            continue
        # C5 必須フィルター: 足更新タイミングのバーのみ
        if require_c5:
            c5_detail = r.get("conditions", {}).get("C5", {})
            if not c5_detail.get("satisfied", False):
                continue
        ts = pd.Timestamp(r["timestamp"])
        # BacktestEngine は lowercase を期待
        sig_map[ts] = r["signal"].lower()

    # OHLCV インデックスに合わせた pd.Series を作成
    signal_series = pd.Series(index=ohlcv_df.index, dtype=object, name="signal")
    for ts, sig in sig_map.items():
        if ts in signal_series.index:
            signal_series[ts] = sig

    log.info(
        f"スキャン完了: {instrument} {timeframe}, "
        f"総バー={len(ohlcv_df)}, シグナル={len(sig_map)}"
    )
    return signal_series, records


# ============================================================
# 条件別充足率分析
# ============================================================

def analyze_conditions(records: list[dict]) -> dict:
    """条件別充足率と同時充足パターンを分析。"""
    if not records:
        return {}

    total = len(records)
    cond_ids = ["C1", "C2", "C3", "C4", "C5"]

    # 条件別充足率
    hit_rates = {}
    for cid in cond_ids:
        count = sum(
            1 for r in records
            if r.get("conditions", {}).get(cid, {}).get("satisfied", False)
        )
        hit_rates[cid] = round(count / total * 100, 1)

    # グレード分布
    grades = {"A": 0, "B": 0, "C": 0}
    for r in records:
        g = r.get("grade", "C")
        grades[g] = grades.get(g, 0) + 1

    # シグナル方向分布
    directions = {"LONG": 0, "SHORT": 0, "NO_SIGNAL": 0}
    for r in records:
        sig = r.get("signal") or "NO_SIGNAL"
        directions[sig] = directions.get(sig, 0) + 1

    # 同時充足パターン (上位5件)
    from collections import Counter
    combo_counter: Counter = Counter()
    for r in records:
        conds = r.get("conditions", {})
        satisfied_ids = tuple(
            cid for cid in cond_ids
            if conds.get(cid, {}).get("satisfied", False)
        )
        if satisfied_ids:
            combo_counter[satisfied_ids] += 1

    top_combos = [
        {"+".join(combo): count}
        for combo, count in combo_counter.most_common(5)
    ]

    return {
        "total_bars_scanned": total,
        "condition_hit_rates_pct": hit_rates,
        "grade_distribution": grades,
        "signal_distribution": directions,
        "top_condition_combos": top_combos,
    }


# ============================================================
# PDCA [CHECK] — 結果を判定
# ============================================================

def check_results(summary: dict, variant_name: str) -> dict:
    """バックテスト結果を合格基準に照らして判定。"""
    trades      = summary.get("total_trades", 0)
    wr          = summary.get("win_rate_pct", 0.0)
    pf          = summary.get("profit_factor", 0.0)
    mdd         = abs(summary.get("max_drawdown_pct", 100.0))
    sharpe      = summary.get("sharpe_ratio") or 0.0
    calmar      = summary.get("calmar_ratio") or 0.0
    total_ret   = summary.get("total_return_pct", 0.0)

    passed_checks = {
        "PF≥1.8":     pf >= CRITERIA["min_profit_factor"],
        "MDD≤15%":    mdd <= CRITERIA["max_drawdown_pct"],
        "WR≥60%":     wr >= CRITERIA["min_win_rate_pct"],
        "N≥30":       trades >= CRITERIA["min_trades"],
        "Sharpe≥1.5": sharpe >= CRITERIA["min_sharpe"],
    }
    passed = all(passed_checks.values())

    return {
        "variant":       variant_name,
        "trades":        trades,
        "win_rate_pct":  round(wr, 1),
        "profit_factor": round(pf, 2),
        "sharpe":        round(sharpe, 2),
        "calmar":        round(calmar, 2),
        "mdd_pct":       round(mdd, 1),
        "total_ret_pct": round(total_ret, 1),
        "passed":        passed,
        "checks":        passed_checks,
    }


# ============================================================
# バリアント定義
# ============================================================

def build_variants() -> list[dict]:
    """テストするバリアント一覧を生成。"""
    base_entry_cfg = {
        "c1": {"atr_multiplier": 1.5, "min_touch_count": 2, "level_lookback": 100},
        "c2": {"min_strength": 0.3},
        "c3": {"require_confirmed_pattern": True},
        "c4": {"pivot_window": 5, "min_bars": 25},
        # require_trend=True: C5 充足には HTF 方向が必要 (修正済: 前日完了足で判定)
        "c5": {"require_trend": True},
    }

    def make_config(entry_threshold=3, sl=2.0, tp=4.0, warmup=60, signal_only=False):
        return {
            "conditions": base_entry_cfg,
            "scorer": {
                "entry_threshold": entry_threshold,
                "strong_signal_threshold": 4,
            },
            "scanner": {
                "warmup_bars": warmup,
                "sl_atr_mult": sl,
                "tp_atr_mult": tp,
                "signal_only": signal_only,
            },
        }

    def make_bt_config(sl=2.0, tp=4.0, init_cash=10_000.0, risk_pct=2.0, exit_on_signal=True):
        return {
            "backtest": {
                "initial_balance": init_cash,
                "risk_pct": risk_pct,
                "sl_atr_mult": sl,
                "tp_atr_mult": tp,
                "exit_on_signal": exit_on_signal,
            },
            "logging": {"jsonl_dir": str(TRADE_LOGS_DIR)},
        }

    return [
        # ================================================================
        # PDCA Cycle 4: exit_on_signal=False — 純粋な方向予測エッジ測定
        # ================================================================
        # 仮説: Cycle 3 baseline WR=35.1% > 33.3% (2:1 RR 損益分岐) → 正のエッジ存在
        # 問題: exit_on_signal=True がシグナル反転で早期決済 → TP到達率0.4%
        # 修正: exit_on_signal=False で SL/TP まで保有 → 理論PF=(0.35×4)/(0.65×2)=1.08

        # --- [Cycle4-A] ベースライン hold (全時間帯, 2:1 RR, SL/TPまで保有) ---
        {
            "name": "hold_baseline_H4",
            "data_file": "XAUUSD_4h.csv",
            "timeframe": "H4",
            "instrument": "XAU_USD",
            "scan_cfg": make_config(entry_threshold=3, sl=2.0, tp=4.0),
            "bt_cfg":   make_bt_config(sl=2.0, tp=4.0, exit_on_signal=False),
            "require_c5": False,
        },
        # --- [Cycle4-B] 広い RR hold (3:1, 損益分岐WR=25%, 理論PF=1.62) ---
        {
            "name": "hold_wide_rr_H4",
            "data_file": "XAUUSD_4h.csv",
            "timeframe": "H4",
            "instrument": "XAU_USD",
            "scan_cfg": make_config(entry_threshold=3, sl=2.0, tp=6.0),
            "bt_cfg":   make_bt_config(sl=2.0, tp=6.0, exit_on_signal=False),
            "require_c5": False,
        },
        # --- [Cycle4-C] Aグレード hold (4条件, SL/TP保有) ---
        {
            "name": "hold_strict_H4",
            "data_file": "XAUUSD_4h.csv",
            "timeframe": "H4",
            "instrument": "XAU_USD",
            "scan_cfg": make_config(entry_threshold=4, sl=2.0, tp=4.0),
            "bt_cfg":   make_bt_config(sl=2.0, tp=4.0, exit_on_signal=False),
            "require_c5": False,
        },
        # --- [Cycle4-D] ロンドン/NY C5 + hold ---
        {
            "name": "hold_london_ny_H4",
            "data_file": "XAUUSD_4h.csv",
            "timeframe": "H4",
            "instrument": "XAU_USD",
            "scan_cfg": make_config(entry_threshold=3, sl=2.0, tp=4.0),
            "bt_cfg":   make_bt_config(sl=2.0, tp=4.0, exit_on_signal=False),
            "require_c5": True,
        },
        # --- [Cycle4-E] 比較用: Cycle3 ベースライン (exit_on_signal=True) ---
        {
            "name": "baseline_all_sessions_H4",
            "data_file": "XAUUSD_4h.csv",
            "timeframe": "H4",
            "instrument": "XAU_USD",
            "scan_cfg": make_config(entry_threshold=3, sl=2.0, tp=4.0),
            "bt_cfg":   make_bt_config(sl=2.0, tp=4.0, exit_on_signal=True),
            "require_c5": False,
        },
    ]


# ============================================================
# [ACT] knowledge.json 更新
# ============================================================

def update_knowledge(results: list[dict], condition_analysis: dict) -> None:
    """knowledge.json に PDCA 結果を追記。"""
    try:
        if KNOWLEDGE_JSON.exists():
            with KNOWLEDGE_JSON.open(encoding="utf-8") as f:
                knowledge = json.load(f)
        else:
            knowledge = {}

        passed = [r for r in results if r["passed"]]
        cycle_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        insight_entry = {
            "timestamp": cycle_ts,
            "type": "yagami_pdca",
            "total_variants": len(results),
            "passed_variants": len(passed),
            "best_variant": max(results, key=lambda r: r["sharpe"])["variant"] if results else None,
            "condition_hit_rates": condition_analysis.get("condition_hit_rates_pct", {}),
            "top_combos": condition_analysis.get("top_condition_combos", []),
        }

        if "insights" not in knowledge:
            knowledge["insights"] = []
        knowledge["insights"].append(insight_entry)

        if passed:
            best = max(passed, key=lambda r: r["sharpe"])
            if "best_strategies" not in knowledge:
                knowledge["best_strategies"] = []
            knowledge["best_strategies"].append({
                "name": best["variant"],
                "type": "yagami_5cond",
                "sharpe": best["sharpe"],
                "calmar": best["calmar"],
                "win_rate_pct": best["win_rate_pct"],
                "profit_factor": best["profit_factor"],
                "mdd_pct": best["mdd_pct"],
                "trades": best["trades"],
                "cycle": cycle_ts,
            })

        with KNOWLEDGE_JSON.open("w", encoding="utf-8") as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

        print(f"\n✓ knowledge.json 更新: {KNOWLEDGE_JSON}")
    except Exception as e:
        print(f"\n⚠ knowledge.json 更新失敗: {e}")


# ============================================================
# メイン PDCA ループ
# ============================================================

def run_pdca() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRADE_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  やがみ5条件 PDCA サイクル — 定量分析")
    print("=" * 65)

    variants = build_variants()
    all_results: list[dict] = []
    all_records: list[dict] = []

    for v in variants:
        print(f"\n▶ [{v['name']}] スキャン中...")

        # データロード (Bug 4 対応)
        try:
            ohlcv_df = load_ohlcv(v["data_file"])
        except FileNotFoundError:
            print(f"  ✗ データファイルなし: {v['data_file']}")
            continue

        print(f"  データ: {len(ohlcv_df)} バー ({ohlcv_df.index[0].date()} 〜 {ohlcv_df.index[-1].date()})")

        # [DO] スキャン
        signal_series, records = scan_to_signal_series(
            ohlcv_df, v["scan_cfg"], v["instrument"], v["timeframe"],
            require_c5=v.get("require_c5", False),
        )
        all_records.extend(records)

        n_signals = signal_series.notna().sum()
        print(f"  シグナル数: {n_signals} / {len(ohlcv_df)} バー ({n_signals/len(ohlcv_df)*100:.1f}%)")

        if n_signals == 0:
            print(f"  ✗ シグナルゼロ — バックテストをスキップ")
            all_results.append({
                "variant": v["name"], "trades": 0, "win_rate_pct": 0.0,
                "profit_factor": 0.0, "sharpe": 0.0, "calmar": 0.0,
                "mdd_pct": 0.0, "total_ret_pct": 0.0, "passed": False,
                "checks": {k: False for k in ["PF≥1.8","MDD≤15%","WR≥60%","N≥30","Sharpe≥1.5"]},
            })
            continue

        # [DO] バックテスト
        print(f"  バックテスト実行中...")
        bt_engine = QuantBacktestEngine(v["bt_cfg"])
        try:
            bt_result = bt_engine.run_with_signals(
                ohlcv_df=ohlcv_df,
                signal_series=signal_series,
                instrument=v["instrument"],
                timeframe=v["timeframe"],
                rule_ids=["C1", "C2", "C3", "C4", "C5"],
            )
            summary = bt_result["summary"]
        except Exception as e:
            print(f"  ✗ バックテストエラー: {e}")
            all_results.append({
                "variant": v["name"], "trades": 0, "win_rate_pct": 0.0,
                "profit_factor": 0.0, "sharpe": 0.0, "calmar": 0.0,
                "mdd_pct": 0.0, "total_ret_pct": 0.0, "passed": False,
                "checks": {k: False for k in ["PF≥1.8","MDD≤15%","WR≥60%","N≥30","Sharpe≥1.5"]},
            })
            continue

        # [CHECK] 判定
        checked = check_results(summary, v["name"])
        all_results.append(checked)

        status = "✅ PASS" if checked["passed"] else "❌ FAIL"
        print(
            f"  {status} | trades={checked['trades']} | "
            f"WR={checked['win_rate_pct']}% | PF={checked['profit_factor']} | "
            f"Sharpe={checked['sharpe']} | MDD={checked['mdd_pct']}%"
        )

    # ============================================================
    # [CHECK] 結果テーブル表示
    # ============================================================

    print("\n" + "=" * 65)
    print("  PDCA [CHECK] — バリアント比較")
    print("=" * 65)

    header = f"{'variant':<22} {'N':>5} {'WR%':>6} {'PF':>5} {'Sharpe':>7} {'Calmar':>7} {'MDD%':>6} {'Pass'}"
    print(header)
    print("-" * 65)

    for r in all_results:
        passed_str = "YES" if r["passed"] else "no"
        print(
            f"{r['variant']:<22} {r['trades']:>5} {r['win_rate_pct']:>6.1f} "
            f"{r['profit_factor']:>5.2f} {r['sharpe']:>7.2f} {r['calmar']:>7.2f} "
            f"{r['mdd_pct']:>6.1f} {passed_str}"
        )

    # ============================================================
    # [CHECK] 条件別分析
    # ============================================================

    print("\n" + "=" * 65)
    print("  条件別充足率分析 (H4 baseline)")
    print("=" * 65)

    # H4 baseline のレコードのみ
    h4_records = [
        r for r in all_records
        if r.get("timeframe") == "H4"
    ]
    cond_analysis = analyze_conditions(h4_records)

    if cond_analysis:
        print(f"  スキャンバー数: {cond_analysis['total_bars_scanned']}")
        print("\n  条件充足率:")
        for cid, rate in cond_analysis["condition_hit_rates_pct"].items():
            bar = "█" * int(rate / 5)
            print(f"    {cid}: {rate:5.1f}%  {bar}")

        print("\n  グレード分布:")
        gd = cond_analysis["grade_distribution"]
        total_g = sum(gd.values()) or 1
        for grade, cnt in gd.items():
            pct = cnt / total_g * 100
            print(f"    {grade}: {cnt} ({pct:.1f}%)")

        print("\n  シグナル分布:")
        for sig, cnt in cond_analysis["signal_distribution"].items():
            print(f"    {sig}: {cnt}")

        print("\n  同時充足パターン Top5:")
        for combo_dict in cond_analysis["top_condition_combos"]:
            for combo, cnt in combo_dict.items():
                print(f"    {combo}: {cnt}回")

    # ============================================================
    # [ACT] CSV 保存
    # ============================================================

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"yagami_pdca_{ts_str}.csv"
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ 結果 CSV: {csv_path}")

    # [ACT] knowledge.json 更新
    update_knowledge(all_results, cond_analysis if h4_records else {})

    # 合格バリアントのサマリー
    passed = [r for r in all_results if r["passed"]]
    print(f"\n{'=' * 65}")
    print(f"  合格バリアント: {len(passed)} / {len(all_results)}")
    if passed:
        best = max(passed, key=lambda r: r["sharpe"])
        print(f"  最良: {best['variant']} (Sharpe={best['sharpe']}, PF={best['profit_factor']})")
    else:
        print("  ⚠ 合格バリアントなし — パラメータ調整が必要")
        # ボトルネック診断
        if cond_analysis:
            rates = cond_analysis["condition_hit_rates_pct"]
            bottleneck = min(rates, key=rates.get)
            print(f"  診断: 最低充足率は {bottleneck} ({rates[bottleneck]}%) — この条件が厳しすぎる可能性")
    print("=" * 65)


if __name__ == "__main__":
    run_pdca()
