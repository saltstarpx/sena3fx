"""
v76 vs v76_improved 比較バックテスト
=====================================
1分足データがないため、15分足から1分足を代替生成して比較。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'ohlc')


def load_csv(path):
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def fake_1m_from_15m(data_15m):
    """15分足を1分足タイムスタンプに展開（各15分足の先頭を1分足として扱う）"""
    return data_15m.copy()


def evaluate_signals(signals, data_15m):
    """シグナルの損益を簡易シミュレーション"""
    results = []
    for sig in signals:
        t = sig["time"]
        ep = sig["ep"]
        sl = sig["sl"]
        tp = sig["tp"]
        d = sig["dir"]

        future = data_15m[data_15m.index > t]
        if len(future) == 0:
            continue

        pnl = None
        exit_time = None
        exit_reason = None

        for _, bar in future.iterrows():
            if d == 1:  # ロング
                if bar["low"] <= sl:
                    pnl = (sl - ep) * 100 * d
                    exit_reason = "SL"
                    exit_time = bar.name
                    break
                if bar["high"] >= tp:
                    pnl = (tp - ep) * 100 * d
                    exit_reason = "TP"
                    exit_time = bar.name
                    break
            else:  # ショート
                if bar["high"] >= sl:
                    pnl = (sl - ep) * 100 * d
                    exit_reason = "SL"
                    exit_time = bar.name
                    break
                if bar["low"] <= tp:
                    pnl = (tp - ep) * 100 * d
                    exit_reason = "TP"
                    exit_time = bar.name
                    break

        if pnl is not None:
            results.append({
                "time": t,
                "dir": "long" if d == 1 else "short",
                "tf": sig["tf"],
                "pattern": sig["pattern"],
                "ep": ep,
                "sl": sl,
                "tp": tp,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "exit_time": exit_time,
            })

    return results


def print_summary(name, results):
    if not results:
        print(f"\n{name}: シグナル0件")
        return

    df = pd.DataFrame(results)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    total_pnl = df["pnl"].sum()
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  トレード数:  {len(df)}")
    print(f"  勝ち:        {len(wins)} / 負け: {len(losses)}")
    print(f"  勝率:        {win_rate:.1f}%")
    print(f"  PF:          {pf:.3f}")
    print(f"  純損益:      {total_pnl:,.0f}")
    print(f"  平均損益:    {df['pnl'].mean():,.1f}")

    if len(df) > 0:
        print(f"\n  --- 内訳 ---")
        for _, row in df.iterrows():
            d = row['dir']
            print(f"  {row['time']}  {d:>5s}  {row['tf']}  {row['pattern']:<15s}  PnL={row['pnl']:>+8.1f}  {row['exit_reason']}")


def main():
    # XAUUSDデータ
    path_15m = os.path.join(DATA_DIR, 'XAUUSD_2025_15m.csv')
    path_4h = os.path.join(DATA_DIR, 'XAUUSD_2025_4h.csv')

    if not os.path.exists(path_15m) or not os.path.exists(path_4h):
        print("XAUUSD_2025_15m.csv / XAUUSD_2025_4h.csv が必要です")
        return

    data_15m = load_csv(path_15m)
    data_4h = load_csv(path_4h)
    data_1m = fake_1m_from_15m(data_15m)

    print(f"データ期間: {data_15m.index[0]} ~ {data_15m.index[-1]}")
    print(f"15分足: {len(data_15m)}本, 4時間足: {len(data_4h)}本")

    # ── v76 オリジナル ──
    from strategies.current.yagami_mtf_v76 import generate_signals as gen_v76
    signals_v76 = gen_v76(data_1m, data_15m, data_4h, spread_pips=0.2, rr_ratio=2.5)
    results_v76 = evaluate_signals(signals_v76, data_15m)
    print_summary("v76 オリジナル (RR=2.5)", results_v76)

    # ── v76 improved ──
    from strategies.current.yagami_mtf_v76_improved import generate_signals as gen_improved
    signals_imp = gen_improved(data_1m, data_15m, data_4h, spread_pips=0.2, rr_ratio=3.0)
    results_imp = evaluate_signals(signals_imp, data_15m)
    print_summary("v76 improved (RR=3.0, qlib統合)", results_imp)

    # ── 比較 ──
    print(f"\n{'='*60}")
    print(f"  比較サマリー")
    print(f"{'='*60}")
    n_v76 = len(results_v76)
    n_imp = len(results_imp)
    pnl_v76 = sum(r["pnl"] for r in results_v76)
    pnl_imp = sum(r["pnl"] for r in results_imp)
    print(f"  v76:      {n_v76}トレード, 純損益 {pnl_v76:>+10,.0f}")
    print(f"  improved: {n_imp}トレード, 純損益 {pnl_imp:>+10,.0f}")
    print(f"  シグナル数変化: {n_v76} → {n_imp} ({n_imp - n_v76:+d})")
    if n_v76 > 0:
        filter_rate = (1 - n_imp / n_v76) * 100
        print(f"  フィルター除外率: {filter_rate:.0f}%")


if __name__ == '__main__':
    main()
