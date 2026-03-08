"""
RUN-20260305-003: 方向A・方向B 並行バックテスト
=================================================
方向A（シグナル精度向上）と方向B（EMAトレンド型HTFフィルター）を
同時並行で検証し、ベースライン（RUN-001）と比較する。

方向A: 9パラメータセット
方向B: 9パラメータセット
ベースライン: 1セット
合計: 19セット
"""
import sys, os, csv, json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.backtest import BacktestEngine
from strategies.yagami_pa import signal_pa1_reversal
from strategies.signal_filter_a import signal_direction_a, DIRECTION_A_PARAMS
from strategies.signal_filter_b import signal_direction_b, DIRECTION_B_PARAMS

DATA_1H  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'usdjpy_1h.csv')
DATA_4H  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ohlc', 'USDJPY_4h.csv')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_1h():
    df = pd.read_csv(DATA_1H, index_col='timestamp', parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def load_4h():
    df = pd.read_csv(DATA_4H, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def make_engine(sl_atr=1.0, tp_atr=3.0):
    return BacktestEngine(
        init_cash=5_000_000, risk_pct=0.02, pip=0.01, slippage_pips=0.5,
        default_sl_atr=sl_atr, default_tp_atr=tp_atr,
        use_dynamic_sl=True, sl_n_confirm=1, pyramid_entries=0,
        trail_start_atr=0.0, target_max_dd=0.20, target_min_wr=0.35
    )

def extract(result, name, direction):
    return {
        'name': name,
        'direction': direction,
        'total_trades': result['total_trades'],
        'win_rate_pct': result['win_rate_pct'],
        'profit_factor': result['profit_factor'],
        'max_drawdown_pct': result['max_drawdown_pct'],
        'total_return_pct': result['total_return_pct'],
        'rr_ratio': result['rr_ratio'],
        'passed': result['passed'],
        'trades': result.get('trades', []),
    }

def score(r):
    pf = min(r['profit_factor'], 5.0)
    wr = r['win_rate_pct'] / 100
    dd = r['max_drawdown_pct'] / 100
    n  = r['total_trades']
    if n < 3:
        return 0.0
    return pf * wr * (1 - dd) * np.log(n + 1)


if __name__ == '__main__':
    print("=== RUN-20260305-003: 方向A・方向B 並行バックテスト ===")
    bars_1h = load_1h()
    bars_4h = load_4h()
    print(f"1時間足: {len(bars_1h)} bars | 4時間足: {len(bars_4h)} bars")

    all_results = []

    # ---- ベースライン ----
    print("\n[ベースライン] PA1_Reversal_TightSL")
    try:
        raw_sig = signal_pa1_reversal(bars_1h, zone_atr=1.5, lookback=20)
        result = make_engine().run(data=bars_1h, signal_func=lambda b: raw_sig,
                                   freq='1h', name='BASELINE')
        if result:
            r = extract(result, 'BASELINE', 'baseline')
            all_results.append(r)
            print(f"  N={r['total_trades']} | WR={r['win_rate_pct']:.1f}% | "
                  f"PF={r['profit_factor']:.3f} | DD={r['max_drawdown_pct']:.1f}% | "
                  f"Ret={r['total_return_pct']:.2f}%")
    except Exception as e:
        print(f"  エラー: {e}")

    # ---- 方向A ----
    print("\n=== 方向A: シグナル精度向上フィルター ===")
    for label, use_ib, use_vol, vol_mult, use_sess, sess in DIRECTION_A_PARAMS:
        try:
            sig = signal_direction_a(
                bars_1h,
                use_inside_bar=use_ib,
                use_volume=use_vol,
                volume_mult=vol_mult,
                use_session=use_sess,
                sessions=sess,
            )
            n_sig = sig.isin(['long','short']).sum()
            if n_sig == 0:
                print(f"  [{label}] シグナルなし、スキップ")
                continue
            result = make_engine().run(data=bars_1h, signal_func=lambda b: sig,
                                       freq='1h', name=label)
            if result:
                r = extract(result, label, 'A')
                all_results.append(r)
                print(f"  [{label}] N={r['total_trades']}(sig={n_sig}) | "
                      f"WR={r['win_rate_pct']:.1f}% | PF={r['profit_factor']:.3f} | "
                      f"DD={r['max_drawdown_pct']:.1f}% | Ret={r['total_return_pct']:.2f}%")
        except Exception as e:
            print(f"  [{label}] エラー: {e}")
            import traceback; traceback.print_exc()

    # ---- 方向B ----
    print("\n=== 方向B: EMAトレンド型HTFフィルター ===")
    for label, ef, es, sp, ms in DIRECTION_B_PARAMS:
        try:
            sig = signal_direction_b(
                bars_1h, bars_4h,
                ema_fast=ef, ema_slow=es,
                slope_period=sp, min_slope=ms,
            )
            n_sig = sig.isin(['long','short']).sum()
            if n_sig == 0:
                print(f"  [{label}] シグナルなし、スキップ")
                continue
            result = make_engine().run(data=bars_1h, signal_func=lambda b: sig,
                                       freq='1h', name=label)
            if result:
                r = extract(result, label, 'B')
                all_results.append(r)
                print(f"  [{label}] N={r['total_trades']}(sig={n_sig}) | "
                      f"WR={r['win_rate_pct']:.1f}% | PF={r['profit_factor']:.3f} | "
                      f"DD={r['max_drawdown_pct']:.1f}% | Ret={r['total_return_pct']:.2f}%")
        except Exception as e:
            print(f"  [{label}] エラー: {e}")
            import traceback; traceback.print_exc()

    # ---- ランキング ----
    print("\n=== 全戦略ランキング（上位8） ===")
    scored = sorted([(score(r), r) for r in all_results], key=lambda x: x[0], reverse=True)
    for i, (s, r) in enumerate(scored[:8]):
        print(f"  {i+1}. [{r['direction']}] {r['name']}: "
              f"score={s:.3f} | PF={r['profit_factor']:.3f} | "
              f"WR={r['win_rate_pct']:.1f}% | DD={r['max_drawdown_pct']:.1f}% | N={r['total_trades']}")

    # ---- 保存 ----
    fieldnames = ['name','direction','total_trades','win_rate_pct','profit_factor',
                  'max_drawdown_pct','total_return_pct','rr_ratio','passed']
    out_path = os.path.join(RESULTS_DIR, 'run003_summary.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nサマリー保存: {out_path}")

    # 最良戦略のトレード詳細保存
    if scored:
        best = scored[0][1]
        if best.get('trades'):
            tp = os.path.join(RESULTS_DIR, f"run003_best_trades_{best['name']}.csv")
            pd.DataFrame(best['trades']).to_csv(tp, index=False)
            print(f"最良トレード詳細: {tp}")
        with open(os.path.join(RESULTS_DIR, 'run003_best_name.txt'), 'w') as f:
            f.write(best['name'])

    print("\n完了。")
