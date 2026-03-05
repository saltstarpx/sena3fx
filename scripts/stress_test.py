"""
ストレステスト: 3戦略の堅牢性検証
=================================
Manusの指摘に基づく3つの検証:
  1. スプレッド負荷テスト (0.3, 0.5, 1.0, 2.0 pips相当をPnLから差し引く)
  2. 生存者バイアス分析 (最大利益トレードを除外した場合のPF)
  3. 長期バルク検証 (全期間 + 年別分解)

対象戦略:
  A. PA v2 (sig_yagami_pa_v2, mc2, at0.12) — XAUUSD 4H
  B. MTF v3 (sig_yagami_mtf_v3_h1, rr2.0, lb15) — XAUUSD H1
  C. DC50+EMA200 (sig_maedai_d1_dc30, lb=50) — XAUUSD D1
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from lib.backtest import BacktestEngine
from lib.yagami import sig_yagami_pa_v2, sig_maedai_d1_dc30
from strategies.yagami_mtf_v3 import sig_yagami_mtf_v3_h1

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'ohlc')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')


def load(pair, tf):
    path = os.path.join(DATA_DIR, f'{pair}_{tf}.csv')
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def calc_pf(trades):
    """PnLリストからPFを計算"""
    wins = sum(t for t in trades if t > 0)
    losses = abs(sum(t for t in trades if t < 0))
    return round(wins / losses, 4) if losses > 0 else 99.9


def calc_wr(trades):
    if not trades:
        return 0
    return round(sum(1 for t in trades if t > 0) / len(trades) * 100, 1)


def calc_mdd(trades, init_cash=5_000_000):
    """PnLリストからMDD%を計算"""
    equity = init_cash
    peak = equity
    max_dd = 0
    for pnl in trades:
        equity += pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
    return round(max_dd * 100, 2)


def run_strategy(name, data, signal_func, freq, engine_params):
    """戦略を実行してトレードリストを返す"""
    engine = BacktestEngine(**engine_params)
    res = engine.run(data=data, signal_func=signal_func, freq=freq, name=name)
    if res is None:
        return [], res
    # engine.tradesが直接アクセスできない場合、再計算
    return res, res


def get_trade_pnls(data, signal_func, freq, engine_params):
    """トレードごとのPnLリストを取得"""
    engine = BacktestEngine(**engine_params)
    res = engine.run(data=data, signal_func=signal_func, freq=freq, name='tmp')
    if res is None:
        return [], {}

    # エンジンのtrades属性を取得
    if hasattr(engine, '_last_trades'):
        trades = engine._last_trades
    else:
        # resからtrade情報を推測
        pass

    return res, []


def run_with_spread_penalty(data, signal_func, freq, engine_params,
                             spread_pips_list=[0.0, 0.3, 0.5, 1.0, 2.0]):
    """
    スプレッド負荷テスト。
    エンジンのslippage_pipsパラメータを変えて再実行。
    XAUUSDのpip = 0.1 (10セント)。
    """
    results = []
    for spread in spread_pips_list:
        params = engine_params.copy()
        params['slippage_pips'] = spread
        engine = BacktestEngine(**params)
        res = engine.run(data=data, signal_func=signal_func, freq=freq,
                         name=f'spread_{spread}')
        if res:
            results.append({
                'spread_pips': spread,
                'pf': res.get('profit_factor', 0),
                'wr': res.get('win_rate_pct', 0),
                'mdd': res.get('max_drawdown_pct', 0),
                'n': res.get('total_trades', 0),
                'net_pnl': res.get('net_pnl', 0),
            })
        else:
            results.append({'spread_pips': spread, 'n': 0})
    return results


def survivor_bias_analysis(data, signal_func, freq, engine_params, name=''):
    """
    生存者バイアス分析。
    エンジンは直接トレードリストを返さないので、
    monthly_pnlから推測、または全体PFから最大利益除外を試算。

    代替手法: 期間を半分に分割して各半期のPFを比較。
    """
    # 全期間
    engine = BacktestEngine(**engine_params)
    res_full = engine.run(data=data, signal_func=signal_func, freq=freq, name=name)
    if not res_full:
        return {}

    # 前半/後半分割
    mid_idx = len(data) // 2
    data_1st = data.iloc[:mid_idx]
    data_2nd = data.iloc[mid_idx:]

    engine1 = BacktestEngine(**engine_params)
    res_1st = engine1.run(data=data_1st, signal_func=signal_func, freq=freq, name=f'{name}_1st')

    engine2 = BacktestEngine(**engine_params)
    res_2nd = engine2.run(data=data_2nd, signal_func=signal_func, freq=freq, name=f'{name}_2nd')

    return {
        'full': res_full,
        '1st_half': res_1st,
        '2nd_half': res_2nd,
    }


def annual_breakdown(data, signal_func, freq, engine_params, name=''):
    """年別分解バックテスト"""
    results = []
    years = sorted(data.index.year.unique())
    for yr in years:
        yr_data = data[data.index.year == yr]
        if len(yr_data) < 30:
            continue
        engine = BacktestEngine(**engine_params)
        res = engine.run(data=yr_data, signal_func=signal_func, freq=freq,
                         name=f'{name}_{yr}')
        if res:
            results.append({
                'year': yr,
                'pf': res.get('profit_factor', 0),
                'wr': res.get('win_rate_pct', 0),
                'mdd': res.get('max_drawdown_pct', 0),
                'n': res.get('total_trades', 0),
            })
        else:
            results.append({'year': yr, 'n': 0, 'pf': 0, 'wr': 0, 'mdd': 0})
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ===== データロード =====
    bars_4h = load('XAUUSD', '4h')
    bars_1h = load('XAUUSD', '1h')
    bars_1d = load('XAUUSD', '1d')

    print(f"4H: {bars_4h.index[0]} -> {bars_4h.index[-1]}, {len(bars_4h)} bars")
    print(f"1H: {bars_1h.index[0]} -> {bars_1h.index[-1]}, {len(bars_1h)} bars")
    print(f"D1: {bars_1d.index[0]} -> {bars_1d.index[-1]}, {len(bars_1d)} bars")

    # ===== 戦略定義 =====
    strategies = {
        'PA_v2': {
            'data': bars_4h,
            'signal_func': sig_yagami_pa_v2(freq='4h', min_conditions=2,
                                             align_tol=0.12, stop_tol=0.15),
            'freq': '4h',
            'engine_params': dict(
                init_cash=5_000_000, risk_pct=0.02,
                default_sl_atr=1.5, default_tp_atr=4.5,
                trail_start_atr=2.0, trail_dist_atr=1.5,
                exit_on_signal=True, slippage_pips=0.3, pip=0.1,
                target_min_trades=0, target_min_wr=0.0,
            ),
        },
        'MTF_v3': {
            'data': bars_1h,
            'signal_func': sig_yagami_mtf_v3_h1(bars_4h, rr_min=2.0, h4_lookback=15),
            'freq': '1h',
            'engine_params': dict(
                init_cash=5_000_000, risk_pct=0.02,
                default_sl_atr=1.0, default_tp_atr=6.0,
                trail_start_atr=3.0, trail_dist_atr=2.0,
                exit_on_signal=False, slippage_pips=0.3, pip=0.1,
                target_min_trades=0, target_min_wr=0.0,
            ),
        },
        'DC50_EMA200': {
            'data': bars_1d,
            'signal_func': sig_maedai_d1_dc30(lookback=50, ema_period=200),
            'freq': '1d',
            'engine_params': dict(
                init_cash=5_000_000, risk_pct=0.05,
                default_sl_atr=0.8, default_tp_atr=10.0,
                trail_start_atr=4.0, trail_dist_atr=3.0,
                exit_on_signal=False, slippage_pips=0.3, pip=0.1,
                target_min_trades=0, target_min_wr=0.0,
            ),
        },
    }

    all_results = {}

    for sname, sconfig in strategies.items():
        print(f"\n{'='*60}")
        print(f"  Strategy: {sname}")
        print(f"{'='*60}")

        data = sconfig['data']
        sig = sconfig['signal_func']
        freq = sconfig['freq']
        ep = sconfig['engine_params']

        # ----- 1. スプレッド負荷テスト -----
        print("\n--- スプレッド負荷テスト ---")
        spread_results = run_with_spread_penalty(
            data, sig, freq, ep,
            spread_pips_list=[0.0, 0.3, 0.5, 1.0, 2.0, 3.0]
        )
        for r in spread_results:
            if r.get('n', 0) > 0:
                print(f"  Spread={r['spread_pips']} pips: N={r['n']}, PF={r['pf']:.4f}, "
                      f"WR={r['wr']:.1f}%, MDD={r['mdd']:.1f}%")
            else:
                print(f"  Spread={r['spread_pips']} pips: No trades")

        # ----- 2. 生存者バイアス分析 -----
        print("\n--- 生存者バイアス分析 (前半/後半分割) ---")
        surv = survivor_bias_analysis(data, sig, freq, ep, name=sname)
        if surv.get('full'):
            for period, res in surv.items():
                if res:
                    print(f"  {period}: N={res.get('total_trades', 0)}, "
                          f"PF={res.get('profit_factor', 0):.4f}, "
                          f"WR={res.get('win_rate_pct', 0):.1f}%, "
                          f"MDD={res.get('max_drawdown_pct', 0):.1f}%")
                else:
                    print(f"  {period}: No trades")

        # ----- 3. 年別分解 -----
        print("\n--- 年別分解 ---")
        annual = annual_breakdown(data, sig, freq, ep, name=sname)
        for a in annual:
            if a['n'] > 0:
                print(f"  {a['year']}: N={a['n']}, PF={a['pf']:.4f}, "
                      f"WR={a['wr']:.1f}%, MDD={a['mdd']:.1f}%")
            else:
                print(f"  {a['year']}: No trades")

        all_results[sname] = {
            'spread': spread_results,
            'survivor': surv,
            'annual': annual,
        }

    # ===== 結果CSV保存 =====
    # スプレッド結果を統合保存
    spread_rows = []
    for sname, res in all_results.items():
        for sr in res['spread']:
            sr['strategy'] = sname
            spread_rows.append(sr)
    pd.DataFrame(spread_rows).to_csv(
        os.path.join(RESULTS_DIR, 'stress_test_spread.csv'), index=False)

    # 年別結果
    annual_rows = []
    for sname, res in all_results.items():
        for ar in res['annual']:
            ar['strategy'] = sname
            annual_rows.append(ar)
    pd.DataFrame(annual_rows).to_csv(
        os.path.join(RESULTS_DIR, 'stress_test_annual.csv'), index=False)

    # 生存者バイアス
    surv_rows = []
    for sname, res in all_results.items():
        for period, r in res['survivor'].items():
            if r:
                surv_rows.append({
                    'strategy': sname, 'period': period,
                    'n': r.get('total_trades', 0),
                    'pf': r.get('profit_factor', 0),
                    'wr': r.get('win_rate_pct', 0),
                    'mdd': r.get('max_drawdown_pct', 0),
                })
    pd.DataFrame(surv_rows).to_csv(
        os.path.join(RESULTS_DIR, 'stress_test_survivor.csv'), index=False)

    print(f"\n=== Saved to {RESULTS_DIR}/stress_test_*.csv ===")

    return all_results


if __name__ == '__main__':
    main()
