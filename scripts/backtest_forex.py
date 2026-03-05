"""
RUN-011: FX通貨ペア横展開バックテスト + ストレステスト + PDCA
================================================================
OANDA API制約 (XAUUSD非対応) に対応し、為替ペアでの戦略有効性を検証。

対象ペア:
  - USDJPY (pip=0.01, 典型spread=0.3-0.5 pips)
  - EURUSD (pip=0.0001, 典型spread=0.1-0.3 pips)
  - GBPUSD (pip=0.0001, 典型spread=0.3-0.5 pips)
  - AUDUSD (pip=0.0001, 典型spread=0.3-0.5 pips)
  - GBPJPY (pip=0.01, 典型spread=0.5-1.0 pips, 4Hデータなし)

対象戦略:
  A. PA v2 (sig_yagami_pa_v2) — 4H足 ※4Hデータがあるペアのみ
  B. MTF v3 H1 (sig_yagami_mtf_v3_h1) — H1足 ※4Hデータ必須
  C. DC50+EMA200 (sig_maedai_d1_dc30) — D1足

検証項目:
  1. 全期間バックテスト (基準スプレッド)
  2. スプレッド負荷テスト (0.0 ~ 3.0 pips)
  3. 前半/後半分割 (生存者バイアス)
  4. 年別分解
  5. Walk-Forward (有望戦略のみ)
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

# ===== 通貨ペア定義 =====
PAIRS = {
    'USDJPY': {
        'pip': 0.01,
        'spread_typical': 0.3,
        'has_4h': True,
        'has_1d': True,
    },
    'EURUSD': {
        'pip': 0.0001,
        'spread_typical': 0.2,
        'has_4h': True,
        'has_1d': True,
    },
    'GBPUSD': {
        'pip': 0.0001,
        'spread_typical': 0.4,
        'has_4h': True,
        'has_1d': True,
    },
    'AUDUSD': {
        'pip': 0.0001,
        'spread_typical': 0.3,
        'has_4h': True,
        'has_1d': True,
    },
    'GBPJPY': {
        'pip': 0.01,
        'spread_typical': 0.8,
        'has_4h': False,  # 4Hデータなし
        'has_1d': True,
    },
}


def load(pair, tf):
    path = os.path.join(DATA_DIR, f'{pair}_{tf}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def fmt_res(res):
    """結果dictを1行サマリー文字列に整形"""
    if not res:
        return "No trades"
    n = res.get('total_trades', 0)
    if n == 0:
        return "No trades"
    pf = res.get('profit_factor', 0)
    wr = res.get('win_rate_pct', 0)
    mdd = res.get('max_drawdown_pct', 0)
    net = res.get('net_pnl', 0)
    return f"N={n}, PF={pf:.3f}, WR={wr:.1f}%, MDD={mdd:.1f}%, Net={net:,.0f}"


# ===== 1. 全期間バックテスト =====
def run_full_backtest(pair, pair_cfg, bars_4h, bars_1h, bars_1d):
    """全戦略の全期間バックテスト"""
    results = []
    pip = pair_cfg['pip']
    spread = pair_cfg['spread_typical']

    # --- PA v2 (4H) ---
    if pair_cfg['has_4h'] and bars_4h is not None and len(bars_4h) > 100:
        sig = sig_yagami_pa_v2(freq='4h', min_conditions=2,
                                align_tol=0.12, stop_tol=0.15)
        engine = BacktestEngine(
            init_cash=5_000_000, risk_pct=0.02,
            default_sl_atr=1.5, default_tp_atr=4.5,
            trail_start_atr=2.0, trail_dist_atr=1.5,
            exit_on_signal=True, slippage_pips=spread, pip=pip,
            target_min_trades=0, target_min_wr=0.0,
        )
        res = engine.run(data=bars_4h, signal_func=sig, freq='4h',
                         name=f'{pair}_PA_v2_4H')
        results.append({
            'pair': pair, 'strategy': 'PA_v2_4H', 'res': res,
            'n': res.get('total_trades', 0) if res else 0,
            'pf': res.get('profit_factor', 0) if res else 0,
            'wr': res.get('win_rate_pct', 0) if res else 0,
            'mdd': res.get('max_drawdown_pct', 0) if res else 0,
            'net_pnl': res.get('net_pnl', 0) if res else 0,
        })
        print(f"  PA v2 (4H): {fmt_res(res)}")

    # --- MTF v3 (H1, 4H環境) ---
    if pair_cfg['has_4h'] and bars_4h is not None and bars_1h is not None:
        if len(bars_4h) > 50 and len(bars_1h) > 200:
            sig = sig_yagami_mtf_v3_h1(bars_4h, rr_min=2.0, h4_lookback=15)
            engine = BacktestEngine(
                init_cash=5_000_000, risk_pct=0.02,
                default_sl_atr=1.0, default_tp_atr=6.0,
                trail_start_atr=3.0, trail_dist_atr=2.0,
                exit_on_signal=False, slippage_pips=spread, pip=pip,
                target_min_trades=0, target_min_wr=0.0,
            )
            res = engine.run(data=bars_1h, signal_func=sig, freq='1h',
                             name=f'{pair}_MTF_v3_H1')
            results.append({
                'pair': pair, 'strategy': 'MTF_v3_H1', 'res': res,
                'n': res.get('total_trades', 0) if res else 0,
                'pf': res.get('profit_factor', 0) if res else 0,
                'wr': res.get('win_rate_pct', 0) if res else 0,
                'mdd': res.get('max_drawdown_pct', 0) if res else 0,
                'net_pnl': res.get('net_pnl', 0) if res else 0,
            })
            print(f"  MTF v3 (H1): {fmt_res(res)}")

    # --- DC50+EMA200 (D1) ---
    if bars_1d is not None and len(bars_1d) > 250:
        sig = sig_maedai_d1_dc30(lookback=50, ema_period=200)
        engine = BacktestEngine(
            init_cash=5_000_000, risk_pct=0.05,
            default_sl_atr=0.8, default_tp_atr=10.0,
            trail_start_atr=4.0, trail_dist_atr=3.0,
            exit_on_signal=False, slippage_pips=spread, pip=pip,
            target_min_trades=0, target_min_wr=0.0,
        )
        res = engine.run(data=bars_1d, signal_func=sig, freq='1d',
                         name=f'{pair}_DC50_EMA200')
        results.append({
            'pair': pair, 'strategy': 'DC50_EMA200', 'res': res,
            'n': res.get('total_trades', 0) if res else 0,
            'pf': res.get('profit_factor', 0) if res else 0,
            'wr': res.get('win_rate_pct', 0) if res else 0,
            'mdd': res.get('max_drawdown_pct', 0) if res else 0,
            'net_pnl': res.get('net_pnl', 0) if res else 0,
        })
        print(f"  DC50+EMA200 (D1): {fmt_res(res)}")

    return results


# ===== 2. スプレッド負荷テスト =====
def run_spread_test(pair, pair_cfg, data, signal_func, freq, base_engine_params,
                    strategy_name, spread_list=[0.0, 0.3, 0.5, 1.0, 2.0, 3.0]):
    """スプレッドを変化させてロバスト性を確認"""
    results = []
    for spread in spread_list:
        params = base_engine_params.copy()
        params['slippage_pips'] = spread
        engine = BacktestEngine(**params)
        res = engine.run(data=data, signal_func=signal_func, freq=freq,
                         name=f'{pair}_{strategy_name}_sp{spread}')
        if res and res.get('total_trades', 0) > 0:
            results.append({
                'pair': pair, 'strategy': strategy_name,
                'spread_pips': spread,
                'n': res.get('total_trades', 0),
                'pf': res.get('profit_factor', 0),
                'wr': res.get('win_rate_pct', 0),
                'mdd': res.get('max_drawdown_pct', 0),
            })
        else:
            results.append({
                'pair': pair, 'strategy': strategy_name,
                'spread_pips': spread, 'n': 0, 'pf': 0, 'wr': 0, 'mdd': 0,
            })
    return results


# ===== 3. 前半/後半分割 =====
def run_half_split(pair, pair_cfg, data, signal_func, freq, engine_params,
                   strategy_name):
    """生存者バイアスチェック: 前半/後半のPF比較"""
    mid = len(data) // 2
    results = {}
    for label, d in [('full', data), ('1st_half', data.iloc[:mid]),
                     ('2nd_half', data.iloc[mid:])]:
        engine = BacktestEngine(**engine_params)
        res = engine.run(data=d, signal_func=signal_func, freq=freq,
                         name=f'{pair}_{strategy_name}_{label}')
        results[label] = {
            'pair': pair, 'strategy': strategy_name, 'period': label,
            'n': res.get('total_trades', 0) if res else 0,
            'pf': res.get('profit_factor', 0) if res else 0,
            'wr': res.get('win_rate_pct', 0) if res else 0,
            'mdd': res.get('max_drawdown_pct', 0) if res else 0,
        }
    return results


# ===== 4. 年別分解 =====
def run_annual(pair, pair_cfg, data, signal_func, freq, engine_params,
               strategy_name):
    """年別のPF/WR/MDD"""
    results = []
    for yr in sorted(data.index.year.unique()):
        yr_data = data[data.index.year == yr]
        if len(yr_data) < 20:
            continue
        engine = BacktestEngine(**engine_params)
        res = engine.run(data=yr_data, signal_func=signal_func, freq=freq,
                         name=f'{pair}_{strategy_name}_{yr}')
        results.append({
            'pair': pair, 'strategy': strategy_name, 'year': yr,
            'n': res.get('total_trades', 0) if res else 0,
            'pf': res.get('profit_factor', 0) if res else 0,
            'wr': res.get('win_rate_pct', 0) if res else 0,
            'mdd': res.get('max_drawdown_pct', 0) if res else 0,
        })
    return results


# ===== 5. Walk-Forward =====
def run_walk_forward(pair, pair_cfg, data, signal_func_factory, freq,
                     engine_params, strategy_name,
                     train_months=6, oos_months=3, step_months=3):
    """拡張ウィンドウ Walk-Forward"""
    data_start = data.index[0]
    data_end = data.index[-1]
    oos_start = data_start + pd.DateOffset(months=train_months)

    folds = []
    fold_num = 0
    while True:
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        if oos_end > data_end:
            if (data_end - oos_start).days >= 30:
                oos_end = data_end
            else:
                break
        folds.append({
            'fold': fold_num,
            'train_end': oos_start,
            'oos_start': oos_start,
            'oos_end': oos_end,
        })
        fold_num += 1
        oos_start += pd.DateOffset(months=step_months)
        if oos_start >= data_end:
            break

    results = []
    for fold in folds:
        train_data = data[data.index < fold['train_end']]
        oos_data = data[(data.index >= fold['oos_start']) &
                        (data.index < fold['oos_end'])]

        sig_is = signal_func_factory()
        sig_oos = signal_func_factory()

        engine_is = BacktestEngine(**engine_params)
        res_is = engine_is.run(data=train_data, signal_func=sig_is, freq=freq,
                               name=f'{pair}_{strategy_name}_IS_{fold["fold"]}')

        engine_oos = BacktestEngine(**engine_params)
        res_oos = engine_oos.run(data=oos_data, signal_func=sig_oos, freq=freq,
                                 name=f'{pair}_{strategy_name}_OOS_{fold["fold"]}')

        n_oos = res_oos.get('total_trades', 0) if res_oos else 0
        pf_oos = res_oos.get('profit_factor', 0) if res_oos else 0
        wr_oos = res_oos.get('win_rate_pct', 0) if res_oos else 0
        mdd_oos = res_oos.get('max_drawdown_pct', 0) if res_oos else 0
        passed = (pf_oos >= 1.2 and n_oos >= 3 and mdd_oos <= 20.0)

        results.append({
            'pair': pair, 'strategy': strategy_name,
            'fold': fold['fold'],
            'oos_start': fold['oos_start'].strftime('%Y-%m-%d'),
            'oos_end': fold['oos_end'].strftime('%Y-%m-%d'),
            'n_is': res_is.get('total_trades', 0) if res_is else 0,
            'pf_is': round(res_is.get('profit_factor', 0) if res_is else 0, 4),
            'n_oos': n_oos,
            'pf_oos': round(pf_oos, 4),
            'wr_oos': round(wr_oos, 2),
            'mdd_oos': round(mdd_oos, 2),
            'passed': passed,
        })

        mark = "PASS" if passed else "FAIL"
        print(f"    Fold {fold['fold']}: IS N={results[-1]['n_is']} PF={results[-1]['pf_is']:.2f} | "
              f"OOS N={n_oos} PF={pf_oos:.2f} WR={wr_oos:.1f}% MDD={mdd_oos:.1f}% [{mark}]")

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_full = []
    all_spread = []
    all_half = []
    all_annual = []
    all_wf = []

    for pair, cfg in PAIRS.items():
        print(f"\n{'='*70}")
        print(f"  {pair} (pip={cfg['pip']}, spread={cfg['spread_typical']})")
        print(f"{'='*70}")

        # データロード
        bars_4h = load(pair, '4h')
        bars_1h = load(pair, '1h')
        bars_1d = load(pair, '1d')

        if bars_4h is not None:
            print(f"  4H: {bars_4h.index[0].date()} -> {bars_4h.index[-1].date()}, {len(bars_4h)} bars")
        if bars_1h is not None:
            print(f"  1H: {bars_1h.index[0].date()} -> {bars_1h.index[-1].date()}, {len(bars_1h)} bars")
        if bars_1d is not None:
            print(f"  D1: {bars_1d.index[0].date()} -> {bars_1d.index[-1].date()}, {len(bars_1d)} bars")

        # ===== 1. 全期間バックテスト =====
        print(f"\n--- 全期間バックテスト ---")
        full_res = run_full_backtest(pair, cfg, bars_4h, bars_1h, bars_1d)
        all_full.extend(full_res)

        # 以降の詳細分析は各戦略ごとに実行
        pip = cfg['pip']
        spread = cfg['spread_typical']

        # === PA v2 (4H) 詳細分析 ===
        if cfg['has_4h'] and bars_4h is not None and len(bars_4h) > 100:
            sig_factory = lambda: sig_yagami_pa_v2(freq='4h', min_conditions=2,
                                                    align_tol=0.12, stop_tol=0.15)
            ep = dict(
                init_cash=5_000_000, risk_pct=0.02,
                default_sl_atr=1.5, default_tp_atr=4.5,
                trail_start_atr=2.0, trail_dist_atr=1.5,
                exit_on_signal=True, slippage_pips=spread, pip=pip,
                target_min_trades=0, target_min_wr=0.0,
            )
            sname = 'PA_v2_4H'

            # スプレッド
            print(f"\n  --- {sname} スプレッド負荷テスト ---")
            sp = run_spread_test(pair, cfg, bars_4h, sig_factory(), '4h', ep, sname)
            all_spread.extend(sp)
            for r in sp:
                if r['n'] > 0:
                    print(f"    Spread={r['spread_pips']}: PF={r['pf']:.3f}, N={r['n']}")

            # 前半/後半
            hf = run_half_split(pair, cfg, bars_4h, sig_factory(), '4h', ep, sname)
            for k, v in hf.items():
                all_half.append(v)
                print(f"    {k}: PF={v['pf']:.3f}, N={v['n']}")

            # 年別
            ann = run_annual(pair, cfg, bars_4h, sig_factory(), '4h', ep, sname)
            all_annual.extend(ann)

            # Walk-Forward
            print(f"\n  --- {sname} Walk-Forward ---")
            wf = run_walk_forward(pair, cfg, bars_4h, sig_factory, '4h', ep, sname)
            all_wf.extend(wf)

        # === MTF v3 (H1) 詳細分析 ===
        if cfg['has_4h'] and bars_4h is not None and bars_1h is not None:
            if len(bars_4h) > 50 and len(bars_1h) > 200:
                # lambdaでキャプチャするために変数をバインド
                _bars_4h = bars_4h
                sig_factory = lambda: sig_yagami_mtf_v3_h1(_bars_4h, rr_min=2.0,
                                                            h4_lookback=15)
                ep = dict(
                    init_cash=5_000_000, risk_pct=0.02,
                    default_sl_atr=1.0, default_tp_atr=6.0,
                    trail_start_atr=3.0, trail_dist_atr=2.0,
                    exit_on_signal=False, slippage_pips=spread, pip=pip,
                    target_min_trades=0, target_min_wr=0.0,
                )
                sname = 'MTF_v3_H1'

                # スプレッド
                print(f"\n  --- {sname} スプレッド負荷テスト ---")
                sp = run_spread_test(pair, cfg, bars_1h, sig_factory(), '1h', ep, sname)
                all_spread.extend(sp)
                for r in sp:
                    if r['n'] > 0:
                        print(f"    Spread={r['spread_pips']}: PF={r['pf']:.3f}, N={r['n']}")

                # 前半/後半
                hf = run_half_split(pair, cfg, bars_1h, sig_factory(), '1h', ep, sname)
                for k, v in hf.items():
                    all_half.append(v)

                # 年別
                ann = run_annual(pair, cfg, bars_1h, sig_factory(), '1h', ep, sname)
                all_annual.extend(ann)

                # Walk-Forward
                print(f"\n  --- {sname} Walk-Forward ---")
                wf = run_walk_forward(pair, cfg, bars_1h, sig_factory, '1h', ep, sname)
                all_wf.extend(wf)

        # === DC50+EMA200 (D1) 詳細分析 ===
        if bars_1d is not None and len(bars_1d) > 250:
            sig_factory = lambda: sig_maedai_d1_dc30(lookback=50, ema_period=200)
            ep = dict(
                init_cash=5_000_000, risk_pct=0.05,
                default_sl_atr=0.8, default_tp_atr=10.0,
                trail_start_atr=4.0, trail_dist_atr=3.0,
                exit_on_signal=False, slippage_pips=spread, pip=pip,
                target_min_trades=0, target_min_wr=0.0,
            )
            sname = 'DC50_EMA200'

            # スプレッド (D1足はスプレッド影響小さいが念のため)
            print(f"\n  --- {sname} スプレッド負荷テスト ---")
            sp = run_spread_test(pair, cfg, bars_1d, sig_factory(), '1d', ep, sname,
                                 spread_list=[0.0, 0.5, 1.0, 2.0])
            all_spread.extend(sp)
            for r in sp:
                if r['n'] > 0:
                    print(f"    Spread={r['spread_pips']}: PF={r['pf']:.3f}, N={r['n']}")

            # 年別
            ann = run_annual(pair, cfg, bars_1d, sig_factory(), '1d', ep, sname)
            all_annual.extend(ann)

            # Walk-Forward (D1は長期ウィンドウ)
            print(f"\n  --- {sname} Walk-Forward ---")
            wf = run_walk_forward(pair, cfg, bars_1d, sig_factory, '1d', ep, sname,
                                  train_months=24, oos_months=6, step_months=6)
            all_wf.extend(wf)

    # ===== CSV保存 =====
    # 全期間結果
    full_rows = [{k: v for k, v in r.items() if k != 'res'} for r in all_full]
    pd.DataFrame(full_rows).to_csv(
        os.path.join(RESULTS_DIR, 'run011_forex_full.csv'), index=False)

    # スプレッド
    pd.DataFrame(all_spread).to_csv(
        os.path.join(RESULTS_DIR, 'run011_forex_spread.csv'), index=False)

    # 前半/後半
    pd.DataFrame(all_half).to_csv(
        os.path.join(RESULTS_DIR, 'run011_forex_halfsplit.csv'), index=False)

    # 年別
    pd.DataFrame(all_annual).to_csv(
        os.path.join(RESULTS_DIR, 'run011_forex_annual.csv'), index=False)

    # Walk-Forward
    pd.DataFrame(all_wf).to_csv(
        os.path.join(RESULTS_DIR, 'run011_forex_wf.csv'), index=False)

    print(f"\n{'='*70}")
    print(f"  全結果を {RESULTS_DIR}/run011_forex_*.csv に保存")
    print(f"{'='*70}")

    # ===== サマリー出力 =====
    print(f"\n{'='*70}")
    print(f"  全期間サマリー")
    print(f"{'='*70}")
    df_full = pd.DataFrame(full_rows)
    for _, row in df_full.iterrows():
        grade = ''
        if row['n'] >= 30 and row['pf'] >= 1.5:
            grade = '★★★'
        elif row['n'] >= 10 and row['pf'] >= 1.2:
            grade = '★★'
        elif row['n'] >= 5 and row['pf'] >= 1.0:
            grade = '★'
        else:
            grade = '×'
        print(f"  {row['pair']:8s} {row['strategy']:15s} N={row['n']:4d} "
              f"PF={row['pf']:6.3f} WR={row['wr']:5.1f}% MDD={row['mdd']:5.1f}% "
              f"[{grade}]")

    # WF サマリー
    if all_wf:
        print(f"\n{'='*70}")
        print(f"  Walk-Forward サマリー")
        print(f"{'='*70}")
        df_wf = pd.DataFrame(all_wf)
        for (p, s), grp in df_wf.groupby(['pair', 'strategy']):
            valid = grp[grp['n_oos'] >= 3]
            if len(valid) == 0:
                print(f"  {p:8s} {s:15s} 有効fold=0")
                continue
            n_pass = int(valid['passed'].sum())
            avg_pf = valid['pf_oos'].mean()
            print(f"  {p:8s} {s:15s} 合格={n_pass}/{len(valid)} "
                  f"OOS平均PF={avg_pf:.3f}")

    return {
        'full': all_full, 'spread': all_spread,
        'half': all_half, 'annual': all_annual, 'wf': all_wf,
    }


if __name__ == '__main__':
    main()
