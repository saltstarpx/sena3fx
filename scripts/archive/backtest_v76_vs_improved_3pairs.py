"""
v76 vs v76_improved 3通貨ペア対決バックテスト
===============================================
対象: USDJPY, EURUSD, GBPUSD
期間: 2025/1/1 ~ 2025/12/31
スプレッド: 0.4 pips
資金: 300万円
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'ohlc')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 設定 ──
PAIRS = ['USDJPY', 'EURUSD', 'GBPUSD']
SPREAD_PIPS = 0.4
INIT_CAPITAL = 3_000_000
START = '2025-01-01'
END = '2025-12-31'

# pip変換: 1pip = 0.01 (JPYペア), 0.0001 (USDペア)
# lot_size: 1ロット=10万通貨
PIP_VALUES = {
    'USDJPY': {'pip': 0.01, 'lot_size': 100_000, 'base_ccy': 'JPY'},
    'EURUSD': {'pip': 0.0001, 'lot_size': 100_000, 'base_ccy': 'USD'},
    'GBPUSD': {'pip': 0.0001, 'lot_size': 100_000, 'base_ccy': 'USD'},
}


def load_csv(path):
    """CSV読み込み（Yahoo/OANDA両形式対応）"""
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    # volumeがない場合追加
    if 'volume' not in df.columns:
        df['volume'] = 0
    return df


def load_pair_data(pair):
    """ペアの1h/4hデータを読み込み、期間フィルター"""
    path_1h = os.path.join(DATA_DIR, f'{pair}_1h.csv')
    path_4h = os.path.join(DATA_DIR, f'{pair}_4h.csv')

    if not os.path.exists(path_1h) or not os.path.exists(path_4h):
        return None, None

    data_1h = load_csv(path_1h)
    data_4h = load_csv(path_4h)

    # 期間フィルター（4hはインジケーター計算のため少し前から取得）
    data_1h_full = data_1h[data_1h.index >= '2024-06-01']
    data_4h_full = data_4h[data_4h.index >= '2024-06-01']

    return data_1h_full, data_4h_full


def evaluate_signals(signals, data_1h, start_date, end_date, pip_val=0.01, lot_size=100_000):
    """
    シグナルの損益をシミュレーション。
    同時ポジションなし（1トレードずつ順次）。

    pnl計算: (exit - ep) * dir * lot_size で円換算（JPYペア直接、USDペアは近似×150）
    1ロット固定。
    """
    results = []
    trade_exit_time = None

    for sig in signals:
        t = sig["time"]

        if t < pd.Timestamp(start_date, tz='UTC') or t > pd.Timestamp(end_date, tz='UTC'):
            continue

        if trade_exit_time is not None and t <= trade_exit_time:
            continue

        ep = sig["ep"]
        sl = sig["sl"]
        tp = sig["tp"]
        d = sig["dir"]

        future = data_1h[data_1h.index > t]
        if len(future) == 0:
            continue

        pnl_raw = None
        exit_time = None
        exit_reason = None

        for _, bar in future.iterrows():
            if bar.name > pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=7):
                break

            if d == 1:  # ロング
                if bar["low"] <= sl:
                    pnl_raw = (sl - ep) * d
                    exit_reason = "SL"
                    exit_time = bar.name
                    break
                if bar["high"] >= tp:
                    pnl_raw = (tp - ep) * d
                    exit_reason = "TP"
                    exit_time = bar.name
                    break
            else:  # ショート
                if bar["high"] >= sl:
                    pnl_raw = (sl - ep) * d
                    exit_reason = "SL"
                    exit_time = bar.name
                    break
                if bar["low"] <= tp:
                    pnl_raw = (tp - ep) * d
                    exit_reason = "TP"
                    exit_time = bar.name
                    break

        if pnl_raw is not None:
            # pips計算
            pnl_pips = pnl_raw / pip_val
            # 円換算 (1ロット): JPYペアはそのまま、USDペアは×150（USDJPY近似）
            pnl_jpy = pnl_raw * lot_size
            if pip_val == 0.0001:  # USDペア → USD建て損益を円換算
                pnl_jpy *= 150  # USDJPY≒150円近似

            trade_exit_time = exit_time
            results.append({
                "time": t,
                "dir": "long" if d == 1 else "short",
                "tf": sig["tf"],
                "pattern": sig["pattern"],
                "ep": round(ep, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "risk_pips": round(sig["risk"] / pip_val, 1),
                "pnl_pips": round(pnl_pips, 1),
                "pnl": round(pnl_jpy, 0),
                "exit_reason": exit_reason,
                "exit_time": exit_time,
            })

    return results


def calc_metrics(results, init_capital):
    """メトリクス計算"""
    if not results:
        return {
            'trades': 0, 'wins': 0, 'losses': 0,
            'wr': 0, 'pf': 0, 'net_pnl': 0,
            'avg_pnl': 0, 'max_win': 0, 'max_loss': 0,
            'mdd': 0, 'ret_pct': 0,
        }

    df = pd.DataFrame(results)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]

    gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # ドローダウン計算
    equity = init_capital + df['pnl'].cumsum()
    peak = equity.cummax()
    dd = (peak - equity) / peak * 100
    mdd = dd.max()

    net_pnl = df['pnl'].sum()

    return {
        'trades': len(df),
        'wins': len(wins),
        'losses': len(losses),
        'wr': len(wins) / len(df) * 100,
        'pf': pf,
        'net_pnl': net_pnl,
        'avg_pnl': df['pnl'].mean(),
        'max_win': df['pnl'].max(),
        'max_loss': df['pnl'].min(),
        'mdd': mdd,
        'ret_pct': net_pnl / init_capital * 100,
    }


def print_metrics_table(pair, m_v76, m_imp):
    """2戦略の比較テーブル"""
    print(f"\n{'─'*70}")
    print(f"  {pair}")
    print(f"{'─'*70}")
    print(f"  {'指標':<16s} {'v76':>14s} {'v76改':>14s} {'差分':>14s}")
    print(f"  {'─'*58}")

    rows = [
        ('トレード数', 'trades', 'd'),
        ('勝ち / 負け', None, None),
        ('勝率 (%)', 'wr', '.1f'),
        ('PF', 'pf', '.3f'),
        ('純損益', 'net_pnl', ',.0f'),
        ('平均損益', 'avg_pnl', ',.0f'),
        ('最大利益', 'max_win', ',.0f'),
        ('最大損失', 'max_loss', ',.0f'),
        ('MDD (%)', 'mdd', '.2f'),
        ('リターン (%)', 'ret_pct', '.2f'),
    ]

    for label, key, fmt in rows:
        if key is None:
            v76_str = f"{m_v76['wins']}/{m_v76['losses']}"
            imp_str = f"{m_imp['wins']}/{m_imp['losses']}"
            print(f"  {label:<16s} {v76_str:>14s} {imp_str:>14s}")
            continue

        v76_val = m_v76[key]
        imp_val = m_imp[key]
        diff = imp_val - v76_val

        print(f"  {label:<16s} {v76_val:>14{fmt}} {imp_val:>14{fmt}} {diff:>+14{fmt}}")


def main():
    from strategies.current.yagami_mtf_v76 import generate_signals as gen_v76
    from strategies.current.yagami_mtf_v76_improved import generate_signals as gen_improved

    all_v76 = []
    all_imp = []
    all_metrics = {}

    print("=" * 70)
    print("  v76 vs v76改 3通貨ペア対決バックテスト")
    print(f"  期間: {START} ~ {END}")
    print(f"  スプレッド: {SPREAD_PIPS} pips / 資金: {INIT_CAPITAL:,}円")
    print("=" * 70)

    for pair in PAIRS:
        print(f"\n  {pair} データ読み込み中...")
        data_1h, data_4h = load_pair_data(pair)
        if data_1h is None:
            print(f"  {pair}: データなし、スキップ")
            continue

        print(f"  1H: {len(data_1h)}本, 4H: {len(data_4h)}本")
        print(f"  期間: {data_1h.index[0]} ~ {data_1h.index[-1]}")

        # 15分足がないため1hを15分足代わりに使用
        data_15m_proxy = data_1h.copy()
        # 1分足がないため1hを1分足代わりに使用
        data_1m_proxy = data_1h.copy()

        pip_info = PIP_VALUES.get(pair, {'pip': 0.0001, 'lot_size': 100_000, 'base_ccy': 'USD'})
        pip_val = pip_info['pip']
        lot_size = pip_info['lot_size']

        # v76はspread = spread_pips * 0.01 とハードコード（JPYペア前提）
        # USDペア(pip=0.0001)では spread_pips を補正する
        if pip_val == 0.0001:
            adjusted_spread_pips = SPREAD_PIPS * 0.01  # 0.4→0.004
        else:
            adjusted_spread_pips = SPREAD_PIPS  # JPYペアはそのまま

        # ── v76 ──
        print(f"  v76 シグナル生成中...")
        signals_v76 = gen_v76(data_1m_proxy, data_15m_proxy, data_4h,
                              spread_pips=adjusted_spread_pips, rr_ratio=2.5)
        results_v76 = evaluate_signals(signals_v76, data_1h, START, END, pip_val, lot_size)
        m_v76 = calc_metrics(results_v76, INIT_CAPITAL)

        # ── v76 improved ──
        print(f"  v76改 シグナル生成中...")
        signals_imp = gen_improved(data_1m_proxy, data_15m_proxy, data_4h,
                                   spread_pips=adjusted_spread_pips, rr_ratio=3.0)
        results_imp = evaluate_signals(signals_imp, data_1h, START, END, pip_val, lot_size)
        m_imp = calc_metrics(results_imp, INIT_CAPITAL)

        print_metrics_table(pair, m_v76, m_imp)
        all_metrics[pair] = {'v76': m_v76, 'improved': m_imp}
        all_v76.extend(results_v76)
        all_imp.extend(results_imp)

    # ── 全体サマリー ──
    m_total_v76 = calc_metrics(all_v76, INIT_CAPITAL)
    m_total_imp = calc_metrics(all_imp, INIT_CAPITAL)
    print_metrics_table("合計 (3ペア)", m_total_v76, m_total_imp)

    # ── 判定 ──
    print(f"\n{'='*70}")
    print("  判定")
    print(f"{'='*70}")

    scores = {'v76': 0, 'improved': 0}
    criteria = [
        ('PF', 'pf', True),
        ('勝率', 'wr', True),
        ('平均損益', 'avg_pnl', True),
        ('MDD', 'mdd', False),  # 低い方が良い
        ('リターン', 'ret_pct', True),
    ]
    for label, key, higher_better in criteria:
        v = m_total_v76[key]
        i = m_total_imp[key]
        if higher_better:
            winner = 'improved' if i > v else 'v76'
        else:
            winner = 'improved' if i < v else 'v76'
        scores[winner] += 1
        mark_v76 = ' <<' if winner == 'v76' else ''
        mark_imp = ' <<' if winner == 'improved' else ''
        print(f"  {label:<10s}: v76={v:>10.3f}{mark_v76}  v76改={i:>10.3f}{mark_imp}")

    print(f"\n  スコア: v76 {scores['v76']} - {scores['improved']} v76改")
    if scores['improved'] > scores['v76']:
        print("  >>> v76改 の勝利!")
    elif scores['v76'] > scores['improved']:
        print("  >>> v76 の勝利!")
    else:
        print("  >>> 引き分け!")

    # ── CSV保存 ──
    rows = []
    for pair, metrics in all_metrics.items():
        for strategy, m in metrics.items():
            rows.append({'pair': pair, 'strategy': strategy, **m})
    rows.append({'pair': 'TOTAL', 'strategy': 'v76', **m_total_v76})
    rows.append({'pair': 'TOTAL', 'strategy': 'improved', **m_total_imp})
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'v76_vs_improved_3pairs.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\n  結果CSV: {out_path}")


if __name__ == '__main__':
    main()
