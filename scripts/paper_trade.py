"""
ペーパートレード（模擬取引）ランナー
======================================
実際の注文は出さずに、最新データに対してシグナルを生成・記録する。
バックテスト結果と比較することで戦略の実環境での乖離を計測する。

使い方:
  # 1回実行（cronやスケジューラーから呼ぶ）
  python scripts/paper_trade.py

  # ループ実行（60秒ごと）
  python scripts/paper_trade.py --loop --interval 60

ログ:
  trade_logs/paper_YYYYMM.csv  — シグナルと仮想PnL
  trade_logs/paper_state.json  — 現在の仮想ポジション
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime, timezone

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from lib.yagami import yagami_signal
from lib.backtest import BacktestEngine

LOG_DIR = os.path.join(BASE_DIR, 'trade_logs')
STATE_FILE = os.path.join(LOG_DIR, 'paper_state.json')
os.makedirs(LOG_DIR, exist_ok=True)


# ---- 設定 ----
STRATEGY_FREQ = '1h'
MIN_GRADE = 'B'          # 'A' にすると厳選エントリーのみ
INIT_CASH = 5_000_000
RISK_PCT = 0.02
PIP = 0.1
SLIPPAGE = 0.3 * PIP


# ---- 状態管理 ----

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'cash': INIT_CASH,
        'position': None,
        'total_trades': 0,
        'wins': 0,
        'losses': 0,
        'total_pnl': 0.0,
        'peak_cash': INIT_CASH,
        'max_dd': 0.0,
        'last_run': None,
    }


def save_state(state: dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def log_trade(trade: dict):
    now = datetime.now(timezone.utc)
    log_path = os.path.join(LOG_DIR, f"paper_{now.strftime('%Y%m')}.csv")
    row = pd.DataFrame([trade])
    if os.path.exists(log_path):
        row.to_csv(log_path, mode='a', header=False, index=False)
    else:
        row.to_csv(log_path, index=False)


# ---- データ取得 ----

def get_latest_bars(freq: str = '1h', n_bars: int = 200) -> pd.DataFrame:
    """
    利用可能なデータソースから最新のOHLCバーを取得。
    ティックデータがあればそれを使い、なければサンプルデータ。
    """
    try:
        from scripts.fetch_data import load_ticks, ticks_to_ohlc
        ticks = load_ticks()
        if ticks is not None and len(ticks) > 1000:
            bars = ticks_to_ohlc(ticks, freq)
            return bars.iloc[-n_bars:]
    except Exception:
        pass

    # フォールバック: サンプルデータ（テスト用）
    from scripts.fetch_data import generate_sample_ohlc
    return generate_sample_ohlc(n_bars, freq)


# ---- SL/TP 計算（BacktestEngineと同じロジック） ----

def _calc_atr(bars: pd.DataFrame, period: int = 14) -> float:
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr_series = pd.Series(tr).rolling(period).mean()
    return float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 2.0


def _calc_sl_tp(bars: pd.DataFrame, direction: str, atr: float) -> tuple:
    """スイングロー/ハイベースのSL/TP計算"""
    close = float(bars['close'].iloc[-1])
    lookback = bars.iloc[-10:]

    if direction == 'long':
        sl_price = float(lookback['low'].min()) - SLIPPAGE
        sl_dist = close - sl_price
        if sl_dist < atr * 0.5:
            sl_dist = atr * 2.0
            sl_price = close - sl_dist
        tp_price = close + sl_dist * 2.0
        entry_price = close + SLIPPAGE
    else:
        sl_price = float(lookback['high'].max()) + SLIPPAGE
        sl_dist = sl_price - close
        if sl_dist < atr * 0.5:
            sl_dist = atr * 2.0
            sl_price = close + sl_dist
        tp_price = close - sl_dist * 2.0
        entry_price = close - SLIPPAGE

    return entry_price, sl_price, tp_price, sl_dist


# ---- ポジション更新 ----

def update_position(state: dict, bars: pd.DataFrame) -> dict:
    """現在ポジションのSL/TP判定（最新バーで）"""
    pos = state['position']
    if pos is None:
        return state

    last = bars.iloc[-1]

    if pos['dir'] == 'long':
        if float(last['low']) <= pos['sl']:
            exit_price = pos['sl']
            exit_reason = 'stop_loss'
        elif float(last['high']) >= pos['tp']:
            exit_price = pos['tp']
            exit_reason = 'take_profit'
        else:
            return state  # ポジション継続
    else:
        if float(last['high']) >= pos['sl']:
            exit_price = pos['sl']
            exit_reason = 'stop_loss'
        elif float(last['low']) <= pos['tp']:
            exit_price = pos['tp']
            exit_reason = 'take_profit'
        else:
            return state

    # 決済
    if pos['dir'] == 'long':
        pnl_pips = (exit_price - pos['entry']) / PIP
    else:
        pnl_pips = (pos['entry'] - exit_price) / PIP
    pnl = pnl_pips * pos['size']

    state['cash'] += pnl
    state['total_pnl'] += pnl
    state['total_trades'] += 1
    if pnl > 0:
        state['wins'] += 1
    else:
        state['losses'] += 1

    if state['cash'] > state['peak_cash']:
        state['peak_cash'] = state['cash']
    dd = (state['peak_cash'] - state['cash']) / state['peak_cash']
    if dd > state['max_dd']:
        state['max_dd'] = dd

    trade_log = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'type': 'exit',
        'direction': pos['dir'],
        'entry_price': pos['entry'],
        'exit_price': exit_price,
        'sl': pos['sl'],
        'tp': pos['tp'],
        'size': pos['size'],
        'pnl': round(pnl, 2),
        'exit_reason': exit_reason,
        'cash': round(state['cash'], 0),
    }
    log_trade(trade_log)

    print(f"  [EXIT] {pos['dir'].upper()} "
          f"entry={pos['entry']:.2f} exit={exit_price:.2f} "
          f"PnL={pnl:+.2f} ({exit_reason})")

    state['position'] = None
    return state


# ---- シグナル生成 ----

def generate_signal(bars: pd.DataFrame) -> str:
    """最新バーのシグナルを生成"""
    signals = yagami_signal(bars, freq=STRATEGY_FREQ, min_grade=MIN_GRADE)
    last_sig = signals.iloc[-1]
    return last_sig if isinstance(last_sig, str) else None


# ---- メイン処理 ----

def run_once(verbose: bool = True) -> dict:
    """1回のペーパートレードチェックを実行"""
    now = datetime.now(timezone.utc)
    if verbose:
        print(f"\n[ペーパートレード] {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    state = load_state()
    bars = get_latest_bars(freq=STRATEGY_FREQ, n_bars=200)

    if bars is None or len(bars) < 30:
        print("  データ不足 - スキップ")
        return state

    if verbose:
        last = bars.iloc[-1]
        print(f"  最新足: {bars.index[-1]} "
              f"O={last['open']:.2f} H={last['high']:.2f} "
              f"L={last['low']:.2f} C={last['close']:.2f}")

    # ポジション更新（SL/TP判定）
    state = update_position(state, bars)

    # 現在のポジション状況を表示
    if state['position']:
        pos = state['position']
        if verbose:
            print(f"  [保有中] {pos['dir'].upper()} "
                  f"entry={pos['entry']:.2f} "
                  f"SL={pos['sl']:.2f} TP={pos['tp']:.2f}")
    else:
        # 新規シグナル検出
        signal = generate_signal(bars)

        if signal in ('long', 'short'):
            atr = _calc_atr(bars)
            entry_price, sl_price, tp_price, sl_dist = _calc_sl_tp(bars, signal, atr)

            sl_pips = sl_dist / PIP
            risk_amount = state['cash'] * RISK_PCT
            size = max(0.01, round(risk_amount / sl_pips, 2)) if sl_pips > 0 else 0.01

            state['position'] = {
                'dir': signal,
                'entry': round(entry_price, 5),
                'sl': round(sl_price, 5),
                'tp': round(tp_price, 5),
                'size': size,
                'entry_time': now.isoformat(),
                'atr': round(atr, 4),
            }

            trade_log = {
                'timestamp': now.isoformat(),
                'type': 'entry',
                'direction': signal,
                'entry_price': round(entry_price, 5),
                'sl': round(sl_price, 5),
                'tp': round(tp_price, 5),
                'size': size,
                'pnl': None,
                'exit_reason': None,
                'cash': round(state['cash'], 0),
            }
            log_trade(trade_log)

            if verbose:
                print(f"  [ENTRY] {signal.upper()} "
                      f"price={entry_price:.2f} "
                      f"SL={sl_price:.2f} TP={tp_price:.2f} "
                      f"size={size:.2f} lots")
        else:
            if verbose:
                print("  シグナルなし")

    # 統計表示
    trades = state['total_trades']
    wr = state['wins'] / trades * 100 if trades > 0 else 0
    ret_pct = (state['cash'] - INIT_CASH) / INIT_CASH * 100

    if verbose:
        print(f"  [統計] 取引数={trades} 勝率={wr:.0f}% "
              f"総PnL={state['total_pnl']:+.0f} "
              f"リターン={ret_pct:+.2f}% "
              f"DD={state['max_dd']*100:.1f}%")

    state['last_run'] = now.isoformat()
    save_state(state)
    return state


def compare_with_backtest(strategy_name: str = 'YagamiB',
                          freq: str = '1h') -> None:
    """ペーパートレード結果とバックテスト結果を比較表示"""
    from lib.yagami import sig_yagami_B

    print(f"\n=== バックテスト vs ペーパートレード 比較 ===")

    # バックテスト実行
    bars = get_latest_bars(freq=freq, n_bars=500)
    engine = BacktestEngine(
        init_cash=INIT_CASH,
        risk_pct=RISK_PCT,
        use_dynamic_sl=True,
    )
    bt_result = engine.run(bars, sig_yagami_B(freq), freq=freq, name=strategy_name)

    # ペーパートレード状態
    state = load_state()
    pt_trades = state['total_trades']
    pt_wr = state['wins'] / pt_trades * 100 if pt_trades > 0 else 0
    pt_ret = (state['cash'] - INIT_CASH) / INIT_CASH * 100

    print(f"\n{'指標':<20} {'バックテスト':>14} {'ペーパートレード':>16}")
    print("-" * 52)

    if bt_result:
        print(f"{'取引数':<20} {bt_result['total_trades']:>14} {pt_trades:>16}")
        print(f"{'勝率':<20} {bt_result['win_rate_pct']:>13.1f}% {pt_wr:>15.1f}%")
        print(f"{'リターン':<20} {bt_result['total_return_pct']:>13.2f}% {pt_ret:>15.2f}%")
        print(f"{'最大DD':<20} {bt_result['max_drawdown_pct']:>13.1f}% {state['max_dd']*100:>15.1f}%")
        print(f"{'PF':<20} {bt_result['profit_factor']:>14.3f} {'N/A':>16}")
    else:
        print("  バックテスト結果なし")

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ペーパートレードランナー')
    parser.add_argument('--loop', action='store_true', help='ループ実行モード')
    parser.add_argument('--interval', type=int, default=60,
                        help='ループ間隔（秒）')
    parser.add_argument('--compare', action='store_true',
                        help='バックテストと比較')
    args = parser.parse_args()

    if args.compare:
        compare_with_backtest()
    elif args.loop:
        print(f"ペーパートレード ループ開始（{args.interval}秒ごと）")
        print("Ctrl+C で停止")
        while True:
            try:
                run_once()
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n停止しました")
                break
    else:
        run_once()
