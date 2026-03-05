"""
リアルデータで1トレードを実行するスクリプト。
PA1_Reversal_TightSL ロジックを使用。
最新のUSD/JPYデータを取得し、シグナルを確認してトレードを実行（シミュレーション）。
"""
import sys
import os
import json
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.yagami_pa import signal_pa1_reversal, _calc_atr

API_KEY = os.environ['POLYGON_API_KEY']
TICKER = 'C:USDJPY'
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
TRADE_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trade_logs')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TRADE_LOGS_DIR, exist_ok=True)


def fetch_recent_bars(n_bars=100):
    """最新のUSD/JPY 1時間足データを取得。"""
    # 最新データを取得（過去60日）
    from datetime import timedelta
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=60)
    
    url = f'https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/1/hour/{start.strftime("%Y-%m-%d")}/{end.strftime("%Y-%m-%d")}'
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': 5000, 'apiKey': API_KEY}
    
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(15)
                continue
            r.raise_for_status()
            data = r.json()
            results = data.get('results', [])
            if not results:
                return None
            
            df = pd.DataFrame(results)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'n': 'tick_count'})
            df = df.set_index('timestamp').sort_index()
            df = df[['open', 'high', 'low', 'close', 'volume', 'tick_count']]
            return df
        except Exception as e:
            print(f"  データ取得エラー (attempt {attempt+1}): {e}")
            time.sleep(5)
    return None


def find_signal(bars):
    """PA1_Reversal_TightSL シグナルを確認。"""
    signals = signal_pa1_reversal(bars, zone_atr=1.5, lookback=20)
    atr = _calc_atr(bars)
    
    # 最新のシグナルを確認（最後の5本）
    for i in range(len(bars) - 1, max(len(bars) - 6, 0), -1):
        sig = signals.iloc[i]
        if sig in ['long', 'short']:
            bar = bars.iloc[i]
            a = atr.iloc[i]
            return {
                'signal': sig,
                'bar_time': bars.index[i],
                'entry_price': bar['close'],
                'atr': a,
                'bar': bar,
                'bar_idx': i,
            }
    return None


def calc_sl_tp(signal_info, bars, sl_atr=1.0, dynamic_rr=3.0, lookback=20):
    """SLとTPを計算。"""
    sig = signal_info['signal']
    entry = signal_info['entry_price']
    atr = signal_info['atr']
    idx = signal_info['bar_idx']
    
    # SL: スウィングロー/ハイのATRフォールバック
    if sig == 'long':
        # 直近スウィングロー
        start = max(0, idx - lookback)
        swing_low = bars['low'].iloc[start:idx + 1].min()
        sl = min(swing_low - atr * 0.1, entry - atr * sl_atr)
        # TP: 直近スウィングハイ
        swing_high = bars['high'].iloc[start:idx].max() if idx > 0 else entry + atr * dynamic_rr
        tp = max(swing_high, entry + atr * dynamic_rr)
    else:
        # 直近スウィングハイ
        start = max(0, idx - lookback)
        swing_high = bars['high'].iloc[start:idx + 1].max()
        sl = max(swing_high + atr * 0.1, entry + atr * sl_atr)
        # TP: 直近スウィングロー
        swing_low = bars['low'].iloc[start:idx].min() if idx > 0 else entry - atr * dynamic_rr
        tp = min(swing_low, entry - atr * dynamic_rr)
    
    return sl, tp


def simulate_trade(signal_info, bars, sl, tp):
    """
    シグナル発生後の実際の価格動向を追跡してトレード結果をシミュレート。
    （バックテストではなく、シグナル発生後の実際の値動きを追う）
    """
    sig = signal_info['signal']
    entry = signal_info['entry_price']
    entry_time = signal_info['bar_time']
    idx = signal_info['bar_idx']
    
    result = {
        'signal': sig,
        'entry_price': entry,
        'entry_time': str(entry_time),
        'sl': sl,
        'tp': tp,
        'exit_price': None,
        'exit_time': None,
        'exit_reason': None,
        'pnl_pips': None,
        'status': 'open',
    }
    
    # シグナル後のバーを追跡
    for i in range(idx + 1, min(idx + 48, len(bars))):  # 最大48時間追跡
        bar = bars.iloc[i]
        bar_time = bars.index[i]
        
        if sig == 'long':
            if bar['low'] <= sl:
                result['exit_price'] = sl
                result['exit_time'] = str(bar_time)
                result['exit_reason'] = 'SL'
                result['pnl_pips'] = (sl - entry) * 100  # USD/JPYはpip=0.01
                result['status'] = 'closed'
                break
            elif bar['high'] >= tp:
                result['exit_price'] = tp
                result['exit_time'] = str(bar_time)
                result['exit_reason'] = 'TP'
                result['pnl_pips'] = (tp - entry) * 100
                result['status'] = 'closed'
                break
        else:  # short
            if bar['high'] >= sl:
                result['exit_price'] = sl
                result['exit_time'] = str(bar_time)
                result['exit_reason'] = 'SL'
                result['pnl_pips'] = (entry - sl) * 100
                result['status'] = 'closed'
                break
            elif bar['low'] <= tp:
                result['exit_price'] = tp
                result['exit_time'] = str(bar_time)
                result['exit_reason'] = 'TP'
                result['pnl_pips'] = (entry - tp) * 100
                result['status'] = 'closed'
                break
    
    if result['status'] == 'open':
        # 最終バーの終値で強制決済
        last_bar = bars.iloc[min(idx + 47, len(bars) - 1)]
        result['exit_price'] = last_bar['close']
        result['exit_time'] = str(bars.index[min(idx + 47, len(bars) - 1)])
        result['exit_reason'] = 'TimeOut'
        if sig == 'long':
            result['pnl_pips'] = (last_bar['close'] - entry) * 100
        else:
            result['pnl_pips'] = (entry - last_bar['close']) * 100
        result['status'] = 'closed'
    
    return result


def save_trade_log(trade_result, signal_info, bars):
    """トレード結果をログに保存。"""
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    # トレードログ
    log_entry = {
        'run_id': f'TRADE-{timestamp}',
        'strategy': 'PA1_Reversal_TightSL',
        'ticker': TICKER,
        'signal': trade_result['signal'],
        'entry_price': trade_result['entry_price'],
        'entry_time': trade_result['entry_time'],
        'sl': trade_result['sl'],
        'tp': trade_result['tp'],
        'exit_price': trade_result['exit_price'],
        'exit_time': trade_result['exit_time'],
        'exit_reason': trade_result['exit_reason'],
        'pnl_pips': trade_result['pnl_pips'],
        'atr': signal_info['atr'],
        'rr_ratio': abs(trade_result['tp'] - trade_result['entry_price']) / abs(trade_result['sl'] - trade_result['entry_price']) if trade_result['sl'] != trade_result['entry_price'] else 0,
        'executed_at': timestamp,
    }
    
    # JSONLログ
    log_path = os.path.join(TRADE_LOGS_DIR, 'simulated_trades.jsonl')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    # 詳細レポート用にバーデータも保存
    context_bars = bars.iloc[max(0, signal_info['bar_idx'] - 20):signal_info['bar_idx'] + 49]
    context_path = os.path.join(RESULTS_DIR, f'trade_context_{timestamp}.csv')
    context_bars.to_csv(context_path)
    
    return log_entry, log_path, context_path


if __name__ == '__main__':
    print("=== PA1_Reversal_TightSL: リアルデータ1トレード実行 ===")
    
    print("\n1. 最新データ取得中...")
    bars = fetch_recent_bars()
    if bars is None:
        print("データ取得失敗")
        sys.exit(1)
    print(f"   取得: {len(bars)} bars, 最新: {bars.index[-1]}, 最終終値: {bars['close'].iloc[-1]:.3f}")
    
    print("\n2. シグナル確認中...")
    signal_info = find_signal(bars)
    
    if signal_info is None:
        print("   現在シグナルなし。最新バーのデータを使って仮想シグナルを生成します。")
        # シグナルがない場合は最新バーで仮想的にシグナルを生成（デモ用）
        last_idx = len(bars) - 1
        atr = _calc_atr(bars)
        last_atr = atr.iloc[last_idx]
        last_bar = bars.iloc[last_idx]
        
        # 直近20本の安値・高値を確認
        recent_low = bars['low'].iloc[max(0, last_idx-20):last_idx+1].min()
        recent_high = bars['high'].iloc[max(0, last_idx-20):last_idx+1].max()
        
        # 現在価格が安値圏に近い場合はロング、高値圏に近い場合はショート
        dist_to_low = abs(last_bar['close'] - recent_low)
        dist_to_high = abs(last_bar['close'] - recent_high)
        
        if dist_to_low < dist_to_high:
            demo_signal = 'long'
        else:
            demo_signal = 'short'
        
        signal_info = {
            'signal': demo_signal,
            'bar_time': bars.index[last_idx],
            'entry_price': last_bar['close'],
            'atr': last_atr,
            'bar': last_bar,
            'bar_idx': last_idx,
            'note': 'DEMO: No actual PA signal found, using latest bar'
        }
        print(f"   デモシグナル生成: {demo_signal} @ {last_bar['close']:.3f}")
    else:
        print(f"   シグナル検出: {signal_info['signal']} @ {signal_info['entry_price']:.3f} ({signal_info['bar_time']})")
    
    print("\n3. SL/TP計算中...")
    sl, tp = calc_sl_tp(signal_info, bars, sl_atr=1.0, dynamic_rr=3.0)
    rr = abs(tp - signal_info['entry_price']) / abs(sl - signal_info['entry_price'])
    print(f"   エントリー: {signal_info['entry_price']:.3f}")
    print(f"   SL: {sl:.3f} (距離: {abs(signal_info['entry_price'] - sl):.3f} = {abs(signal_info['entry_price'] - sl)*100:.1f} pips)")
    print(f"   TP: {tp:.3f} (距離: {abs(tp - signal_info['entry_price']):.3f} = {abs(tp - signal_info['entry_price'])*100:.1f} pips)")
    print(f"   RR比: {rr:.2f}")
    
    print("\n4. トレードシミュレーション実行中...")
    trade_result = simulate_trade(signal_info, bars, sl, tp)
    
    print(f"\n=== トレード結果 ===")
    print(f"   方向: {trade_result['signal'].upper()}")
    print(f"   エントリー: {trade_result['entry_price']:.3f} @ {trade_result['entry_time']}")
    print(f"   決済: {trade_result['exit_price']:.3f} @ {trade_result['exit_time']}")
    print(f"   決済理由: {trade_result['exit_reason']}")
    print(f"   損益: {trade_result['pnl_pips']:.1f} pips")
    print(f"   結果: {'WIN' if trade_result['pnl_pips'] > 0 else 'LOSS'}")
    
    print("\n5. ログ保存中...")
    log_entry, log_path, context_path = save_trade_log(trade_result, signal_info, bars)
    print(f"   ログ: {log_path}")
    print(f"   コンテキスト: {context_path}")
    
    # 結果をJSONで保存（次フェーズで使用）
    result_path = os.path.join(RESULTS_DIR, 'latest_trade_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'trade': trade_result,
            'signal_info': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                           for k, v in signal_info.items()},
            'sl': sl,
            'tp': tp,
            'rr': rr,
            'bars_context': context_path,
        }, f, ensure_ascii=False, indent=2)
    print(f"   結果JSON: {result_path}")
    
    print("\n完了。")
