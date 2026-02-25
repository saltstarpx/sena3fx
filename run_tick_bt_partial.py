"""
取得済みティックデータで本格バックテストを実行する。
2週分（約320万ティック）で精度検証。
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_DIR = os.path.join(BASE_DIR, 'data', 'tick')

# ティックバックテストエンジンをインポート
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
from tick_backtest import (
    TickBacktestEngine, BacktestConfig,
    signal_sma_crossover, signal_rsi_reversal, signal_bbands,
    signal_macd, signal_rsi_sma_combo
)

def load_tick_data():
    """取得済みのティックCSVを結合して読み込み"""
    csv_files = sorted([
        os.path.join(TICK_DIR, f) for f in os.listdir(TICK_DIR)
        if f.startswith('XAUUSD_tick_') and f.endswith('.csv')
    ])
    
    if not csv_files:
        print("ティックデータが見つかりません")
        return None
    
    print(f"ティックデータ読み込み: {len(csv_files)}ファイル")
    all_dfs = []
    for f in csv_files:
        print(f"  {os.path.basename(f)}", end='... ', flush=True)
        df = pd.read_csv(f)
        # timestamp列名の確認
        ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], format='ISO8601', utc=True)
        df = df.set_index(ts_col)
        df.index = df.index.tz_convert(None)
        
        # 列名の正規化
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if 'bid' in cl and 'price' in cl:
                col_map[c] = 'bidPrice'
            elif 'ask' in cl and 'price' in cl:
                col_map[c] = 'askPrice'
            elif 'bid' in cl and 'vol' in cl:
                col_map[c] = 'bidVolume'
            elif 'ask' in cl and 'vol' in cl:
                col_map[c] = 'askVolume'
        if col_map:
            df = df.rename(columns=col_map)
        
        print(f"{len(df):,} ticks")
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # bidPrice, askPrice列が必要
    if 'bidPrice' not in combined.columns:
        # 列名を確認
        print(f"  列: {list(combined.columns)}")
        # open/close等の場合
        if 'open' in combined.columns:
            combined['bidPrice'] = combined['close']
            combined['askPrice'] = combined['close'] + 0.5  # 推定スプレッド
    
    return combined


def main():
    print("=" * 75)
    print("ティックレベル本格バックテスト")
    print(f"実行時刻: {datetime.now()}")
    print("=" * 75)
    
    tick_data = load_tick_data()
    if tick_data is None or len(tick_data) == 0:
        print("データなし。終了。")
        return
    
    print(f"\n合計: {len(tick_data):,} ticks")
    print(f"期間: {tick_data.index[0]} ~ {tick_data.index[-1]}")
    
    if 'askPrice' in tick_data.columns and 'bidPrice' in tick_data.columns:
        spread = tick_data['askPrice'] - tick_data['bidPrice']
        print(f"スプレッド: 平均={spread.mean():.4f}, 中央={spread.median():.4f}, 最大={spread.max():.4f}")
    
    # バックテスト設定
    config = BacktestConfig(
        init_cash=5_000_000,
        risk_per_trade=0.02,
        default_sl_pips=20.0,
        default_tp_pips=40.0,
        use_trailing_stop=False,
        slippage_pips=0.3,
        pip_value=0.1,
    )
    
    engine = TickBacktestEngine(config)
    all_results = []
    
    # 戦略群
    strategies = [
        # SMAクロスオーバー
        ('SMA(10/50)', signal_sma_crossover(None, 10, 50)),
        ('SMA(20/100)', signal_sma_crossover(None, 20, 100)),
        ('SMA(5/20)', signal_sma_crossover(None, 5, 20)),
        ('SMA(30/200)', signal_sma_crossover(None, 30, 200)),
        # RSI
        ('RSI(14,30/70)', signal_rsi_reversal(None, 14, 30, 70)),
        ('RSI(21,25/75)', signal_rsi_reversal(None, 21, 25, 75)),
        ('RSI(7,20/80)', signal_rsi_reversal(None, 7, 20, 80)),
        # ボリンジャーバンド
        ('BB(20,2.0)', signal_bbands(None, 20, 2.0)),
        ('BB(20,2.5)', signal_bbands(None, 20, 2.5)),
        ('BB(30,2.0)', signal_bbands(None, 30, 2.0)),
        # MACD
        ('MACD(12/26/9)', signal_macd(None, 12, 26, 9)),
        ('MACD(8/21/5)', signal_macd(None, 8, 21, 5)),
        ('MACD(16/30/9)', signal_macd(None, 16, 30, 9)),
        # 複合
        ('RSI(14)+SMA(50)', signal_rsi_sma_combo(None, 14, 30, 70, 50)),
        ('RSI(14)+SMA(200)', signal_rsi_sma_combo(None, 14, 30, 70, 200)),
        ('RSI(21)+SMA(100)', signal_rsi_sma_combo(None, 21, 25, 75, 100)),
    ]
    
    # 各時間足でテスト
    for tf in ['1h', '4h']:
        print(f"\n{'='*60}")
        print(f"時間足: {tf}")
        print(f"{'='*60}")
        
        for name, signal_func in strategies:
            full_name = f"{name}_{tf}"
            try:
                result = engine.run(tick_data, signal_func, signal_timeframe=tf, name=full_name)
                if result and result.get('total_trades', 0) > 0:
                    result['timeframe'] = tf
                    all_results.append(result)
            except Exception as e:
                print(f"  [{full_name}] エラー: {e}")
    
    # 結果保存
    if all_results:
        import json
        results_dir = os.path.join(BASE_DIR, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_df = pd.DataFrame(all_results)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(results_dir, f'tick_bt_partial_{ts}.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n結果保存: {csv_path}")
        
        # 合格戦略
        passed = results_df[results_df['passed'] == True]
        if len(passed) > 0:
            print(f"\n{'='*60}")
            print(f"合格戦略: {len(passed)}件")
            print(f"{'='*60}")
            for _, row in passed.iterrows():
                print(f"  {row['strategy']}: PF={row['profit_factor']:.2f}, WR={row['win_rate_pct']:.1f}%, DD={row['max_drawdown_pct']:.1f}%, N={row['total_trades']}")
            
            json_path = os.path.join(results_dir, 'tick_approved_strategies.json')
            with open(json_path, 'w') as f:
                json.dump(passed.to_dict('records'), f, indent=2, default=str)
        else:
            print(f"\n合格戦略なし")
            # 上位3戦略を表示
            if len(results_df) > 0:
                results_df['score'] = results_df['profit_factor'] * results_df['win_rate_pct'] / 100
                top3 = results_df.nlargest(3, 'score')
                print("上位3戦略:")
                for _, row in top3.iterrows():
                    print(f"  {row['strategy']}: PF={row['profit_factor']:.2f}, WR={row['win_rate_pct']:.1f}%, DD={row['max_drawdown_pct']:.1f}%, N={row['total_trades']}")
    
    print(f"\n{'='*75}")
    print("ティックレベル本格バックテスト完了")
    print("=" * 75)


if __name__ == '__main__':
    main()
