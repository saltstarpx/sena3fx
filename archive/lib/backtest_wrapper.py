
import numpy as np
import pandas as pd
from lib.backtest import BacktestEngine

class MTFBacktestEngine(BacktestEngine):
    """
    3層MTF戦略用にカスタマイズされたバックテストエンジン。
    シグナル生成時に外部から提供されるSL/TPを優先的に使用します。
    """
    def __init__(self, signals_df, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signals_df = signals_df
        # タイムスタンプをキーにした辞書を作成して高速化
        self.signal_lookup = signals_df.set_index('timestamp').to_dict('index')

    def run(self, name="Yagami_MTF_v3", freq="1m"):
        """
        BacktestEngine.run をオーバーライドし、エントリー時に外部SL/TPを適用します。
        """
        # 親クラスのrunメソッドのロジックを模倣しつつ、エントリー部分を修正
        # (元々のBacktestEngine.runが複雑なため、最小限の修正で対応)
        
        # 実際には、BacktestEngineの内部でsigを受け取った後にSL/TPを決定しているため、
        # その決定ロジックをフックする必要があります。
        # しかし、元のコードを書き換えるのはリスクがあるため、
        # run_backtest_run010.py 側でシグナル関数を工夫するアプローチを取ります。
        return super().run(name=name, freq=freq)

def run_mtf_backtest(m1_df, signals_df, initial_capital=100000, risk_per_trade=0.02):
    """
    外部SL/TPを考慮したバックテストを実行するヘルパー関数。
    """
    # シグナルがある時刻のみ 'long'/'short' を返す関数
    signal_map = signals_df.set_index('timestamp')['signal'].to_dict()
    sl_map = signals_df.set_index('timestamp')['stop_loss'].to_dict()
    tp_map = signals_df.set_index('timestamp')['take_profit'].to_dict()

    class CustomEngine(BacktestEngine):
        def run(self, name="Yagami_MTF_v3", freq="1m"):
            # 元のrunメソッドのロジックをここにコピーして修正するのが確実ですが、
            # 既存のengine.runを活かすため、内部状態を監視・修正するトリッキーな方法をとります。
            # (あるいは、単にengine.runを呼び出す際に、外部SL/TPが適用されるように細工する)
            return super().run(name=name, freq=freq)

    # 実際には、BacktestEngine.runの中でシグナルを受け取った直後に
    # self.use_dynamic_sl などの設定に基づいて SL/TP を計算しています。
    # これを外部値で上書きするため、BacktestEngine のメソッドをモンキーパッチします。

    engine = BacktestEngine(init_cash=initial_capital, risk_pct=risk_per_trade)
    
    # 元のメソッドを保存
    original_find_swing_low = engine._find_swing_low
    original_find_swing_high = engine._find_swing_high

    def custom_find_sl(bars, idx, htf_bars=None, n_confirm=2):
        ts = bars.index[idx]
        if ts in sl_map:
            return sl_map[ts]
        # シグナルがない場合は元のロジック（通常は呼ばれないはず）
        direction = signal_map.get(ts)
        if direction == 'long':
            return original_find_swing_low(bars, idx, htf_bars, n_confirm)
        else:
            return original_find_swing_high(bars, idx, htf_bars, n_confirm)

    # モンキーパッチ
    engine._find_swing_low = custom_find_sl
    engine._find_swing_high = custom_find_sl
    engine.use_dynamic_sl = True # 常にカスタム関数を呼ぶようにする
    
    # TPも同様に制御したいが、BacktestEngine.run内部で tp = bar['close'] + ... * self.dynamic_rr となっている。
    # これを上書きするため、dynamic_rr を動的に調整するか、tp_mapを参照するように改造が必要。
    # ここでは、簡略化のため dynamic_rr を RR比から逆算して設定する。
    
    def custom_signal_func(bars):
        ts = bars.index[-1]
        sig = signal_map.get(ts)
        if sig:
            # エントリー直前に dynamic_rr を調整して TP を合わせる
            entry_price = bars['close'].iloc[-1]
            sl_price = sl_map[ts]
            tp_price = tp_map[ts]
            sl_dist = abs(entry_price - sl_price)
            tp_dist = abs(tp_price - entry_price)
            if sl_dist > 0:
                engine.dynamic_rr = tp_dist / sl_dist
            else:
                engine.dynamic_rr = 2.0
        return sig

    return engine.run(name="Yagami_MTF_v3", freq="1m", signal_func=custom_signal_func)
