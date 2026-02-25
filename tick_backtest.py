"""
==========================================================
ティックレベル バックテストエンジン v1.0
==========================================================
フォレックステスターの完全代替。

OHLCバックテストとの根本的な違い:
  1. ストップロス/テイクプロフィットがティック単位で正確に発動
  2. スプレッドが実際のBid/Ask差を使用（固定値ではない）
  3. バー内の価格推移を正確にシミュレーション
  4. スリッページの現実的なモデリング

対応戦略:
  - SMAクロスオーバー（トレンドフォロー）
  - RSI逆張り
  - ボリンジャーバンド逆張り
  - MACD
  - 複合戦略（RSI + SMAフィルター）
  - セッション別フィルター付き
  - カスタム戦略（関数で定義可能）
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# データ構造
# ============================================================
@dataclass
class Trade:
    """個別トレード記録"""
    entry_time: datetime
    exit_time: datetime
    direction: str       # 'long' or 'short'
    entry_price: float
    exit_price: float
    spread_at_entry: float
    spread_at_exit: float
    size: float          # ロットサイズ
    pnl: float
    pnl_pct: float
    exit_reason: str     # 'signal', 'stop_loss', 'take_profit', 'trailing_stop'
    duration_seconds: float
    max_favorable: float  # 最大含み益（pips）
    max_adverse: float    # 最大含み損（pips）


@dataclass
class Position:
    """現在のポジション"""
    direction: str
    entry_time: datetime
    entry_price: float
    spread_at_entry: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    max_favorable: float = 0.0
    max_adverse: float = 0.0


@dataclass
class BacktestConfig:
    """バックテスト設定"""
    init_cash: float = 5_000_000
    risk_per_trade: float = 0.02    # 1トレードあたりリスク（資金の2%）
    default_sl_pips: float = 20.0   # デフォルトSL（pips）
    default_tp_pips: float = 40.0   # デフォルトTP（pips）
    use_trailing_stop: bool = False
    trailing_stop_pips: float = 15.0
    slippage_pips: float = 0.5      # スリッページ
    pip_value: float = 0.1          # XAUUSD: 1pip = $0.1
    commission_per_lot: float = 0.0 # 手数料
    max_positions: int = 1          # 同時最大ポジション数


# ============================================================
# インジケーター計算（ティック→バーに変換して計算）
# ============================================================
class Indicators:
    """ティックデータからインジケーターを計算するユーティリティ"""
    
    @staticmethod
    def ticks_to_bars(tick_df, freq='1h'):
        """ティックデータをOHLCバーに変換"""
        price = tick_df['bidPrice']
        ohlc = price.resample(freq).agg(
            open='first', high='max', low='min', close='last'
        )
        if 'askPrice' in tick_df.columns:
            ohlc['spread'] = (tick_df['askPrice'] - tick_df['bidPrice']).resample(freq).mean()
        ohlc['tick_count'] = price.resample(freq).count()
        ohlc = ohlc.dropna(subset=['open'])
        return ohlc
    
    @staticmethod
    def sma(series, period):
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(series, period=20, std_dev=2.0):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return sma, upper, lower
    
    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


# ============================================================
# ティックレベルバックテストエンジン
# ============================================================
class TickBacktestEngine:
    """
    ティックレベルバックテストエンジン
    
    フォレックステスターと同等の精度でバックテストを実行する。
    各ティックごとにSL/TP/トレーリングストップを評価し、
    実際のBid/Askスプレッドを使用する。
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.position: Optional[Position] = None
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.cash = self.config.init_cash
        self.peak_equity = self.config.init_cash
        self.max_drawdown = 0.0
    
    def _calculate_lot_size(self, sl_distance: float) -> float:
        """リスクベースのロットサイズ計算"""
        risk_amount = self.cash * self.config.risk_per_trade
        if sl_distance <= 0:
            sl_distance = self.config.default_sl_pips * self.config.pip_value
        lot_size = risk_amount / (sl_distance / self.config.pip_value)
        return max(0.01, round(lot_size, 2))
    
    def _open_position(self, direction: str, bid: float, ask: float, 
                       timestamp: datetime, sl: float = None, tp: float = None):
        """ポジションをオープン"""
        if self.position is not None:
            return  # 既にポジションあり
        
        spread = ask - bid
        
        if direction == 'long':
            entry_price = ask + self.config.slippage_pips * self.config.pip_value  # Askで買う
            if sl is None:
                sl = entry_price - self.config.default_sl_pips * self.config.pip_value
            if tp is None:
                tp = entry_price + self.config.default_tp_pips * self.config.pip_value
        else:  # short
            entry_price = bid - self.config.slippage_pips * self.config.pip_value  # Bidで売る
            if sl is None:
                sl = entry_price + self.config.default_sl_pips * self.config.pip_value
            if tp is None:
                tp = entry_price - self.config.default_tp_pips * self.config.pip_value
        
        sl_distance = abs(entry_price - sl)
        size = self._calculate_lot_size(sl_distance)
        
        self.position = Position(
            direction=direction,
            entry_time=timestamp,
            entry_price=entry_price,
            spread_at_entry=spread,
            size=size,
            stop_loss=sl,
            take_profit=tp,
            trailing_stop_distance=self.config.trailing_stop_pips * self.config.pip_value if self.config.use_trailing_stop else None,
        )
    
    def _close_position(self, bid: float, ask: float, timestamp: datetime, reason: str):
        """ポジションをクローズ"""
        if self.position is None:
            return
        
        spread = ask - bid
        pos = self.position
        
        if pos.direction == 'long':
            exit_price = bid - self.config.slippage_pips * self.config.pip_value  # Bidで売る
            pnl_per_pip = pos.size
            pnl_pips = (exit_price - pos.entry_price) / self.config.pip_value
        else:
            exit_price = ask + self.config.slippage_pips * self.config.pip_value  # Askで買う
            pnl_per_pip = pos.size
            pnl_pips = (pos.entry_price - exit_price) / self.config.pip_value
        
        pnl = pnl_pips * pnl_per_pip
        pnl -= self.config.commission_per_lot * pos.size  # 手数料
        
        duration = (timestamp - pos.entry_time).total_seconds()
        
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=timestamp,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            spread_at_entry=pos.spread_at_entry,
            spread_at_exit=spread,
            size=pos.size,
            pnl=pnl,
            pnl_pct=pnl / self.cash * 100,
            exit_reason=reason,
            duration_seconds=duration,
            max_favorable=pos.max_favorable,
            max_adverse=pos.max_adverse,
        )
        
        self.trades.append(trade)
        self.cash += pnl
        
        # ドローダウン更新
        if self.cash > self.peak_equity:
            self.peak_equity = self.cash
        dd = (self.peak_equity - self.cash) / self.peak_equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        
        self.position = None
    
    def _check_sl_tp(self, bid: float, ask: float, timestamp: datetime):
        """ティックごとにSL/TP/トレーリングストップをチェック"""
        if self.position is None:
            return
        
        pos = self.position
        
        if pos.direction == 'long':
            current_pnl_pips = (bid - pos.entry_price) / self.config.pip_value
            
            # 最大含み益/損を更新
            if current_pnl_pips > pos.max_favorable:
                pos.max_favorable = current_pnl_pips
            if current_pnl_pips < -pos.max_adverse:
                pos.max_adverse = -current_pnl_pips
            
            # ストップロス
            if pos.stop_loss and bid <= pos.stop_loss:
                self._close_position(bid, ask, timestamp, 'stop_loss')
                return
            
            # テイクプロフィット
            if pos.take_profit and bid >= pos.take_profit:
                self._close_position(bid, ask, timestamp, 'take_profit')
                return
            
            # トレーリングストップ
            if pos.trailing_stop_distance:
                new_trail = bid - pos.trailing_stop_distance
                if pos.trailing_stop_price is None or new_trail > pos.trailing_stop_price:
                    pos.trailing_stop_price = new_trail
                if pos.trailing_stop_price and bid <= pos.trailing_stop_price:
                    self._close_position(bid, ask, timestamp, 'trailing_stop')
                    return
        
        else:  # short
            current_pnl_pips = (pos.entry_price - ask) / self.config.pip_value
            
            if current_pnl_pips > pos.max_favorable:
                pos.max_favorable = current_pnl_pips
            if current_pnl_pips < -pos.max_adverse:
                pos.max_adverse = -current_pnl_pips
            
            if pos.stop_loss and ask >= pos.stop_loss:
                self._close_position(bid, ask, timestamp, 'stop_loss')
                return
            
            if pos.take_profit and ask <= pos.take_profit:
                self._close_position(bid, ask, timestamp, 'take_profit')
                return
            
            if pos.trailing_stop_distance:
                new_trail = ask + pos.trailing_stop_distance
                if pos.trailing_stop_price is None or new_trail < pos.trailing_stop_price:
                    pos.trailing_stop_price = new_trail
                if pos.trailing_stop_price and ask >= pos.trailing_stop_price:
                    self._close_position(bid, ask, timestamp, 'trailing_stop')
                    return
    
    def run(self, tick_data: pd.DataFrame, signal_func: Callable, 
            signal_timeframe: str = '1h', name: str = 'Strategy') -> Dict:
        """
        ティックレベルバックテストを実行
        
        Args:
            tick_data: ティックデータ（bidPrice, askPrice列必須）
            signal_func: シグナル生成関数 (bars_df) -> pd.Series of {'long', 'short', 'close', None}
            signal_timeframe: シグナル計算用の時間足
            name: 戦略名
        
        Returns:
            バックテスト結果の辞書
        """
        # リセット
        self.trades = []
        self.position = None
        self.equity_curve = []
        self.cash = self.config.init_cash
        self.peak_equity = self.config.init_cash
        self.max_drawdown = 0.0
        
        print(f"\n  [{name}] ティックレベルバックテスト開始")
        print(f"    ティック数: {len(tick_data):,}")
        
        # バーデータに変換してシグナルを生成
        bars = Indicators.ticks_to_bars(tick_data, signal_timeframe)
        signals = signal_func(bars)
        
        print(f"    バー数: {len(bars):,} ({signal_timeframe})")
        
        # シグナルをティックのタイムスタンプにマッピング
        # 各バーの開始時刻にシグナルを配置
        signal_dict = {}
        for ts, sig in signals.items():
            if sig is not None and not pd.isna(sig):
                signal_dict[ts] = sig
        
        # ティックごとにループ
        current_bar_start = None
        current_signal = None
        tick_count = 0
        equity_sample_interval = max(1, len(tick_data) // 1000)  # エクイティカーブのサンプリング
        
        for timestamp, row in tick_data.iterrows():
            bid = row['bidPrice']
            ask = row['askPrice']
            tick_count += 1
            
            # SL/TP/トレーリングストップのチェック（毎ティック）
            self._check_sl_tp(bid, ask, timestamp)
            
            # バーの切り替わりでシグナルを確認
            bar_start = timestamp.floor(signal_timeframe)
            if bar_start != current_bar_start:
                current_bar_start = bar_start
                current_signal = signal_dict.get(bar_start, None)
                
                if current_signal is not None:
                    if current_signal == 'long' and self.position is None:
                        self._open_position('long', bid, ask, timestamp)
                    elif current_signal == 'short' and self.position is None:
                        self._open_position('short', bid, ask, timestamp)
                    elif current_signal == 'close' and self.position is not None:
                        self._close_position(bid, ask, timestamp, 'signal')
            
            # エクイティカーブのサンプリング
            if tick_count % equity_sample_interval == 0:
                equity = self.cash
                if self.position:
                    if self.position.direction == 'long':
                        unrealized = (bid - self.position.entry_price) / self.config.pip_value * self.position.size
                    else:
                        unrealized = (self.position.entry_price - ask) / self.config.pip_value * self.position.size
                    equity += unrealized
                self.equity_curve.append((timestamp, equity))
        
        # 残ポジションをクローズ
        if self.position is not None:
            last_row = tick_data.iloc[-1]
            self._close_position(last_row['bidPrice'], last_row['askPrice'], 
                               tick_data.index[-1], 'end_of_data')
        
        return self._generate_report(name)
    
    def _generate_report(self, name: str) -> Dict:
        """バックテスト結果レポートを生成"""
        n_trades = len(self.trades)
        
        if n_trades == 0:
            return {
                'strategy': name,
                'total_trades': 0,
                'passed': False,
                'note': 'トレードなし'
            }
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_win = sum(t.pnl for t in wins) if wins else 0
        total_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        
        win_rate = len(wins) / n_trades
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        # 平均保有時間
        avg_duration = np.mean([t.duration_seconds for t in self.trades])
        avg_duration_hours = avg_duration / 3600
        
        # 決済理由の内訳
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        
        # 平均スプレッド
        avg_spread_entry = np.mean([t.spread_at_entry for t in self.trades])
        avg_spread_exit = np.mean([t.spread_at_exit for t in self.trades])
        
        # MFE/MAE分析
        avg_mfe = np.mean([t.max_favorable for t in self.trades])
        avg_mae = np.mean([t.max_adverse for t in self.trades])
        
        total_return_pct = (self.cash - self.config.init_cash) / self.config.init_cash * 100
        
        # 合格判定
        passed = (
            profit_factor >= 1.5 and
            self.max_drawdown <= 0.10 and
            win_rate >= 0.50 and
            n_trades >= 30
        )
        
        result = {
            'strategy': name,
            'engine': 'tick_level',
            'total_return_pct': round(total_return_pct, 2),
            'total_pnl': round(total_pnl, 2),
            'end_value': round(self.cash, 0),
            'max_drawdown_pct': round(self.max_drawdown * 100, 2),
            'win_rate_pct': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 4),
            'total_trades': n_trades,
            'wins': len(wins),
            'losses': len(losses),
            'avg_win': round(np.mean([t.pnl for t in wins]), 2) if wins else 0,
            'avg_loss': round(np.mean([t.pnl for t in losses]), 2) if losses else 0,
            'avg_duration_hours': round(avg_duration_hours, 1),
            'avg_spread_entry': round(avg_spread_entry, 4),
            'avg_spread_exit': round(avg_spread_exit, 4),
            'avg_mfe_pips': round(avg_mfe, 1),
            'avg_mae_pips': round(avg_mae, 1),
            'exit_reasons': exit_reasons,
            'passed': passed,
        }
        
        status = "PASS" if passed else "fail"
        print(f"    結果: Ret={total_return_pct:+.1f}%  WR={win_rate*100:.1f}%  PF={profit_factor:.2f}  DD={self.max_drawdown*100:.1f}%  N={n_trades}  [{status}]")
        print(f"    決済理由: {exit_reasons}")
        print(f"    平均保有: {avg_duration_hours:.1f}h  平均スプレッド: {avg_spread_entry:.3f}")
        print(f"    MFE/MAE: {avg_mfe:.1f}/{avg_mae:.1f} pips")
        
        return result


# ============================================================
# シグナル生成関数（戦略定義）
# ============================================================
def signal_sma_crossover(bars, fast=20, slow=50):
    """SMAクロスオーバーシグナル"""
    def _gen(bars_df):
        close = bars_df['close']
        fast_ma = Indicators.sma(close, fast)
        slow_ma = Indicators.sma(close, slow)
        
        signals = pd.Series(index=bars_df.index, dtype=object)
        
        prev_fast = fast_ma.shift(1)
        prev_slow = slow_ma.shift(1)
        
        # ゴールデンクロス → long
        signals[(prev_fast <= prev_slow) & (fast_ma > slow_ma)] = 'long'
        # デッドクロス → close
        signals[(prev_fast >= prev_slow) & (fast_ma < slow_ma)] = 'close'
        
        return signals
    return _gen


def signal_rsi_reversal(bars, period=14, oversold=30, overbought=70):
    """RSI逆張りシグナル"""
    def _gen(bars_df):
        close = bars_df['close']
        rsi = Indicators.rsi(close, period)
        
        signals = pd.Series(index=bars_df.index, dtype=object)
        
        prev_rsi = rsi.shift(1)
        
        # RSIが売られすぎから上昇 → long
        signals[(prev_rsi <= oversold) & (rsi > oversold)] = 'long'
        # RSIが買われすぎから下降 → close
        signals[(prev_rsi >= overbought) & (rsi < overbought)] = 'close'
        
        return signals
    return _gen


def signal_bbands(bars, period=20, std_dev=2.0):
    """ボリンジャーバンドシグナル"""
    def _gen(bars_df):
        close = bars_df['close']
        sma, upper, lower = Indicators.bollinger_bands(close, period, std_dev)
        
        signals = pd.Series(index=bars_df.index, dtype=object)
        
        prev_close = close.shift(1)
        
        # 下バンド割れから回復 → long
        signals[(prev_close <= lower.shift(1)) & (close > lower)] = 'long'
        # 上バンド到達 → close
        signals[(prev_close <= upper.shift(1)) & (close > upper)] = 'close'
        
        return signals
    return _gen


def signal_macd(bars, fast=12, slow=26, signal=9):
    """MACDシグナル"""
    def _gen(bars_df):
        close = bars_df['close']
        macd_line, signal_line, _ = Indicators.macd(close, fast, slow, signal)
        
        signals = pd.Series(index=bars_df.index, dtype=object)
        
        prev_macd = macd_line.shift(1)
        prev_signal = signal_line.shift(1)
        
        # MACDがシグナルを上抜け → long
        signals[(prev_macd <= prev_signal) & (macd_line > signal_line)] = 'long'
        # MACDがシグナルを下抜け → close
        signals[(prev_macd >= prev_signal) & (macd_line < signal_line)] = 'close'
        
        return signals
    return _gen


def signal_rsi_sma_combo(bars, rsi_period=14, oversold=30, overbought=70, sma_period=50):
    """RSI + SMAトレンドフィルター複合シグナル"""
    def _gen(bars_df):
        close = bars_df['close']
        rsi = Indicators.rsi(close, rsi_period)
        sma = Indicators.sma(close, sma_period)
        
        signals = pd.Series(index=bars_df.index, dtype=object)
        
        prev_rsi = rsi.shift(1)
        trend_up = close > sma
        
        # RSI売られすぎ + 上昇トレンド → long
        signals[(prev_rsi <= oversold) & (rsi > oversold) & trend_up] = 'long'
        # RSI買われすぎ → close
        signals[(prev_rsi >= overbought) & (rsi < overbought)] = 'close'
        
        return signals
    return _gen


# ============================================================
# メイン実行
# ============================================================
def run_tick_backtest(tick_data_path=None, symbol='XAUUSD'):
    """ティックレベルバックテストを実行"""
    print("=" * 75)
    print("ティックレベル バックテストエンジン v1.0")
    print(f"実行時刻: {datetime.now()}")
    print("=" * 75)
    
    # ティックデータ読み込み
    if tick_data_path is None:
        tick_data_path = os.path.join(DATA_DIR, f'{symbol}_tick_sample.csv')
    
    if not os.path.exists(tick_data_path):
        print(f"エラー: ティックデータが見つかりません: {tick_data_path}")
        return None
    
    print(f"\nデータ読み込み: {tick_data_path}")
    tick_data = pd.read_csv(tick_data_path)
    tick_data['timestamp'] = pd.to_datetime(tick_data['timestamp'], format='ISO8601', utc=True)
    tick_data = tick_data.set_index('timestamp')
    # タイムゾーンを除去してresample互換にする
    tick_data.index = tick_data.index.tz_localize(None) if tick_data.index.tz is None else tick_data.index.tz_convert(None)
    print(f"  ティック数: {len(tick_data):,}")
    print(f"  期間: {tick_data.index[0]} ~ {tick_data.index[-1]}")
    print(f"  平均スプレッド: {(tick_data['askPrice'] - tick_data['bidPrice']).mean():.4f}")
    
    # バックテスト設定
    config = BacktestConfig(
        init_cash=5_000_000,
        risk_per_trade=0.02,
        default_sl_pips=20.0,
        default_tp_pips=40.0,
        slippage_pips=0.3,
        pip_value=0.1,  # XAUUSD
    )
    
    engine = TickBacktestEngine(config)
    all_results = []
    
    # 戦略群を実行
    strategies = [
        ('SMA(10/50)_tick', signal_sma_crossover(None, 10, 50)),
        ('SMA(20/100)_tick', signal_sma_crossover(None, 20, 100)),
        ('RSI(14,30/70)_tick', signal_rsi_reversal(None, 14, 30, 70)),
        ('RSI(21,25/75)_tick', signal_rsi_reversal(None, 21, 25, 75)),
        ('BB(20,2.0)_tick', signal_bbands(None, 20, 2.0)),
        ('BB(20,2.5)_tick', signal_bbands(None, 20, 2.5)),
        ('MACD(12/26/9)_tick', signal_macd(None, 12, 26, 9)),
        ('MACD(16/30/9)_tick', signal_macd(None, 16, 30, 9)),
        ('RSI(14)+SMA(50)_tick', signal_rsi_sma_combo(None, 14, 30, 70, 50)),
        ('RSI(14)+SMA(200)_tick', signal_rsi_sma_combo(None, 14, 30, 70, 200)),
    ]
    
    for name, signal_func in strategies:
        try:
            result = engine.run(tick_data, signal_func, signal_timeframe='1h', name=name)
            if result and result.get('total_trades', 0) > 0:
                all_results.append(result)
        except Exception as e:
            print(f"  [{name}] エラー: {e}")
    
    # 結果をCSVに保存
    if all_results:
        results_df = pd.DataFrame(all_results)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(RESULTS_DIR, f'tick_backtest_results_{timestamp_str}.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n結果保存: {csv_path}")
        
        # 合格戦略
        passed = results_df[results_df['passed'] == True]
        if len(passed) > 0:
            print(f"\n合格戦略: {len(passed)}件")
            for _, row in passed.iterrows():
                print(f"  {row['strategy']}: PF={row['profit_factor']:.2f}, WR={row['win_rate_pct']:.1f}%, DD={row['max_drawdown_pct']:.1f}%")
            
            # 合格戦略をJSON保存
            approved = passed.to_dict('records')
            json_path = os.path.join(RESULTS_DIR, 'tick_approved_strategies.json')
            with open(json_path, 'w') as f:
                json.dump(approved, f, indent=2, default=str)
            print(f"  → {json_path}")
        else:
            print("\n合格戦略なし（基準: PF>=1.5, DD<=10%, WR>=50%, N>=30）")
            print("  ※ サンプルデータ（1日分）ではトレード数が不足する可能性があります")
            print("  ※ より長期のティックデータで再テストが必要です")
    
    print(f"\n{'=' * 75}")
    print("ティックレベルバックテスト完了")
    print("=" * 75)
    
    return all_results


if __name__ == '__main__':
    run_tick_backtest()
