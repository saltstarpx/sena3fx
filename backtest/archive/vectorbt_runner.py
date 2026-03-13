"""
vectorbt デュアルTFバックテストランナー
=========================================
4H × 15m デュアルタイムフレーム戦略をベクトル化バックテストで検証する。

実装ルール:
  - トレンド: 4H足 EMA21
  - エントリー: 15m足 DC ブレイクアウト (4H方向一致)
  - SL: 4H足スイング安値/高値
  - TP: SL距離 × rr_target
  - RRフィルター: RR < min_rr はスキップ
  - 時間フィルター: 年末年始/祝日/土曜日

必要パッケージ:
  pip install vectorbt

実行方法:
  python backtest/vectorbt_runner.py

  または Python から:
  from backtest.vectorbt_runner import run_dual_tf_backtest
  results = run_dual_tf_backtest('data/ohlc/XAUUSD_2025_4h.csv',
                                  'data/ohlc/XAUUSD_2025_15m.csv')
"""

import sys
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.swing   import build_swing_sl_series_long, build_swing_sl_series_short
from lib.dual_tf import align_4h_trend_to_15m, compute_15m_signals_vectorized
from live.time_filter import filter_signals_by_time

# vectorbt の import (なければ警告)
try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False


# ------------------------------------------------------------------ #
#  データ読み込み                                                     #
# ------------------------------------------------------------------ #

def load_ohlc(path: str) -> pd.DataFrame:
    """
    sena3fx 形式の OHLC CSV を読み込む。
    datetime列が TZ付き/なし どちらでも対応。
    """
    df = pd.read_csv(path)

    try:
        dt = pd.to_datetime(df['datetime'], utc=True)
        dt = dt.dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert(None)

    df['datetime'] = dt
    df = df.set_index('datetime').sort_index()
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


# ------------------------------------------------------------------ #
#  メインバックテスト関数                                             #
# ------------------------------------------------------------------ #

def run_dual_tf_backtest(
    bars_4h_path: str,
    bars_15m_path: str,
    instrument: str = 'XAU_USD',
    init_cash: float = 10_000_000.0,
    risk_pct: float = 0.02,
    min_rr: float = 2.0,
    rr_target: float = 2.5,
    ema_days: int = 21,
    dc_lookback: int = 20,
    swing_window: int = 3,
    swing_lookback: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    long_only: bool = True,
    fees: float = 0.0002,
) -> dict:
    """
    デュアルTFバックテストを実行する。

    Args:
        bars_4h_path:  4H足 OHLC CSV のパス
        bars_15m_path: 15m足 OHLC CSV のパス
        instrument:    通貨ペア名 (表示用)
        init_cash:     初期資金 (USD)
        risk_pct:      1トレードのリスク率 (例: 0.02 = 2%)
        min_rr:        最小RR比 (例: 2.0 → 2.0未満はスキップ)
        rr_target:     TP計算用倍率 (TP = entry + rr_target × SL距離)
        ema_days:      4H足トレンドEMA日数
        dc_lookback:   15m足DCブレイクアウト期間 (本数)
        swing_window:  スイング検出ウィンドウ (前後N本)
        swing_lookback: スイング探索期間 (最大遡り本数 @ 4H)
        start_date:    開始日 (例: '2025-04-01')
        end_date:      終了日 (例: '2026-02-28')
        long_only:     True = ロングのみ
        fees:          取引コスト率 (例: 0.0002 = 0.02%)

    Returns:
        dict: {
            'status': 'ok' | 'no_signals' | 'no_vectorbt',
            'n_long': int,
            'n_short': int,
            'pf_long':  vectorbt.Portfolio (or None),
            'pf_short': vectorbt.Portfolio (or None),
        }
    """
    if not HAS_VBT:
        print("[ERROR] vectorbt がインストールされていません。")
        print("  pip install vectorbt")
        return {'status': 'no_vectorbt'}

    _header(f"デュアルTFバックテスト: {instrument}")

    # ---- データ読み込み ---- #
    bars_4h  = load_ohlc(bars_4h_path)
    bars_15m = load_ohlc(bars_15m_path)

    # 期間フィルター
    if start_date:
        bars_4h  = bars_4h[bars_4h.index   >= start_date]
        bars_15m = bars_15m[bars_15m.index >= start_date]
    if end_date:
        bars_4h  = bars_4h[bars_4h.index   <= end_date]
        bars_15m = bars_15m[bars_15m.index <= end_date]

    print(f"4H:  {len(bars_4h):,} 本  [{bars_4h.index[0]}  →  {bars_4h.index[-1]}]")
    print(f"15m: {len(bars_15m):,} 本  [{bars_15m.index[0]}  →  {bars_15m.index[-1]}]")

    # ---- 4H トレンドを 15m にアライン ---- #
    print("\n[1/5] 4H トレンド計算 → 15m インデックスにアライン...")
    trend_15m = align_4h_trend_to_15m(bars_4h, bars_15m, ema_days)

    # ---- 15m シグナル計算 ---- #
    print("[2/5] 15m エントリーシグナル計算...")
    signals_15m = compute_15m_signals_vectorized(bars_15m, trend_15m, dc_lookback)
    print(f"      ロング候補: {(signals_15m == 'long').sum():,} 本")
    print(f"      ショート候補: {(signals_15m == 'short').sum():,} 本")

    # ---- 時間フィルター ---- #
    print("[3/5] 時間フィルター適用 (年末年始/祝日/土曜日)...")
    signals_15m = filter_signals_by_time(signals_15m)
    n_long_raw  = (signals_15m == 'long').sum()
    n_short_raw = (signals_15m == 'short').sum()
    print(f"      フィルター後 ロング: {n_long_raw:,} 本")
    print(f"      フィルター後 ショート: {n_short_raw:,} 本")

    if n_long_raw + n_short_raw == 0:
        print("[WARNING] シグナルが0件。パラメータを確認してください。")
        return {'status': 'no_signals'}

    # ---- SL/TP 計算 ---- #
    print("[4/5] スイングSL/TP計算...")
    sl_long_4h  = build_swing_sl_series_long( bars_4h, swing_window, swing_lookback)
    sl_short_4h = build_swing_sl_series_short(bars_4h, swing_window, swing_lookback)

    # 4H → 15m にフォワードフィル
    combined_idx = bars_4h.index.union(bars_15m.index)
    sl_long_15m  = sl_long_4h.reindex(combined_idx).ffill().reindex(bars_15m.index)
    sl_short_15m = sl_short_4h.reindex(combined_idx).ffill().reindex(bars_15m.index)

    close_15m = bars_15m['close']

    # SL% (対 close 比率) ← vectorbt の sl_stop に使用
    sl_pct_long  = ((close_15m - sl_long_15m)  / close_15m).clip(lower=0.0)
    sl_pct_short = ((sl_short_15m - close_15m) / close_15m).clip(lower=0.0)

    # TP% = rr_target × SL%
    tp_pct_long  = sl_pct_long  * rr_target
    tp_pct_short = sl_pct_short * rr_target

    # ---- RR フィルター適用 ---- #
    # TP = entry + rr_target × SL → actual RR = rr_target
    # min_rr チェック: rr_target >= min_rr かつ SL距離 > 0
    rr_ok_long  = (sl_pct_long  > 1e-4) & (rr_target >= min_rr)
    rr_ok_short = (sl_pct_short > 1e-4) & (rr_target >= min_rr)

    entries_long  = (signals_15m == 'long')  & rr_ok_long
    entries_short = (signals_15m == 'short') & rr_ok_short

    if long_only:
        entries_short = pd.Series(False, index=bars_15m.index)

    n_long  = entries_long.sum()
    n_short = entries_short.sum()
    print(f"      RRフィルター後 ロング: {n_long:,} 本")
    print(f"      RRフィルター後 ショート: {n_short:,} 本")

    # ---- vectorbt ポートフォリオ ---- #
    print("[5/5] vectorbt ポートフォリオ構築中...")

    # リスク額ベースのポジションサイズ (固定リスク方式)
    # size = risk_amount / (close × sl_pct)  → ユニット数
    risk_amount = init_cash * risk_pct

    pf_long  = None
    pf_short = None

    no_exits = pd.Series(False, index=bars_15m.index)

    # ---- ロングポートフォリオ ---- #
    if n_long > 0:
        # safe clip で 0除算を防ぐ
        sl_l = sl_pct_long.fillna(0.01).clip(lower=1e-4)
        tp_l = tp_pct_long.fillna(sl_l * rr_target).clip(lower=1e-4)

        pf_long = vbt.Portfolio.from_signals(
            close=close_15m,
            entries=entries_long,
            exits=no_exits,
            sl_stop=sl_l,
            tp_stop=tp_l,
            init_cash=init_cash,
            size=risk_amount,
            size_type='value',
            fees=fees,
            freq='15min',
        )

    # ---- ショートポートフォリオ ---- #
    if n_short > 0 and not long_only:
        sl_s = sl_pct_short.fillna(0.01).clip(lower=1e-4)
        tp_s = tp_pct_short.fillna(sl_s * rr_target).clip(lower=1e-4)

        pf_short = vbt.Portfolio.from_signals(
            close=close_15m,
            entries=no_exits,
            exits=no_exits,
            short_entries=entries_short,
            short_exits=no_exits,
            sl_stop=sl_s,
            tp_stop=tp_s,
            init_cash=init_cash,
            size=risk_amount,
            size_type='value',
            fees=fees,
            freq='15min',
        )

    # ---- 統計出力 ---- #
    print()
    _header("バックテスト結果")

    results = {
        'status':   'ok',
        'instrument': instrument,
        'n_long':   int(n_long),
        'n_short':  int(n_short),
        'pf_long':  pf_long,
        'pf_short': pf_short,
    }

    if pf_long is not None:
        print("\n[ロング戦略]")
        stats = _print_stats(pf_long)
        results['stats_long'] = stats

    if pf_short is not None:
        print("\n[ショート戦略]")
        stats = _print_stats(pf_short)
        results['stats_short'] = stats

    if pf_long is None and pf_short is None:
        print("  シグナルが0件のため統計なし。")

    return results


# ------------------------------------------------------------------ #
#  ユーティリティ                                                     #
# ------------------------------------------------------------------ #

def _header(title: str, width: int = 60):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _print_stats(pf) -> dict:
    """vectorbt ポートフォリオの統計を整形して出力する"""
    try:
        stats = pf.stats()
        rows = {
            '総トレード数':       stats.get('Total Trades',      'N/A'),
            '勝率':               f"{stats.get('Win Rate [%]', 0):.1f}%",
            '総リターン':         f"{stats.get('Total Return [%]', 0):.1f}%",
            '年率換算リターン':   f"{stats.get('Annualized Return [%]', 0):.1f}%",
            '最大ドローダウン':   f"{stats.get('Max Drawdown [%]', 0):.1f}%",
            'シャープレシオ':     f"{stats.get('Sharpe Ratio', 0):.3f}",
            'ソルティノレシオ':   f"{stats.get('Sortino Ratio', 0):.3f}",
            'カルマーレシオ':     f"{stats.get('Calmar Ratio', 0):.3f}",
            '期待値/トレード':    f"{stats.get('Expectancy', 0):.2f}",
            '最終残高':           f"${stats.get('End Value', 0):,.0f}",
        }
        for k, v in rows.items():
            print(f"  {k:<20}: {v}")
        return dict(stats)
    except Exception as e:
        print(f"  [統計エラー: {e}]")
        return {}


# ------------------------------------------------------------------ #
#  コマンドライン実行                                                 #
# ------------------------------------------------------------------ #

def main():
    """
    コマンドラインから実行:
      python backtest/vectorbt_runner.py
    """
    root = ROOT

    # --- Gold (XAU_USD) --- #
    run_dual_tf_backtest(
        bars_4h_path  = str(root / 'data' / 'ohlc' / 'XAUUSD_2025_4h.csv'),
        bars_15m_path = str(root / 'data' / 'ohlc' / 'XAUUSD_2025_15m.csv'),
        instrument    = 'XAU_USD',
        init_cash     = 10_000_000,
        risk_pct      = 0.02,
        min_rr        = 2.0,
        rr_target     = 2.5,
        ema_days      = 21,
        dc_lookback   = 20,
        swing_window  = 3,
        swing_lookback= 10,
        start_date    = '2025-04-01',
        end_date      = '2026-02-28',
        long_only     = True,
    )

    # --- Silver (XAG_USD) --- #
    run_dual_tf_backtest(
        bars_4h_path  = str(root / 'data' / 'ohlc' / 'XAGUSD_2025_4h.csv'),
        bars_15m_path = str(root / 'data' / 'ohlc' / 'XAGUSD_2025_15m.csv'),
        instrument    = 'XAG_USD',
        init_cash     = 10_000_000,
        risk_pct      = 0.02,
        min_rr        = 2.0,
        rr_target     = 2.5,
        ema_days      = 21,
        dc_lookback   = 20,
        swing_window  = 3,
        swing_lookback= 10,
        start_date    = '2025-04-01',
        end_date      = '2026-02-28',
        long_only     = True,
    )


if __name__ == '__main__':
    main()
