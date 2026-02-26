"""
ウォークフォワードテストエンジン
=================================
バックテストの過学習を検出し、戦略の汎化性能を検証する。

方式: ローリングウィンドウ
  - in_sample_ratio (例: 0.7) の期間で戦略を評価
  - out_of_sample_ratio (例: 0.3) の期間で汎化性能を測定
  - ウィンドウをスライドさせながら繰り返す

合格基準:
  - OOS/IS のPF比率 >= 0.7 (過学習が少ない)
  - OOS勝率 >= 40%
  - OOS期間に渡って一貫した正のリターン
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
import os
import json

from .backtest import BacktestEngine


class WalkForwardEngine:
    """
    ウォークフォワードテストエンジン。

    使い方:
        engine = WalkForwardEngine()
        result = engine.run(bars, signal_func, name='MyStrategy')
        print(result['passed'], result['oos_pf'])
    """

    def __init__(self,
                 in_sample_bars: int = 500,
                 out_of_sample_bars: int = 150,
                 step_bars: int = 100,
                 init_cash: float = 5_000_000,
                 risk_pct: float = 0.02):
        """
        Args:
            in_sample_bars: 訓練期間バー数
            out_of_sample_bars: 検証期間バー数
            step_bars: ウィンドウのスライド幅
            init_cash: 初期資金
            risk_pct: 1トレードリスク比率
        """
        self.in_sample_bars = in_sample_bars
        self.out_of_sample_bars = out_of_sample_bars
        self.step_bars = step_bars
        self.backtest_engine = BacktestEngine(
            init_cash=init_cash,
            risk_pct=risk_pct,
            use_dynamic_sl=True,
        )

    def run(self, bars: pd.DataFrame, signal_func: Callable,
            freq: str = '1h', name: str = 'Strategy') -> Dict:
        """
        ウォークフォワードテストを実行。

        Args:
            bars: OHLCバーデータ（十分な期間が必要）
            signal_func: bars -> pd.Series of 'long'/'short'/None
            freq: 時間足
            name: 戦略名

        Returns:
            Dict: 集計結果と各期間の詳細
        """
        total_bars = len(bars)
        window = self.in_sample_bars + self.out_of_sample_bars

        if total_bars < window:
            return {
                'strategy': name,
                'error': f'データ不足: {total_bars} < {window}',
                'passed': False,
                'windows': [],
            }

        windows = []
        start = 0

        while start + window <= total_bars:
            is_bars = bars.iloc[start: start + self.in_sample_bars]
            oos_bars = bars.iloc[start + self.in_sample_bars: start + window]

            # In-sample バックテスト
            is_result = self.backtest_engine.run(
                is_bars, signal_func, freq=freq,
                name=f'{name}_IS_{start}',
            )

            # Out-of-sample バックテスト
            oos_result = self.backtest_engine.run(
                oos_bars, signal_func, freq=freq,
                name=f'{name}_OOS_{start}',
            )

            if is_result and oos_result:
                is_start = bars.index[start]
                is_end = bars.index[start + self.in_sample_bars - 1]
                oos_end = bars.index[start + window - 1]

                windows.append({
                    'window_start': str(is_start),
                    'is_end': str(is_end),
                    'oos_end': str(oos_end),
                    'is_pf': is_result.get('profit_factor', 0),
                    'oos_pf': oos_result.get('profit_factor', 0),
                    'is_wr': is_result.get('win_rate_pct', 0),
                    'oos_wr': oos_result.get('win_rate_pct', 0),
                    'is_ret': is_result.get('total_return_pct', 0),
                    'oos_ret': oos_result.get('total_return_pct', 0),
                    'is_trades': is_result.get('total_trades', 0),
                    'oos_trades': oos_result.get('total_trades', 0),
                    'is_dd': is_result.get('max_drawdown_pct', 0),
                    'oos_dd': oos_result.get('max_drawdown_pct', 0),
                })

            start += self.step_bars

        return self._aggregate(name, windows, freq)

    def _aggregate(self, name: str, windows: List[Dict], freq: str) -> Dict:
        """ウィンドウ結果を集計して総合判定"""
        if not windows:
            return {
                'strategy': name,
                'error': 'テスト結果なし（トレード数不足の可能性）',
                'passed': False,
                'windows': [],
            }

        oos_pfs = [w['oos_pf'] for w in windows if w['oos_trades'] > 0]
        oos_wrs = [w['oos_wr'] for w in windows if w['oos_trades'] > 0]
        oos_rets = [w['oos_ret'] for w in windows]
        is_pfs = [w['is_pf'] for w in windows if w['is_trades'] > 0]

        avg_oos_pf = np.mean(oos_pfs) if oos_pfs else 0
        avg_oos_wr = np.mean(oos_wrs) if oos_wrs else 0
        avg_oos_ret = np.mean(oos_rets)
        avg_is_pf = np.mean(is_pfs) if is_pfs else 0

        # IS/OOS PF比率（1.0に近いほど過学習が少ない）
        pf_ratio = avg_oos_pf / avg_is_pf if avg_is_pf > 0 else 0

        # OOS期間でプラスリターンのウィンドウ割合
        positive_oos = sum(1 for r in oos_rets if r > 0)
        consistency = positive_oos / len(oos_rets) if oos_rets else 0

        # 合格判定
        # - OOS PF >= 1.2 (IS >= 1.5 より緩め)
        # - IS/OOS PF比率 >= 0.6 (過学習フィルター)
        # - OOS勝率 >= 40%
        # - プラスOOS期間 >= 60%
        passed = (
            avg_oos_pf >= 1.2
            and pf_ratio >= 0.6
            and avg_oos_wr >= 40.0
            and consistency >= 0.6
        )

        return {
            'strategy': name,
            'timeframe': freq,
            'engine': 'walk_forward_v1',
            'n_windows': len(windows),
            'avg_is_pf': round(avg_is_pf, 4),
            'avg_oos_pf': round(avg_oos_pf, 4),
            'pf_ratio': round(pf_ratio, 4),
            'avg_oos_wr': round(avg_oos_wr, 2),
            'avg_oos_ret': round(avg_oos_ret, 2),
            'oos_consistency': round(consistency * 100, 1),
            'passed': passed,
            'windows': windows,
        }


def run_walk_forward(bars: pd.DataFrame,
                     strategies: List[Tuple[str, Callable]],
                     freq: str = '1h',
                     in_sample_bars: int = 500,
                     out_of_sample_bars: int = 150,
                     step_bars: int = 100) -> pd.DataFrame:
    """
    複数戦略のウォークフォワードテストを一括実行。

    Returns:
        pd.DataFrame: 各戦略の集計結果
    """
    engine = WalkForwardEngine(
        in_sample_bars=in_sample_bars,
        out_of_sample_bars=out_of_sample_bars,
        step_bars=step_bars,
    )

    results = []
    for name, sfunc in strategies:
        print(f"  WF: {name} ...", end='', flush=True)
        r = engine.run(bars, sfunc, freq=freq, name=name)
        results.append({
            'strategy': r['strategy'],
            'n_windows': r.get('n_windows', 0),
            'avg_is_pf': r.get('avg_is_pf', 0),
            'avg_oos_pf': r.get('avg_oos_pf', 0),
            'pf_ratio': r.get('pf_ratio', 0),
            'avg_oos_wr': r.get('avg_oos_wr', 0),
            'avg_oos_ret': r.get('avg_oos_ret', 0),
            'oos_consistency': r.get('oos_consistency', 0),
            'passed': r.get('passed', False),
            'error': r.get('error', ''),
        })
        status = "PASS" if r.get('passed') else "fail"
        print(f" OOS_PF={r.get('avg_oos_pf', 0):.2f} "
              f"ratio={r.get('pf_ratio', 0):.2f} "
              f"consist={r.get('oos_consistency', 0):.0f}% [{status}]")

    return pd.DataFrame(results)
