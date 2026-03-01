"""
レジーム転換モデル — HMM (Hidden Markov Model) v2.0
====================================================
v1.0: 2状態 (レンジ/トレンド)
v2.0: 3状態 (レンジ/低ボラトレンド/高ボラトレンド) — v13新設

ラベリング規則:
  ボラティリティ: 低→中→高 の順で rank=0,1,2
  平均リターン: 正ならトレンド系、負/ゼロならレンジ寄り
  最終ラベル:
    rank=0 (最低ボラ)            → 0: range     (レンジ)
    rank=1 (中ボラ, mean_ret>0)  → 1: low_trend (低ボラトレンド)
    rank=1 (中ボラ, mean_ret<=0) → 0: range
    rank=2 (最高ボラ)            → 2: high_trend (高ボラトレンド)

使い方:
    detector = HiddenMarkovRegimeDetector(n_states=3)
    detector.fit(daily_close)
    regimes = detector.predict(daily_close)
    # 0=レンジ, 1=低ボラトレンド, 2=高ボラトレンド
"""

import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm as _hmm
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


class HiddenMarkovRegimeDetector:
    """
    日足終値からレジームを推定するHMMベースの検出器。

    n_states=2: 旧バージョン互換（range=0, trend=1）
    n_states=3: v2.0 (range=0, low_trend=1, high_trend=2)

    Args:
        n_states   : 隠れ状態数 (2 or 3)
        n_iter     : EM学習の反復回数
        random_seed: 再現性用シード
    """

    LABEL_RANGE      = 0
    LABEL_LOW_TREND  = 1
    LABEL_HIGH_TREND = 2

    def __init__(self, n_states: int = 3, n_iter: int = 200,
                 random_seed: int = 42):
        if not _HMM_AVAILABLE:
            raise ImportError("hmmlearn が必要です: pip install hmmlearn")
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_seed = random_seed
        self._model = None
        self._state_map: dict[int, int] = {}   # raw_state → label (0/1/2)
        self._state_means: np.ndarray | None = None
        self._state_vols:  np.ndarray | None = None

    # ──────────────────────────────────────────────────
    def fit(self, close: pd.Series) -> 'HiddenMarkovRegimeDetector':
        """
        日足終値系列でHMMを学習する。

        観測変数: [log_return, abs_log_return] の2次元
        (リターンの方向とボラを同時に学習させる)
        """
        log_ret = np.log(close / close.shift(1)).dropna().values
        obs = np.column_stack([log_ret, np.abs(log_ret)])  # (T, 2)

        model = _hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='diag',
            n_iter=self.n_iter,
            random_state=self.random_seed,
        )
        model.fit(obs)
        self._model = model

        # ボラティリティ = abs_return の平均値 (観測次元1)
        vols  = model.means_[:, 1]             # shape (n_states,)
        means = model.means_[:, 0]             # shape (n_states,): 平均リターン

        self._state_means = means
        self._state_vols  = vols

        # ラベル割当
        vol_rank = np.argsort(vols)  # [低ボラidx, ..., 高ボラidx]
        self._state_map = {}

        if self.n_states == 2:
            self._state_map[vol_rank[0]] = self.LABEL_RANGE
            self._state_map[vol_rank[1]] = self.LABEL_HIGH_TREND
        else:
            # 3状態
            # rank 0 (最低ボラ) → range
            self._state_map[vol_rank[0]] = self.LABEL_RANGE
            # rank 2 (最高ボラ) → high_trend
            self._state_map[vol_rank[-1]] = self.LABEL_HIGH_TREND
            # rank 1 (中ボラ): mean_return > 0 → low_trend, else → range
            mid_idx = vol_rank[1]
            self._state_map[mid_idx] = (
                self.LABEL_LOW_TREND if means[mid_idx] > 0 else self.LABEL_RANGE
            )

        return self

    # ──────────────────────────────────────────────────
    def predict(self, close: pd.Series) -> pd.Series:
        """
        終値系列のレジームを予測する。

        Returns:
            pd.Series[int]: 0=レンジ, 1=低ボラトレンド, 2=高ボラトレンド
        """
        if self._model is None:
            raise RuntimeError("先に fit() を呼び出してください")

        log_ret = np.log(close / close.shift(1)).dropna().values
        obs = np.column_stack([log_ret, np.abs(log_ret)])
        raw = self._model.predict(obs)

        mapped = np.array([self._state_map.get(int(s), 0) for s in raw])

        result = pd.Series(0, index=close.index, dtype=int)
        result.iloc[1:] = mapped
        result.iloc[0]  = int(mapped[0]) if len(mapped) > 0 else 0
        return result

    # ──────────────────────────────────────────────────
    def predict_current(self, close: pd.Series) -> int:
        """最新バーのレジームを返す。"""
        return int(self.predict(close).iloc[-1])

    def regime_stats(self) -> dict:
        """各状態の統計を返す。label別に集約。"""
        if self._model is None:
            return {}
        label_names = {0: 'range', 1: 'low_trend', 2: 'high_trend'}
        out = {}
        for raw_state, label in self._state_map.items():
            key = f'state_{raw_state}({label_names.get(label, label)})'
            out[key] = {
                'label': label_names.get(label, str(label)),
                'mean_return': float(self._state_means[raw_state]),
                'volatility':  float(self._state_vols[raw_state]),
            }
        return out

    def regime_distribution(self, close: pd.Series) -> dict:
        """各レジームの出現割合を返す (0〜1)。"""
        reg = self.predict(close)
        total = len(reg)
        labels = {0: 'range', 1: 'low_trend', 2: 'high_trend'}
        return {
            labels[k]: round(float((reg == k).sum() / total), 4)
            for k in sorted(labels.keys())
        }

    def __repr__(self):
        fitted = f'fitted, map={self._state_map}' if self._model else 'not fitted'
        return f"HiddenMarkovRegimeDetector(n_states={self.n_states}, {fitted})"
