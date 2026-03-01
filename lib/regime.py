"""
レジーム転換モデル — HMM (Hidden Markov Model) v1.0
====================================================
価格データから「トレンド」「レンジ」の2状態を検出する。

使い方:
    detector = HiddenMarkovRegimeDetector(n_states=2)
    detector.fit(daily_close)
    regimes = detector.predict(daily_close)
    # 0=レンジ, 1=トレンド

参考: hmmlearn GaussianHMM
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

    フィット: 日次リターン (log return) を観測値としてGaussianHMMを学習。
    ラベル:   ボラティリティが高い状態 → レジーム1 (トレンド)
              ボラティリティが低い状態 → レジーム0 (レンジ)

    Args:
        n_states   : 隠れ状態数 (デフォルト2)
        n_iter     : EM学習の反復回数 (デフォルト100)
        random_seed: 再現性用シード
    """

    def __init__(self, n_states: int = 2, n_iter: int = 100,
                 random_seed: int = 42):
        if not _HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn が必要です: pip install hmmlearn"
            )
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_seed = random_seed
        self._model = None
        self._trend_state: int = 1   # 高ボラ → トレンド
        self._range_state: int = 0   # 低ボラ → レンジ

    def fit(self, close: pd.Series) -> 'HiddenMarkovRegimeDetector':
        """
        日足終値系列でHMMを学習する。

        Args:
            close: 日次終値の pd.Series (index は datetime)
        Returns:
            self
        """
        log_ret = np.log(close / close.shift(1)).dropna().values.reshape(-1, 1)

        model = _hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.random_seed,
        )
        model.fit(log_ret)
        self._model = model

        # ボラティリティで状態ラベルを決定
        # covars_ shape: (n_states, 1, 1) for 'full'
        vols = np.sqrt(model.covars_[:, 0, 0])
        trend_idx = int(np.argmax(vols))   # 最もボラが高い = トレンド
        range_idx = int(np.argmin(vols))   # 最もボラが低い = レンジ

        self._trend_state = trend_idx
        self._range_state = range_idx

        # 状態統計をキャッシュ
        self._state_means = model.means_[:, 0]
        self._state_vols  = vols

        return self

    def predict(self, close: pd.Series) -> pd.Series:
        """
        終値系列のレジームを予測する。

        Returns:
            pd.Series[int]: 0=レンジ, 1=トレンド (index は close と同じ)
        """
        if self._model is None:
            raise RuntimeError("先に fit() を呼び出してください")

        log_ret = np.log(close / close.shift(1)).dropna()
        raw_states = self._model.predict(log_ret.values.reshape(-1, 1))

        # 0/1 の正規化ラベルに変換
        normalized = np.where(raw_states == self._trend_state, 1, 0)

        # 最初の1本 (NaN だった行) に前の値を前埋め
        result = pd.Series(0, index=close.index, dtype=int)
        result.iloc[1:] = normalized
        result.iloc[0] = normalized[0] if len(normalized) > 0 else 0

        return result

    def predict_current(self, close: pd.Series) -> int:
        """最新バーのレジームを返す (0=レンジ, 1=トレンド)。"""
        return int(self.predict(close).iloc[-1])

    def regime_stats(self) -> dict:
        """各レジームの平均リターン・ボラティリティを返す。"""
        if self._model is None:
            return {}
        return {
            f'state_{i}': {
                'label': 'trend' if i == self._trend_state else 'range',
                'mean_return': float(self._state_means[i]),
                'volatility': float(self._state_vols[i]),
            }
            for i in range(self.n_states)
        }

    def __repr__(self):
        trend_lbl = (
            f'trend_state={self._trend_state}, '
            f'vol={self._state_vols[self._trend_state]:.5f}'
        ) if self._model else 'not fitted'
        return f"HiddenMarkovRegimeDetector(n_states={self.n_states}, {trend_lbl})"
