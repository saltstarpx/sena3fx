"""
レジーム転換モデル — HMM (Hidden Markov Model) v3.0
====================================================
v1.0: 2状態 (レンジ/トレンド), 1D [log_return]
v2.0: 3状態 (レンジ/低ボラトレンド/高ボラトレンド), 2D [log_return, abs_return]
v3.0: 3状態, 5D [log_return, abs_return, ATR(14), ADX(14), RSI(14)] — v14新設

ラベリング規則:
  ボラティリティ: 低→中→高 の順で rank=0,1,2
  平均リターン: 正ならトレンド系、負/ゼロならレンジ寄り
  最終ラベル:
    rank=0 (最低ボラ)            → 0: range     (レンジ)
    rank=1 (中ボラ, mean_ret>0)  → 1: low_trend (低ボラトレンド)
    rank=1 (中ボラ, mean_ret<=0) → 0: range
    rank=2 (最高ボラ)            → 2: high_trend (高ボラトレンド)

使い方 (v3.0):
    detector = HiddenMarkovRegimeDetector(n_states=3)
    detector.fit(daily_close, ohlc_df=df_1d)   # 5D特徴量
    regimes = detector.predict(daily_close, ohlc_df=df_1d)
    # 0=レンジ, 1=低ボラトレンド, 2=高ボラトレンド

使い方 (v2.0互換):
    detector.fit(daily_close)                  # 2D特徴量（後方互換）
"""

import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm as _hmm
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


def _compute_atr(h, l, c, p=14):
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()


def _compute_adx(h, l, c, p=14):
    up = h.diff()
    down = -l.diff()
    dm_plus  = np.where((up > down) & (up > 0), up, 0.0)
    dm_minus = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    di_plus  = 100 * pd.Series(dm_plus,  index=h.index).ewm(
        alpha=1 / p, min_periods=p, adjust=False).mean() / atr_s
    di_minus = 100 * pd.Series(dm_minus, index=h.index).ewm(
        alpha=1 / p, min_periods=p, adjust=False).mean() / atr_s
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()


def _compute_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    return 100 - 100 / (1 + g.rolling(p).mean() / l.rolling(p).mean())


def _zscore(s: pd.Series) -> pd.Series:
    """欠損除外 z スコア正規化。"""
    m = s.mean()
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return s - m
    return (s - m) / sd


class HiddenMarkovRegimeDetector:
    """
    日足終値 (+ オプションでOHLC) からレジームを推定するHMMベースの検出器。

    n_states=2: 旧バージョン互換（range=0, trend=1）
    n_states=3: 3状態 (range=0, low_trend=1, high_trend=2)

    特徴量モード:
      - ohlc_df が None の場合 : 2D [log_return, abs_return] (v2.0互換)
      - ohlc_df が指定の場合   : 5D [log_return, abs_return, ATR, ADX, RSI] (v3.0)

    Args:
        n_states   : 隠れ状態数 (2 or 3)
        n_iter     : EM学習の反復回数
        random_seed: 再現性用シード
    """

    LABEL_RANGE      = 0
    LABEL_LOW_TREND  = 1
    LABEL_HIGH_TREND = 2

    def __init__(self, n_states: int = 3, n_iter: int = 300,
                 random_seed: int = 42):
        if not _HMM_AVAILABLE:
            raise ImportError("hmmlearn が必要です: pip install hmmlearn")
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_seed = random_seed
        self._model = None
        self._state_map: dict = {}
        self._state_means: np.ndarray | None = None
        self._state_vols:  np.ndarray | None = None
        self._use_5d: bool = False

    # ──────────────────────────────────────────────────
    def _build_features_2d(self, close: pd.Series) -> tuple:
        """2D特徴量 [log_return, abs_return] を構築して (obs, valid_idx) を返す。"""
        log_ret = np.log(close / close.shift(1))
        valid = log_ret.notna()
        lr = log_ret[valid].values
        obs = np.column_stack([lr, np.abs(lr)])
        return obs, close.index[valid]

    def _build_features_5d(self, close: pd.Series,
                           ohlc_df: pd.DataFrame) -> tuple:
        """
        5D特徴量 [log_return, abs_return, ATR_z, ADX_z, RSI_z] を構築。

        ohlc_df は close と同一インデックスの日足 OHLC DataFrame。
        各特徴量は z スコア正規化。欠損行は除外。
        """
        h, l, c = ohlc_df['high'], ohlc_df['low'], ohlc_df['close']

        log_ret = np.log(close / close.shift(1))
        atr_raw = _compute_atr(h, l, c, p=14)
        adx_raw = _compute_adx(h, l, c, p=14)
        rsi_raw = _compute_rsi(c, p=14)

        df = pd.DataFrame({
            'log_ret': log_ret,
            'abs_ret': log_ret.abs(),
            'atr':     atr_raw,
            'adx':     adx_raw,
            'rsi':     rsi_raw,
        }, index=close.index)
        df = df.dropna()

        # z スコア正規化 (ATR, ADX, RSI のスケールを統一)
        df['atr'] = _zscore(df['atr'])
        df['adx'] = _zscore(df['adx'])
        df['rsi'] = _zscore(df['rsi'])

        obs = df[['log_ret', 'abs_ret', 'atr', 'adx', 'rsi']].values
        return obs, df.index

    # ──────────────────────────────────────────────────
    def fit(self, close: pd.Series,
            ohlc_df: pd.DataFrame | None = None) -> 'HiddenMarkovRegimeDetector':
        """
        日足終値系列でHMMを学習する。

        Args:
            close  : 日足終値
            ohlc_df: 日足OHLC DataFrame (h/l/c/o 列を含む)。
                     指定時は5D特徴量、Noneなら2D特徴量 (v2.0互換)。
        """
        self._use_5d = ohlc_df is not None

        if self._use_5d:
            obs, _ = self._build_features_5d(close, ohlc_df)
            n_features = 5
        else:
            obs, _ = self._build_features_2d(close)
            n_features = 2

        model = _hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='diag',
            n_iter=self.n_iter,
            random_state=self.random_seed,
        )
        model.fit(obs)
        self._model = model

        # ボラティリティ = abs_return (観測次元1) の平均値で状態をランク付け
        vols  = model.means_[:, 1]   # abs_return 次元
        means = model.means_[:, 0]   # log_return 次元

        self._state_means = means
        self._state_vols  = vols

        # ラベル割当
        vol_rank = np.argsort(vols)
        self._state_map = {}

        if self.n_states == 2:
            self._state_map[vol_rank[0]] = self.LABEL_RANGE
            self._state_map[vol_rank[1]] = self.LABEL_HIGH_TREND
        else:
            self._state_map[vol_rank[0]] = self.LABEL_RANGE
            self._state_map[vol_rank[-1]] = self.LABEL_HIGH_TREND
            mid_idx = vol_rank[1]
            self._state_map[mid_idx] = (
                self.LABEL_LOW_TREND if means[mid_idx] > 0 else self.LABEL_RANGE
            )

        return self

    # ──────────────────────────────────────────────────
    def predict(self, close: pd.Series,
                ohlc_df: pd.DataFrame | None = None) -> pd.Series:
        """
        終値系列のレジームを予測する。

        fit() 時に ohlc_df を渡した場合は predict() にも同じ ohlc_df を渡すこと。

        Returns:
            pd.Series[int]: 0=レンジ, 1=低ボラトレンド, 2=高ボラトレンド
        """
        if self._model is None:
            raise RuntimeError("先に fit() を呼び出してください")

        use_5d = ohlc_df is not None or self._use_5d

        if use_5d and ohlc_df is not None:
            obs, valid_idx = self._build_features_5d(close, ohlc_df)
        else:
            obs, valid_idx = self._build_features_2d(close)

        raw = self._model.predict(obs)
        mapped = np.array([self._state_map.get(int(s), 0) for s in raw])

        result = pd.Series(0, index=close.index, dtype=int)
        result.loc[valid_idx] = mapped

        # 先頭の欠損（NaN区間）を最初の有効値で埋める
        first_valid_pos = result.index.get_loc(valid_idx[0]) if len(valid_idx) > 0 else 0
        if first_valid_pos > 0:
            result.iloc[:first_valid_pos] = result.iloc[first_valid_pos]

        return result

    # ──────────────────────────────────────────────────
    def predict_current(self, close: pd.Series,
                        ohlc_df: pd.DataFrame | None = None) -> int:
        """最新バーのレジームを返す。"""
        return int(self.predict(close, ohlc_df=ohlc_df).iloc[-1])

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

    def regime_distribution(self, close: pd.Series,
                            ohlc_df: pd.DataFrame | None = None) -> dict:
        """各レジームの出現割合を返す (0〜1)。"""
        reg = self.predict(close, ohlc_df=ohlc_df)
        total = len(reg)
        labels = {0: 'range', 1: 'low_trend', 2: 'high_trend'}
        return {
            labels[k]: round(float((reg == k).sum() / total), 4)
            for k in sorted(labels.keys())
        }

    def feature_stats_by_regime(self, close: pd.Series,
                                ohlc_df: pd.DataFrame) -> dict:
        """
        各レジームの特徴量平均値を返す (ダッシュボード表示用)。

        Returns:
            dict: {regime_name: {feature: mean_value}}
        """
        if self._model is None or ohlc_df is None:
            return {}

        h, l, c = ohlc_df['high'], ohlc_df['low'], ohlc_df['close']
        log_ret = np.log(close / close.shift(1))
        atr_raw = _compute_atr(h, l, c, p=14)
        adx_raw = _compute_adx(h, l, c, p=14)
        rsi_raw = _compute_rsi(c, p=14)

        df = pd.DataFrame({
            'log_ret': log_ret,
            'atr':     atr_raw,
            'adx':     adx_raw,
            'rsi':     rsi_raw,
        }, index=close.index).dropna()

        reg = self.predict(close, ohlc_df=ohlc_df).reindex(df.index, fill_value=0)
        df['regime'] = reg

        label_names = {0: 'range', 1: 'low_trend', 2: 'high_trend'}
        out = {}
        for lbl_id, lbl_name in label_names.items():
            subset = df[df['regime'] == lbl_id]
            if len(subset) == 0:
                out[lbl_name] = {'count': 0}
                continue
            out[lbl_name] = {
                'count':    len(subset),
                'avg_atr':  round(float(subset['atr'].mean()), 4),
                'avg_adx':  round(float(subset['adx'].mean()), 2),
                'avg_rsi':  round(float(subset['rsi'].mean()), 2),
                'mean_ret': round(float(subset['log_ret'].mean()) * 100, 4),
            }
        return out

    def __repr__(self):
        mode = '5D' if self._use_5d else '2D'
        fitted = f'fitted({mode}), map={self._state_map}' if self._model else 'not fitted'
        return f"HiddenMarkovRegimeDetector(n_states={self.n_states}, {fitted})"
