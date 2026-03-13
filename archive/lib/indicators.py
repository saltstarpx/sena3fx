"""
インジケーターベース戦略シグナル関数
====================================
SMA, RSI, BB, MACD等の従来型テクニカル指標戦略。
tick_backtest_fast.py (v2.0) から移植。
"""
import pandas as pd
import numpy as np


class Ind:
    """テクニカルインジケーター計算クラス"""
    @staticmethod
    def sma(s, p): return s.rolling(p).mean()
    @staticmethod
    def ema(s, p): return s.ewm(span=p, adjust=False).mean()
    @staticmethod
    def rsi(s, p=14):
        d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
        return 100 - 100/(1 + g.rolling(p).mean() / l.rolling(p).mean())
    @staticmethod
    def bbands(s, p=20, sd=2.0):
        m = s.rolling(p).mean(); st = s.rolling(p).std()
        return m, m+sd*st, m-sd*st
    @staticmethod
    def macd(s, f=12, sl=26, sg=9):
        ef = s.ewm(span=f, adjust=False).mean()
        es = s.ewm(span=sl, adjust=False).mean()
        ml = ef-es; sig = ml.ewm(span=sg, adjust=False).mean()
        return ml, sig, ml-sig
    @staticmethod
    def atr(h, l, c, p=14):
        tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
        return tr.rolling(p).mean()
    @staticmethod
    def adx(h, l, c, p=14):
        """
        Average Directional Index (ADX) — トレンド強度指標。
        0〜100: 25以上でトレンド相場、20以下でレンジ相場。
        """
        up   = h.diff()
        down = -l.diff()
        dm_plus  = np.where((up > down) & (up > 0), up,  0.0)
        dm_minus = np.where((down > up) & (down > 0), down, 0.0)

        tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
        atr_s = tr.ewm(alpha=1/p, min_periods=p, adjust=False).mean()

        di_plus  = 100 * pd.Series(dm_plus,  index=h.index).ewm(
            alpha=1/p, min_periods=p, adjust=False).mean() / atr_s
        di_minus = 100 * pd.Series(dm_minus, index=h.index).ewm(
            alpha=1/p, min_periods=p, adjust=False).mean() / atr_s

        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
        adx_s = dx.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
        return adx_s


def sig_sma(fast=20, slow=50):
    def _f(bars):
        c = bars['close']
        fm = Ind.sma(c, fast); sm = Ind.sma(c, slow)
        s = pd.Series(index=bars.index, dtype=object)
        pf = fm.shift(1); ps = sm.shift(1)
        s[(pf <= ps) & (fm > sm)] = 'long'
        s[(pf >= ps) & (fm < sm)] = 'close'
        return s
    return _f

def sig_rsi(period=14, os_lv=30, ob_lv=70):
    def _f(bars):
        rsi = Ind.rsi(bars['close'], period)
        s = pd.Series(index=bars.index, dtype=object)
        pr = rsi.shift(1)
        s[(pr <= os_lv) & (rsi > os_lv)] = 'long'
        s[(pr >= ob_lv) & (rsi < ob_lv)] = 'close'
        return s
    return _f

def sig_bb(period=20, sd=2.0):
    def _f(bars):
        c = bars['close']
        _, upper, lower = Ind.bbands(c, period, sd)
        s = pd.Series(index=bars.index, dtype=object)
        pc = c.shift(1)
        s[(pc <= lower.shift(1)) & (c > lower)] = 'long'
        s[(pc <= upper.shift(1)) & (c > upper)] = 'close'
        return s
    return _f

def sig_macd(fast=12, slow=26, signal=9):
    def _f(bars):
        ml, sl, _ = Ind.macd(bars['close'], fast, slow, signal)
        s = pd.Series(index=bars.index, dtype=object)
        pm = ml.shift(1); ps = sl.shift(1)
        s[(pm <= ps) & (ml > sl)] = 'long'
        s[(pm >= ps) & (ml < sl)] = 'close'
        return s
    return _f

def sig_rsi_sma(rsi_p=14, os_lv=30, ob_lv=70, sma_p=50):
    def _f(bars):
        c = bars['close']
        rsi = Ind.rsi(c, rsi_p)
        sma = Ind.sma(c, sma_p)
        s = pd.Series(index=bars.index, dtype=object)
        pr = rsi.shift(1)
        up = c > sma
        s[(pr <= os_lv) & (rsi > os_lv) & up] = 'long'
        s[(pr >= ob_lv) & (rsi < ob_lv)] = 'close'
        return s
    return _f

def sig_macd_rsi(macd_f=12, macd_s=26, macd_sig=9, rsi_p=14, rsi_thresh=50):
    """MACD + RSIフィルター（RSI>50のときのみロング）"""
    def _f(bars):
        c = bars['close']
        ml, sl, _ = Ind.macd(c, macd_f, macd_s, macd_sig)
        rsi = Ind.rsi(c, rsi_p)
        s = pd.Series(index=bars.index, dtype=object)
        pm = ml.shift(1); ps = sl.shift(1)
        s[(pm <= ps) & (ml > sl) & (rsi > rsi_thresh)] = 'long'
        s[(pm >= ps) & (ml < sl)] = 'close'
        return s
    return _f

def sig_bb_rsi(bb_p=20, bb_sd=2.0, rsi_p=14, rsi_os=30, rsi_ob=70):
    """ボリンジャーバンド + RSI複合"""
    def _f(bars):
        c = bars['close']
        _, upper, lower = Ind.bbands(c, bb_p, bb_sd)
        rsi = Ind.rsi(c, rsi_p)
        s = pd.Series(index=bars.index, dtype=object)
        s[(c < lower) & (rsi < rsi_os)] = 'long'
        s[(c > upper) & (rsi > rsi_ob)] = 'close'
        return s
    return _f
