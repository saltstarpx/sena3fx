"""
pytest 共通フィクスチャ — 合成OHLCVデータファクトリ。

# テスト用合成データ — 実市場データではありません
全てのテストデータはここで生成される合成データを使用する。
実際の市場データは一切使用しない（データ捏造防止ポリシー準拠）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================
# テスト用合成データファクトリ
# # テスト用合成データ — 実市場データではありません
# ============================================================

def make_synthetic_ohlcv(
    n: int = 200,
    base_price: float = 2000.0,
    volatility: float = 10.0,
    trend: float = 0.0,
    seed: int = 42,
    freq: str = "4h",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """
    テスト用の合成OHLCVデータを生成する。

    # テスト用合成データ — 実市場データではありません

    Args:
        n:           バー数
        base_price:  初期価格
        volatility:  1バーあたりの価格変動幅 (USD)
        trend:       1バーあたりのトレンド (正=上昇, 負=下降)
        seed:        乱数シード（再現性のため）
        freq:        時間足 ('4h', '1h', '15min', '1D')
        start:       開始日時

    Returns:
        pd.DataFrame: columns=[open, high, low, close, volume], index=datetime
    """
    # テスト用合成データ — 実市場データではありません
    rng = np.random.default_rng(seed)

    # 価格系列生成
    returns = rng.normal(trend, volatility, n)
    closes = np.cumsum(returns) + base_price
    closes = np.maximum(closes, base_price * 0.1)  # 負値防止

    # OHLC 生成
    intra_bar_range = rng.uniform(volatility * 0.5, volatility * 1.5, n)
    highs = closes + intra_bar_range * rng.uniform(0.3, 0.7, n)
    lows  = closes - intra_bar_range * rng.uniform(0.3, 0.7, n)
    opens = np.roll(closes, 1)
    opens[0] = base_price

    volumes = rng.integers(100, 10000, n).astype(float)

    # 日時インデックス生成 (pandas 2.x lowercase aliases)
    freq_map = {"4h": "4h", "1h": "1h", "15min": "15min", "1D": "1D"}
    pd_freq = freq_map.get(freq, "4h")
    index = pd.date_range(start=start, periods=n, freq=pd_freq)

    df = pd.DataFrame(
        {
            "open":   opens,
            "high":   highs,
            "low":    lows,
            "close":  closes,
            "volume": volumes,
        },
        index=index,
    )
    df.index.name = "datetime"

    # OHLC 整合性保証
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"]  = df[["open", "high", "low", "close"]].min(axis=1)

    return df


def make_trending_ohlcv(
    n: int = 200,
    base_price: float = 2000.0,
    direction: str = "up",
    strength: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    トレンドのある合成OHLCVデータを生成。

    # テスト用合成データ — 実市場データではありません
    """
    trend = strength if direction == "up" else -strength
    return make_synthetic_ohlcv(
        n=n,
        base_price=base_price,
        volatility=strength * 0.5,
        trend=trend,
        seed=seed,
    )


def make_flat_ohlcv(
    n: int = 200,
    base_price: float = 2000.0,
    noise: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    レンジ相場（フラット）の合成OHLCVデータを生成。

    # テスト用合成データ — 実市場データではありません
    """
    return make_synthetic_ohlcv(
        n=n,
        base_price=base_price,
        volatility=noise,
        trend=0.0,
        seed=seed,
    )


# ============================================================
# pytest フィクスチャ
# ============================================================

@pytest.fixture
def synthetic_ohlcv():
    """標準的な合成OHLCVデータ (200バー, 4H)。
    # テスト用合成データ — 実市場データではありません
    """
    return make_synthetic_ohlcv(n=200, seed=42)


@pytest.fixture
def short_ohlcv():
    """短い合成OHLCVデータ (30バー) — データ不足テスト用。
    # テスト用合成データ — 実市場データではありません
    """
    return make_synthetic_ohlcv(n=30, seed=42)


@pytest.fixture
def trending_up_ohlcv():
    """上昇トレンドの合成OHLCVデータ。
    # テスト用合成データ — 実市場データではありません
    """
    return make_trending_ohlcv(n=200, direction="up", seed=42)


@pytest.fixture
def trending_down_ohlcv():
    """下降トレンドの合成OHLCVデータ。
    # テスト用合成データ — 実市場データではありません
    """
    return make_trending_ohlcv(n=200, direction="down", seed=42)


@pytest.fixture
def flat_ohlcv():
    """レンジ相場（フラット）の合成OHLCVデータ。
    # テスト用合成データ — 実市場データではありません
    """
    return make_flat_ohlcv(n=200, seed=42)


@pytest.fixture
def sentinel_price():
    """センチネル価格 — ライブバー（iloc[-1]）への注入用。"""
    return 999_999.99


@pytest.fixture
def entry_config():
    """entry_engine テスト用設定。"""
    return {
        "conditions": {
            "c1": {
                "atr_multiplier": 1.5,
                "min_touch_count": 2,
                "level_lookback": 100,
            },
            "c2": {"min_strength": 0.3},
            "c3": {"require_confirmed_pattern": True},
            "c4": {"pivot_window": 5, "min_bars": 25},
            "c5": {"htf_alignment_required": False, "asia_breakout_penalty": True},
        },
        "scorer": {
            "entry_threshold": 3,
            "strong_signal_threshold": 4,
        },
        "scanner": {
            "warmup_bars": 60,
            "sl_atr_mult": 2.0,
            "tp_atr_mult": 4.0,
            "signal_only": False,  # テストでは全バー出力
        },
    }
