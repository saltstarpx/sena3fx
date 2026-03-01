# Teammate B: Maedai Strategy Documentation

## 概要

低勝率・高リスクリワード (RR) のトレンドフォロー戦略。
「背を近くして何度も挑戦し、大きな値動きを取る」思想を追求する。

**評価基準**: Sharpe Ratio > 1.5, Calmar Ratio > 3.0, 年間トレード数 > 50

## 「背を近く、大きく取る」思想

マエダイ式トレンドフォローの核心は以下の通り:

1. **タイトなストップ** (SL = ATR x 0.8) で損失を最小化
2. **大きなテイクプロフィット** (TP = ATR x 10+) で利益を最大化
3. **何度でも挑戦**: 損切りされても同方向のトレンドが続く限り再エントリー
4. **大きな時間軸のブレイクを捕捉**: Donchian Channel で明確なブレイクアウトを検知

勝率は低くなるが (30-40%)、1回の勝ちトレードで複数回の負けを補い、
RR比率 3:1 以上を目指す。

## Donchian ブレイクアウト戦略

### 基本ロジック

```
ロングシグナル: 終値 > 直近N本の高値 (Donchian上限)
ショートシグナル: 終値 < 直近N本の安値 (Donchian下限)
```

- Donchian Channel は `shift(1)` で現在バーを含まない
- アジア時間のブレイクアウトはデフォルトで除外 (`session_filter_on=True`)
- EMAフィルターでトレンド方向に順張りのみ許可

### パラメータグリッド

`strategies/maedai_breakout.py` の `DC_PARAM_GRID`:

| DC期間 (lookback_days) | EMA期間 (ema_days) | 説明 |
|------------------------|-------------------|------|
| 10 | 200 | 短期ブレイク + 長期トレンド |
| 15 | 200 | 中短期ブレイク + 長期トレンド |
| 20 | 200 | 標準ブレイク + 長期トレンド |
| 30 | 200 | 中長期ブレイク + 長期トレンド |
| 40 | 200 | 長期ブレイク + 長期トレンド |
| 15 | 100 | 中短期ブレイク + 中期トレンド |
| 20 | 100 | 標準ブレイク + 中期トレンド |
| 30 | 100 | 中長期ブレイク + 中期トレンド |

各パラメータセットに対してUSD強弱フィルター付きバリアントも自動生成される。

## シグナル関数一覧

コア実装は `lib/yagami.py`、戦略ハブは `strategies/maedai_breakout.py`。

### 基本シグナル

| 関数 | 説明 |
|------|------|
| `sig_maedai_breakout(freq, lookback, ...)` | 基本Donchianブレイクアウト |
| `sig_maedai_breakout_v2(freq, ...)` | 改良版 (ATR確認フィルター付き) |
| `sig_maedai_best(freq)` | 最良パラメータのプリセット |
| `sig_maedai_htf_breakout(lookback_htf, ...)` | MTFブレイクアウト (4H方向 + 1H精密エントリー) |
| `sig_maedai_htf_pullback(lookback_htf, ...)` | MTFプルバック (4Hトレンド + 1H押し目買い) |
| `sig_maedai_d1_dc30(lookback, ema_period)` | 日足DC30ブレイクアウト |
| `sig_maedai_d1_dc_multi(lookback, ...)` | 日足マルチパラメータ |

### 統合シグナル

| 関数 | 説明 |
|------|------|
| `sig_maedai_dc_ema_tf(freq, lookback_days, ema_days)` | DC + EMA + 時間足の統合シグナル |
| `sig_maedai_yagami_union(freq, lookback_days, ema_days)` | Maedai + Yagami のユニオン (どちらかで発火) |

### USD強弱フィルター付きバリアント (`strategies/maedai_breakout.py`)

| 変数名 | ベース |
|--------|--------|
| `sig_maedai_dc_usd` | `sig_maedai_dc_ema_tf` + USD threshold=75 |
| `sig_maedai_union_usd` | `sig_maedai_yagami_union` + USD threshold=75 |
| `sig_maedai_best_usd` | `sig_maedai_best` + USD threshold=75 |

## ATR トレーリングストップ

Maedai戦略のストップロスはATRベースで管理される:

- **エントリー時SL**: ATR x 0.8 (タイト)
- **テイクプロフィット**: ATR x 10.0 (大きく)
- **トレーリング**: `BacktestEngine` の `trailing_stop_atr` パラメータで制御

ATRトレーリングストップの動作:
1. エントリー後、価格が有利方向に動くとストップが追従
2. ストップ幅は ATR x トレーリング倍数で固定
3. 利益が伸びるほどストップも引き上がり、反転時に利益を確保

## バックテスト実行例

```python
from strategies.maedai_breakout import maedai_dc_variants, maedai_full_variants

# Donchianパラメータ探索
for name, sig_func in maedai_dc_variants(freq='4h'):
    result = engine.run(
        data=df,
        signal_func=sig_func,
        name=name,
        default_sl_atr=0.8,
        default_tp_atr=10.0,
    )

# 全バリアント実行
for name, sig_func in maedai_full_variants(freq='4h'):
    result = engine.run(data=df, signal_func=sig_func, name=name)
```

## 評価指標の計算

`lib/backtest.py` の `_report()` メソッドで以下が自動計算される:

- **Sharpe Ratio**: `mean(trade_returns) / std(trade_returns) * sqrt(trades_per_year)`
- **Calmar Ratio**: `annualized_return% / max_drawdown%`
- **Trades per Year**: `total_trades / years_active`
