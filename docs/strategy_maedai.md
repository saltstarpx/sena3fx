# Teammate B: Maedai — 高RRトレンドフォロー

## 概要

低勝率・高リスクリワードのトレンドフォロー戦略。
「背を近くして何度も挑戦し、大きな値動きを取る」思想。

**評価基準**: Sharpe Ratio > 1.5, Calmar Ratio > 3.0, 年間トレード数 > 50

## コア戦略: Donchian ブレイクアウト

- Donchian チャネル (N日間の高値/安値) のブレイクでエントリー
- EMA(200) フィルターでトレンド方向のみ許可
- ATR ベースのトレーリングストップで利益追従

## パラメータグリッド (Task 3 結果)

| DC期間 | EMA | Sharpe | Calmar | PF | DD |
|--------|-----|--------|--------|----|----|
| DC30 | 200 | **1.060** | **3.558** | 1.731 | 3.0% |
| DC30 | 100 | **1.060** | **3.558** | 1.731 | 3.0% |
| DC40 | 200 | 0.941 | 3.045 | 1.628 | 3.0% |
| DC20 | 200 | 0.734 | 0.961 | 1.432 | 8.8% |
| Union | - | **1.859** | **4.153** | 2.061 | 12.7% |

**最適パラメータ**: DC30 + EMA200 (Calmar 3.558, DD 3.0%)

## シグナル関数

| 関数 | 説明 | ファイル |
|------|------|----------|
| `sig_maedai_dc_ema_tf` | DC+EMA (推奨) | `lib/yagami.py` |
| `sig_maedai_yagami_union` | DC+やがみ統合 | `lib/yagami.py` |
| `sig_maedai_best` | 最適化済み | `lib/yagami.py` |
| `sig_maedai_dc_usd` | DC + USD強弱フィルター | `strategies/maedai_breakout.py` |

## エンジン設定 (Maedai専用)

```python
BacktestEngine(
    risk_pct=0.03,          # 控えめリスク (3%)
    default_sl_atr=1.5,     # タイトSL
    default_tp_atr=6.0,     # 大きなTP
    dynamic_rr=3.0,         # 高RR狙い
    trail_start_atr=3.0,    # 3ATR利益で追従開始
    trail_dist_atr=1.5,     # 追従距離
    target_min_wr=0.25,     # 低WR許容
    target_rr_threshold=3.0,
)
```
