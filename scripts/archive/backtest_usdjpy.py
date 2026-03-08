"""
USDJPY バックテスト: DCブレイク + EMA200 + ADX(>25) + Kelly(f=0.25)
XAUUSDの最終戦略と同じロジックをUSDJPYに適用して性能を比較する。
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ===== パラメータ =====
INITIAL_CAPITAL = 5_000_000   # 初期資金 500万円
KELLY_F = 0.25                 # Kelly分数
DC_WINDOW = 20                 # ドンチャンチャネル期間
EMA_PERIOD = 200               # EMA期間
ADX_PERIOD = 14                # ADX期間
ADX_THRESHOLD = 25             # ADXフィルター閾値
ATR_PERIOD = 14                # ATR期間
ATR_SL_MULT = 2.0              # ストップロス: ATR × 2.0
ATR_TP_MULT = 4.0              # テイクプロフィット: ATR × 4.0
SPREAD_PIPS = 0.03             # スプレッド（円）
LOT_SIZE = 100_000             # 1ロット = 10万通貨
MIN_LOT = 1_000                # 最小発注単位

# 季節フィルター（7月・9月は取引しない）
EXCLUDED_MONTHS = [7, 9]


def compute_atr(df, period=14):
    """ATRを計算する"""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_adx(df, period=14):
    """ADXを計算する"""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def run_backtest(df):
    """バックテストを実行してトレードリストを返す"""
    # 指標計算
    df = df.copy()
    df['ema200'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['dc_high'] = df['high'].shift(1).rolling(DC_WINDOW).max()
    df['dc_low'] = df['low'].shift(1).rolling(DC_WINDOW).min()
    df['atr'] = compute_atr(df, ATR_PERIOD)
    df['adx'] = compute_adx(df, ADX_PERIOD)

    capital = INITIAL_CAPITAL
    trades = []
    position = None  # {'side': 'long'/'short', 'entry': float, 'sl': float, 'tp': float, 'units': int, 'entry_time': datetime}
    equity_curve = [capital]

    for i in range(EMA_PERIOD + 1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # 季節フィルター
        if row.name.month in EXCLUDED_MONTHS:
            equity_curve.append(capital)
            continue

        # ポジション保有中の場合：SL/TP判定
        if position is not None:
            if position['side'] == 'long':
                # ロング: SL（安値がSL以下）またはTP（高値がTP以上）
                if row['low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry'] - SPREAD_PIPS) * position['units']
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': row.name,
                        'side': 'long',
                        'entry': position['entry'],
                        'exit': position['sl'],
                        'pnl': pnl,
                        'result': 'loss'
                    })
                    position = None
                elif row['high'] >= position['tp']:
                    pnl = (position['tp'] - position['entry'] - SPREAD_PIPS) * position['units']
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': row.name,
                        'side': 'long',
                        'entry': position['entry'],
                        'exit': position['tp'],
                        'pnl': pnl,
                        'result': 'win'
                    })
                    position = None
            else:  # short
                if row['high'] >= position['sl']:
                    pnl = (position['entry'] - position['sl'] - SPREAD_PIPS) * position['units']
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': row.name,
                        'side': 'short',
                        'entry': position['entry'],
                        'exit': position['sl'],
                        'pnl': pnl,
                        'result': 'loss'
                    })
                    position = None
                elif row['low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp'] - SPREAD_PIPS) * position['units']
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': row.name,
                        'side': 'short',
                        'entry': position['entry'],
                        'exit': position['tp'],
                        'pnl': pnl,
                        'result': 'win'
                    })
                    position = None

        # ポジションなしの場合：エントリー判定
        if position is None:
            atr = row['atr']
            adx = row['adx']
            ema = row['ema200']

            if pd.isna(atr) or pd.isna(adx) or pd.isna(ema) or pd.isna(row['dc_high']) or pd.isna(row['dc_low']):
                equity_curve.append(capital)
                continue

            # ADXフィルター
            if adx < ADX_THRESHOLD:
                equity_curve.append(capital)
                continue

            # Kelly基準でロットサイズを決定
            # リスク金額 = 資金 × Kelly_f
            risk_amount = capital * KELLY_F
            sl_distance = atr * ATR_SL_MULT  # 円単位

            # USDJPYの場合: 1通貨あたりのPnL = sl_distance / current_price (USD換算)
            # 日本円口座の場合: 1通貨あたりのPnL = sl_distance 円
            # risk_amount(円) = sl_distance(円) × units → units = risk_amount / sl_distance
            # ただしKellyが過大にならないよう資金の2%を最大リスクとしてキャップ
            max_risk = capital * 0.02  # 最大2%リスク
            risk_amount = min(capital * KELLY_F * 0.02, max_risk)  # Kelly × 2% cap
            units = int((risk_amount / sl_distance) / MIN_LOT) * MIN_LOT
            units = max(units, MIN_LOT)

            # ロングエントリー: 前足高値ブレイク + EMA200上
            if row['close'] > prev_row['dc_high'] and row['close'] > ema:
                entry = row['close'] + SPREAD_PIPS
                sl = entry - sl_distance
                tp = entry + atr * ATR_TP_MULT
                position = {
                    'side': 'long',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'units': units,
                    'entry_time': row.name
                }

            # ショートエントリー: 前足安値ブレイク + EMA200下
            elif row['close'] < prev_row['dc_low'] and row['close'] < ema:
                entry = row['close'] - SPREAD_PIPS
                sl = entry + sl_distance
                tp = entry - atr * ATR_TP_MULT
                position = {
                    'side': 'short',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'units': units,
                    'entry_time': row.name
                }

        equity_curve.append(capital)

    return trades, equity_curve


def compute_metrics(trades, equity_curve, initial_capital):
    """パフォーマンス指標を計算する"""
    if not trades:
        return {}

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    wins = (trades_df['result'] == 'win').sum()
    win_rate = wins / total_trades * 100

    final_capital = equity_curve[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100

    # 日次リターンに変換（4時間足 → 1日6本）
    equity_series = pd.Series(equity_curve)
    daily_equity = equity_series.groupby(equity_series.index // 6).last()
    daily_returns = daily_equity.pct_change().dropna()

    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # 最大ドローダウン
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_dd = drawdown.min()

    # Calmar
    annual_return = total_return / (len(equity_curve) / 6 / 252)
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    avg_win = trades_df[trades_df['result'] == 'win']['pnl'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['result'] == 'loss']['pnl'].mean() if (total_trades - wins) > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        'total_trades': total_trades,
        'win_rate': round(win_rate, 1),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'max_dd_pct': round(abs(max_dd), 1),
        'total_return_pct': round(total_return, 1),
        'final_capital': int(final_capital),
        'avg_rr': round(rr, 2)
    }


if __name__ == "__main__":
    print("=" * 60)
    print("USDJPY バックテスト: DCブレイク + EMA200 + ADX(>25) + Kelly")
    print("=" * 60)

    # データ読み込み
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ohlc', 'USDJPY_H4.csv')
    df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    print(f"データ: {len(df)}本 ({df.index[0].date()} 〜 {df.index[-1].date()})")

    # バックテスト実行
    trades, equity_curve = run_backtest(df)

    # 指標計算
    metrics = compute_metrics(trades, equity_curve, INITIAL_CAPITAL)

    print(f"\n--- 結果 ---")
    print(f"総トレード数:    {metrics['total_trades']}件")
    print(f"勝率:            {metrics['win_rate']}%")
    print(f"平均RR:          {metrics['avg_rr']}")
    print(f"Sharpe比:        {metrics['sharpe']}")
    print(f"Calmar比:        {metrics['calmar']}")
    print(f"最大DD:          -{metrics['max_dd_pct']}%")
    print(f"総リターン:      +{metrics['total_return_pct']}%")
    print(f"最終資産:        ¥{metrics['final_capital']:,}")

    print(f"\n--- XAUUSDとの比較 ---")
    print(f"{'指標':<15} {'USDJPY':>12} {'XAUUSD':>12}")
    print(f"{'-'*40}")
    print(f"{'Sharpe':<15} {metrics['sharpe']:>12} {'2.250':>12}")
    print(f"{'Calmar':<15} {metrics['calmar']:>12} {'11.534':>12}")
    print(f"{'MDD%':<15} {metrics['max_dd_pct']:>12} {'16.1':>12}")
    print(f"{'勝率%':<15} {metrics['win_rate']:>12} {'61.0':>12}")
    print(f"{'トレード数':<15} {metrics['total_trades']:>12} {'41':>12}")

    # 承認基準チェック（Sharpe > 1.5 かつ Calmar > 5.0）
    approved = metrics['sharpe'] > 1.5 and metrics['calmar'] > 5.0
    print(f"\n--- 承認基準チェック (Sharpe>1.5 かつ Calmar>5.0) ---")
    print(f"結果: {'✅ 承認' if approved else '❌ 非承認'}")

    # 結果をCSVに保存
    if trades:
        trades_df = pd.DataFrame(trades)
        out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'backtest_usdjpy_results.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        trades_df.to_csv(out_path, index=False)
        print(f"\nトレードログ保存: {out_path}")

    # performance_logに追記
    log_path = os.path.join(os.path.dirname(__file__), '..', 'performance_log.csv')
    log_entry = pd.DataFrame([{
        'version': 'USDJPY_DCBreak_EMA200_ADX_Kelly',
        'instrument': 'USDJPY',
        'sharpe': metrics['sharpe'],
        'calmar': metrics['calmar'],
        'max_dd_pct': metrics['max_dd_pct'],
        'win_rate': metrics['win_rate'],
        'total_trades': metrics['total_trades'],
        'final_capital': metrics['final_capital'],
        'approved': approved
    }])
    if os.path.exists(log_path):
        log_entry.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(log_path, index=False)
    print(f"performance_log更新: {log_path}")
