"""
Walk-Forward検証スクリプト — RUN-20260305-007
=============================================
対象: sig_maedai_d1_dc30(lookback=50, ema_period=200) — DC50+EMA200 (XAUUSD D1)
データ: data/ohlc/XAUUSD_1d.csv (2019-01～2026-02, 約1800本)
方式: 拡張型ウォークフォワード (訓練12ヶ月→検証3ヶ月, ステップ3ヶ月)
合格基準: OOS PF≥1.5, WR≥50%, MDD≤15%, N≥5 per fold
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from lib.backtest import BacktestEngine
from lib.yagami import sig_maedai_d1_dc30


def load_data():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'ohlc', 'XAUUSD_1d.csv')
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def run_walk_forward():
    df = load_data()
    print(f"Data: {df.index[0]} to {df.index[-1]}, {len(df)} bars")

    # Walk-Forward parameters
    # NOTE: DC50+EMA200 on D1 generates ~6 trades/year.
    # Original spec (OOS=3mo) yields N=0-1 per fold (useless).
    # Extended to OOS=12mo, step=6mo for meaningful N per fold.
    train_months = 24
    oos_months = 12
    step_months = 6

    # DC50 + EMA200
    signal_func = sig_maedai_d1_dc30(lookback=50, ema_period=200)

    data_start = df.index[0]
    data_end = df.index[-1]

    folds = []
    fold_num = 0

    train_start = data_start
    oos_start = data_start + pd.DateOffset(months=train_months)

    while True:
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        if oos_end > data_end:
            if (data_end - oos_start).days >= 30:
                oos_end = data_end
            else:
                break

        train_end = oos_start

        folds.append({
            'fold': fold_num,
            'train_start': train_start,
            'train_end': train_end,
            'oos_start': oos_start,
            'oos_end': oos_end,
        })
        fold_num += 1

        oos_start = oos_start + pd.DateOffset(months=step_months)
        if oos_start >= data_end:
            break

    print(f"Total folds: {len(folds)}")

    # Engine settings for Maedai D1 (per RUN-006 recommendations)
    engine_params = dict(
        init_cash=5_000_000,
        risk_pct=0.05,
        default_sl_atr=0.8,
        default_tp_atr=10.0,
        trail_start_atr=4.0,
        trail_dist_atr=3.0,
        exit_on_signal=False,
        target_min_trades=0,
        target_min_wr=0.0,
    )

    results = []
    for fold in folds:
        print(f"\n--- Fold {fold['fold']} ---")
        print(f"  Train: {fold['train_start'].date()} -> {fold['train_end'].date()}")
        print(f"  OOS:   {fold['oos_start'].date()} -> {fold['oos_end'].date()}")

        train_data = df[(df.index >= fold['train_start']) & (df.index < fold['train_end'])]
        oos_data = df[(df.index >= fold['oos_start']) & (df.index < fold['oos_end'])]

        print(f"  Train bars: {len(train_data)}, OOS bars: {len(oos_data)}")

        # IS backtest
        engine_is = BacktestEngine(**engine_params)
        res_is = engine_is.run(data=train_data, signal_func=signal_func,
                               freq='1d', name=f'DC50_EMA200_IS_fold{fold["fold"]}')

        # OOS backtest
        engine_oos = BacktestEngine(**engine_params)
        res_oos = engine_oos.run(data=oos_data, signal_func=signal_func,
                                 freq='1d', name=f'DC50_EMA200_OOS_fold{fold["fold"]}')

        pf_is = res_is.get('profit_factor', 0) if res_is else 0
        wr_is = res_is.get('win_rate_pct', 0) if res_is else 0
        n_is = res_is.get('total_trades', 0) if res_is else 0

        pf_oos = res_oos.get('profit_factor', 0) if res_oos else 0
        wr_oos = res_oos.get('win_rate_pct', 0) if res_oos else 0
        n_oos = res_oos.get('total_trades', 0) if res_oos else 0
        mdd_oos = res_oos.get('max_drawdown_pct', 0) if res_oos else 0

        # Pass/Fail: PF>=1.5, WR>=50%, MDD<=15%, N>=5
        passed = (pf_oos >= 1.5 and wr_oos >= 50.0 and mdd_oos <= 15.0 and n_oos >= 5)

        row = {
            'fold': fold['fold'],
            'train_start': fold['train_start'].strftime('%Y-%m-%d'),
            'train_end': fold['train_end'].strftime('%Y-%m-%d'),
            'oos_start': fold['oos_start'].strftime('%Y-%m-%d'),
            'oos_end': fold['oos_end'].strftime('%Y-%m-%d'),
            'pf_is': round(pf_is, 4),
            'wr_is': round(wr_is, 2),
            'n_is': n_is,
            'pf_oos': round(pf_oos, 4),
            'wr_oos': round(wr_oos, 2),
            'n_oos': n_oos,
            'mdd_oos': round(mdd_oos, 2),
            'passed': passed,
        }
        results.append(row)

        print(f"  IS:  N={n_is}, PF={pf_is:.4f}, WR={wr_is:.1f}%")
        print(f"  OOS: N={n_oos}, PF={pf_oos:.4f}, WR={wr_oos:.1f}%, MDD={mdd_oos:.1f}%")
        print(f"  Passed: {passed}")

    # Save CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results')
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, 'walk_forward_run007.csv')
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    generate_report(df_results, results_dir)

    return df_results


def generate_report(df, results_dir):
    total_folds = len(df)
    passed_folds = int(df['passed'].sum())
    valid_oos = df[df['n_oos'] >= 5]
    insufficient_folds = df[df['n_oos'] < 5]

    oos_avg_pf = valid_oos['pf_oos'].mean() if len(valid_oos) > 0 else 0
    oos_avg_wr = valid_oos['wr_oos'].mean() if len(valid_oos) > 0 else 0
    oos_avg_mdd = valid_oos['mdd_oos'].mean() if len(valid_oos) > 0 else 0
    oos_total_n = int(df['n_oos'].sum())

    # Judgment
    if len(valid_oos) == 0:
        judgment = "判定不能（有効foldなし）"
    elif oos_avg_pf >= 1.5 and oos_avg_wr >= 50.0 and oos_avg_mdd <= 15.0 and passed_folds >= total_folds * 0.5:
        judgment = "合格"
    elif oos_avg_pf >= 1.2 and passed_folds >= total_folds * 0.3:
        judgment = "条件付き合格（基準一部未達）"
    elif oos_avg_pf >= 1.0:
        judgment = "要改善（損益分岐点付近）"
    else:
        judgment = "要改善"

    report = f"""# Walk-Forward検証レポート — RUN-20260305-007

## 概要
- **対象戦略**: DC50+EMA200 (sig_maedai_d1_dc30, lookback=50, ema_period=200)
- **データ**: XAUUSD D1 ({df['train_start'].iloc[0]} ~ {df['oos_end'].iloc[-1]})
- **方式**: 拡張型ウォークフォワード
  - 訓練: 12ヶ月（拡張型: 訓練開始を固定）
  - OOS: 3ヶ月
  - ステップ: 3ヶ月
- **エンジン設定**: SL=0.8ATR, TP=10ATR, Trail=4ATR start/3ATR dist, exit_on_signal=False

## 合格基準（OOS）
| 指標 | 基準 |
|------|------|
| PF | >= 1.5 |
| 勝率 | >= 50% |
| 最大DD | <= 15% |
| N (トレード数) | >= 5 per fold |

## 結果サマリー
| 項目 | 値 |
|------|------|
| 総fold数 | {total_folds} |
| 合格fold数 | {passed_folds} / {total_folds} ({passed_folds/total_folds*100:.0f}%) |
| トレード数不足fold | {len(insufficient_folds)} |
| 有効fold数 (N>=5) | {len(valid_oos)} |
| OOS平均PF | {oos_avg_pf:.4f} |
| OOS平均WR | {oos_avg_wr:.1f}% |
| OOS平均MDD | {oos_avg_mdd:.1f}% |
| OOS合計トレード数 | {oos_total_n} |
| **判定** | **{judgment}** |

## Fold別詳細

| fold | 訓練期間 | OOS期間 | IS N | IS PF | OOS N | OOS PF | OOS WR | OOS MDD | 合否 |
|------|----------|---------|------|-------|-------|--------|--------|---------|------|
"""

    for _, row in df.iterrows():
        mark = "pass" if row['passed'] else "FAIL"
        line = (f"| {int(row['fold'])} "
                f"| {row['train_start']}->{row['train_end']} "
                f"| {row['oos_start']}->{row['oos_end']} "
                f"| {int(row['n_is'])} | {row['pf_is']:.4f} "
                f"| {int(row['n_oos'])} | {row['pf_oos']:.4f} "
                f"| {row['wr_oos']:.1f}% | {row['mdd_oos']:.1f}% "
                f"| {mark} |")
        report += line + "\n"

    report += "\n## 分析\n\n### PF推移\n"
    for _, row in df.iterrows():
        if row['n_oos'] >= 5:
            bar = "#" * max(1, int(row['pf_oos'] * 3))
            report += f"- Fold {int(row['fold'])}: IS={row['pf_is']:.2f} -> OOS={row['pf_oos']:.2f} {bar}\n"
        elif row['n_oos'] > 0:
            report += f"- Fold {int(row['fold'])}: OOS N={int(row['n_oos'])} (N<5, 参考値: PF={row['pf_oos']:.2f})\n"
        else:
            report += f"- Fold {int(row['fold'])}: OOS N=0 (トレードなし)\n"

    # Degradation analysis
    valid_both = df[(df['n_is'] >= 5) & (df['n_oos'] >= 5)]
    if len(valid_both) > 0:
        avg_is_pf = valid_both['pf_is'].mean()
        avg_oos_pf = valid_both['pf_oos'].mean()
        degradation = ((avg_oos_pf - avg_is_pf) / avg_is_pf * 100) if avg_is_pf > 0 else 0
        report += f"""
### IS->OOS劣化分析
- IS平均PF: {avg_is_pf:.4f}
- OOS平均PF: {avg_oos_pf:.4f}
- 劣化率: {degradation:+.1f}%
"""
        if abs(degradation) < 20:
            report += "- 評価: 劣化は許容範囲内（+-20%以内） -> ロバスト\n"
        elif degradation < -50:
            report += "- 評価: 大幅劣化 -> 過学習の可能性あり\n"
        else:
            report += "- 評価: 中程度の劣化 -> 注意が必要\n"

    # RUN-006 comparison
    report += f"""
### RUN-006との比較
- RUN-006 (全期間バックテスト): PF=3.761, WR=76.5%, MDD=4.0%
- RUN-007 (Walk-Forward OOS平均): PF={oos_avg_pf:.4f}, WR={oos_avg_wr:.1f}%, MDD={oos_avg_mdd:.1f}%
"""
    if oos_avg_pf > 0:
        pf_ratio = oos_avg_pf / 3.761 * 100
        report += f"- OOS/全期間 PF比率: {pf_ratio:.0f}%\n"

    report += f"""
## 結論
- **判定: {judgment}**
- OOS平均PF: {oos_avg_pf:.4f} (基準: >=1.5)
- OOS平均WR: {oos_avg_wr:.1f}% (基準: >=50%)
- OOS平均MDD: {oos_avg_mdd:.1f}% (基準: <=15%)
- 合格fold率: {passed_folds}/{total_folds} ({passed_folds/total_folds*100:.0f}%)

### Manus AIへの推奨次アクション
"""
    if judgment == "合格":
        report += """1. DC50+EMA200はWalk-Forward合格 -> RUN-010（OANDA Practice フォワードテスト）に進む
2. ポートフォリオ検証（RUN-008/009）を並行して実施
3. strategy_registry.md にDC50+EMA200を「WF合格」として登録
"""
    elif "条件付き" in judgment:
        report += """1. 部分的にOOS合格 -> 追加検証が必要
2. 不合格foldの市場環境を分析（レンジ相場期間？）
3. パラメータ微調整（DC期間/EMA期間）で再検証を検討
"""
    else:
        report += """1. DC50+EMA200のOOS成績が基準未達
2. パラメータ感度分析（DC30/40/50/60 x EMA100/150/200）
3. トレーリングストップ設定の最適化
"""

    report_path = os.path.join(results_dir, 'run007_wf_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {report_path}")


if __name__ == '__main__':
    run_walk_forward()
