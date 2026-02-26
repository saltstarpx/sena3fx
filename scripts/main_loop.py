"""
完全自律型トレードPDCAエージェント v3.0 メインループ
=====================================================
やがみメソッド準拠 + ティックレベルバックテスト統合

[Plan] 戦略生成: やがみ5条件ベースの戦略バリエーション
[Do]   バックテスト: 全戦略を複数時間足でテスト
[Check] 分析: 結果をスコアリング、合格判定
[Act]  改善: 知見を蓄積、レポート生成
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from lib.backtest import BacktestEngine
from lib.walk_forward import run_walk_forward
from lib.yagami import (
    sig_yagami_A, sig_yagami_B,
    sig_yagami_reversal_only, sig_yagami_double_bottom,
    sig_yagami_pattern_break, sig_yagami_london_ny,
)
from lib.candle import detect_single_candle, detect_price_action
from lib.patterns import detect_chart_patterns

# インジケーター戦略
from lib.indicators import (
    sig_sma, sig_rsi, sig_bb, sig_macd,
    sig_rsi_sma, sig_macd_rsi, sig_bb_rsi,
    Ind,
)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
KNOWLEDGE_FILE = os.path.join(BASE_DIR, 'knowledge.json')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_knowledge():
    """蓄積知見を読み込み"""
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE) as f:
            return json.load(f)
    return {
        'cycle_count': 0,
        'best_strategies': [],
        'banned_strategies': [],
        'insights': [],
    }


def save_knowledge(knowledge):
    with open(KNOWLEDGE_FILE, 'w') as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False, default=str)


def load_data():
    """利用可能なデータを読み込み（ティック or OHLC）"""
    from scripts.fetch_data import load_ticks, ticks_to_ohlc, generate_sample_ohlc

    ticks = load_ticks()
    if ticks is not None and len(ticks) > 10000:
        print(f"ティックデータ: {len(ticks):,} ticks")
        bars_1h = ticks_to_ohlc(ticks, '1h')
        bars_4h = ticks_to_ohlc(ticks, '4h')
        return {'ticks': ticks, '1h': bars_1h, '4h': bars_4h}

    # フォールバック: サンプルデータ
    print("ティックデータなし → サンプルデータで実行")
    bars_1h = generate_sample_ohlc(720, '1h')  # 30日分
    bars_4h = generate_sample_ohlc(180, '4h')
    return {'ticks': None, '1h': bars_1h, '4h': bars_4h}


def build_strategies():
    """
    テスト対象戦略リストを構築。
    やがみメソッド戦略 + 従来のインジケーター戦略
    """
    strategies = []

    # === やがみメソッド戦略（コア） ===
    strategies.append(('YagamiA', sig_yagami_A()))
    strategies.append(('YagamiB', sig_yagami_B()))
    strategies.append(('Yagami_Reversal', sig_yagami_reversal_only()))
    strategies.append(('Yagami_DblBottom', sig_yagami_double_bottom()))
    strategies.append(('Yagami_PatternBrk', sig_yagami_pattern_break()))
    strategies.append(('Yagami_LonNY', sig_yagami_london_ny()))

    # 4h版
    strategies.append(('YagamiA_4h', sig_yagami_A('4h')))
    strategies.append(('YagamiB_4h', sig_yagami_B('4h')))
    strategies.append(('Yagami_Reversal_4h', sig_yagami_reversal_only('4h')))
    strategies.append(('Yagami_LonNY_4h', sig_yagami_london_ny('4h')))

    # === 従来のインジケーター戦略 ===
    for fast, slow in [(5,20), (10,50), (20,100)]:
        strategies.append((f'SMA({fast}/{slow})', sig_sma(fast, slow)))

    for p, os_lv, ob_lv in [(14,30,70), (7,25,75), (21,30,70)]:
        strategies.append((f'RSI({p},{os_lv}/{ob_lv})', sig_rsi(p, os_lv, ob_lv)))

    strategies.append(('MACD(12/26/9)', sig_macd(12, 26, 9)))
    strategies.append(('MACD(8/26/5)', sig_macd(8, 26, 5)))
    strategies.append(('MACD(8/30/5)', sig_macd(8, 30, 5)))

    strategies.append(('RSI14+SMA50', sig_rsi_sma(14, 30, 70, 50)))
    strategies.append(('MACD+RSI50', sig_macd_rsi(12, 26, 9, 14, 50)))
    strategies.append(('BB20+RSI', sig_bb_rsi(20, 2.0, 14, 30, 70)))

    return strategies


def run_pdca_cycle():
    """PDCAサイクル1回実行"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    knowledge = load_knowledge()
    cycle_num = knowledge['cycle_count'] + 1

    print("=" * 75)
    print(f"PDCA サイクル #{cycle_num}")
    print(f"実行時刻: {datetime.now()}")
    print(f"エンジン: やがみメソッド v3.0")
    print("=" * 75)

    # ===== [PLAN] =====
    print("\n[PLAN] 戦略生成")
    strategies = build_strategies()
    print(f"  テスト戦略数: {len(strategies)}")

    # ===== [DO] =====
    print("\n[DO] バックテスト実行")
    data = load_data()

    engine_dynamic = BacktestEngine(
        init_cash=5_000_000, risk_pct=0.02,
        default_sl_atr=2.0, default_tp_atr=4.0,
        use_dynamic_sl=True,
    )
    engine_fixed = BacktestEngine(
        init_cash=5_000_000, risk_pct=0.02,
        default_sl_atr=2.0, default_tp_atr=4.0,
        use_dynamic_sl=False,
    )

    all_results = []

    for tf in ['1h', '4h']:
        bars = data.get(tf)
        if bars is None or len(bars) < 30:
            continue

        print(f"\n--- 時間足: {tf} ({len(bars)} bars) ---")

        for name, sfunc in strategies:
            full_name = f"{name}_{tf}"
            try:
                # 動的SL版
                r = engine_dynamic.run(bars, sfunc, freq=tf, name=f"{full_name}_dynSL")
                if r and r.get('total_trades', 0) > 0:
                    r_clean = {k: v for k, v in r.items() if k != 'trades'}
                    all_results.append(r_clean)
                    status = "PASS" if r['passed'] else "fail"
                    print(f"  {full_name}_dynSL: "
                          f"Ret={r['total_return_pct']:+.1f}% "
                          f"WR={r['win_rate_pct']:.0f}% "
                          f"PF={r['profit_factor']:.2f} "
                          f"RR={r['rr_ratio']:.1f} "
                          f"DD={r['max_drawdown_pct']:.1f}% "
                          f"N={r['total_trades']} [{status}]")

                # 固定SL版
                r2 = engine_fixed.run(bars, sfunc, freq=tf, name=f"{full_name}_fixSL")
                if r2 and r2.get('total_trades', 0) > 0:
                    r2_clean = {k: v for k, v in r2.items() if k != 'trades'}
                    all_results.append(r2_clean)

            except Exception as e:
                print(f"  {full_name}: エラー - {e}")

    # ===== [DO-2] ウォークフォワードテスト =====
    # データが十分にある場合のみ実行（1h: 650本以上）
    wf_results_df = None
    bars_1h = data.get('1h')
    if bars_1h is not None and len(bars_1h) >= 650:
        print("\n[DO-WF] ウォークフォワードテスト（1h）")
        # やがみメソッド戦略のみ対象（計算コスト削減）
        yagami_strategies = [s for s in strategies
                             if s[0].startswith('Yagami') and '_4h' not in s[0]]
        wf_results_df = run_walk_forward(
            bars_1h, yagami_strategies,
            freq='1h',
            in_sample_bars=500,
            out_of_sample_bars=150,
            step_bars=100,
        )
        wf_passed = wf_results_df[wf_results_df['passed'] == True]
        print(f"  WFテスト合格: {len(wf_passed)}/{len(wf_results_df)} 戦略")
    else:
        print("\n[DO-WF] データ不足のためウォークフォワードテストをスキップ "
              f"(必要: 650本, 現在: {len(bars_1h) if bars_1h is not None else 0}本)")

    # ===== [CHECK] =====
    print("\n[CHECK] 結果分析")

    if not all_results:
        print("  テスト結果なし")
        return

    results_df = pd.DataFrame(all_results)

    # CSV保存
    csv_path = os.path.join(RESULTS_DIR, f'pdca_{cycle_num}_{ts}.csv')
    results_df.to_csv(csv_path, index=False)

    passed = results_df[results_df['passed'] == True]
    yagami_results = results_df[results_df['strategy'].str.startswith('Yagami')]

    print(f"\n  全戦略数: {len(results_df)}")
    print(f"  トレードあり: {len(results_df[results_df['total_trades'] > 0])}")
    print(f"  合格戦略数: {len(passed)}")

    if len(yagami_results) > 0:
        print(f"\n  --- やがみメソッド戦略 ---")
        yagami_with_trades = yagami_results[yagami_results['total_trades'] > 0]
        if len(yagami_with_trades) > 0:
            best_yagami = yagami_with_trades.nlargest(5, 'profit_factor')
            for _, r in best_yagami.iterrows():
                print(f"    {r['strategy']}: "
                      f"PF={r['profit_factor']:.2f} "
                      f"WR={r['win_rate_pct']:.0f}% "
                      f"RR={r['rr_ratio']:.1f} "
                      f"N={r['total_trades']}")

    if len(passed) > 0:
        print(f"\n  --- 合格戦略 ---")
        for _, r in passed.iterrows():
            print(f"    {r['strategy']}: "
                  f"PF={r['profit_factor']:.2f} "
                  f"WR={r['win_rate_pct']:.0f}% "
                  f"Ret={r['total_return_pct']:+.1f}%")

    # ===== [ACT] =====
    print("\n[ACT] 知見蓄積・レポート生成")

    # 知見更新
    knowledge['cycle_count'] = cycle_num
    if wf_results_df is not None and len(wf_results_df) > 0:
        wf_passed = wf_results_df[wf_results_df['passed'] == True]
        if len(wf_passed) > 0:
            if 'wf_passed_strategies' not in knowledge:
                knowledge['wf_passed_strategies'] = []
            for _, r in wf_passed.iterrows():
                knowledge['wf_passed_strategies'].append({
                    'name': r['strategy'],
                    'oos_pf': r['avg_oos_pf'],
                    'pf_ratio': r['pf_ratio'],
                    'oos_wr': r['avg_oos_wr'],
                    'consistency': r['oos_consistency'],
                    'cycle': cycle_num,
                })

    if len(passed) > 0:
        for _, r in passed.iterrows():
            knowledge['best_strategies'].append({
                'name': r['strategy'],
                'pf': r['profit_factor'],
                'wr': r['win_rate_pct'],
                'dd': r['max_drawdown_pct'],
                'n': r['total_trades'],
                'cycle': cycle_num,
            })

    # インサイト自動生成
    insights = []
    if len(results_df) > 0:
        avg_ret = results_df['total_return_pct'].mean()
        insights.append(f"サイクル#{cycle_num}: 平均リターン {avg_ret:.2f}%")

        yagami_avg = yagami_results['total_return_pct'].mean() if len(yagami_results) > 0 else 0
        other_avg = results_df[~results_df['strategy'].str.startswith('Yagami')]['total_return_pct'].mean()
        if not np.isnan(yagami_avg) and not np.isnan(other_avg):
            insights.append(
                f"やがみ戦略 平均リターン: {yagami_avg:.2f}% vs "
                f"インジケーター戦略: {other_avg:.2f}%"
            )

    knowledge['insights'].extend(insights)
    save_knowledge(knowledge)

    # レポート生成
    report = generate_report(cycle_num, results_df, passed, knowledge,
                             wf_results_df=wf_results_df)
    report_path = os.path.join(REPORTS_DIR, f'pdca_cycle_{cycle_num}_{ts}.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n  レポート保存: {report_path}")
    print(f"  知見保存: {KNOWLEDGE_FILE}")

    print(f"\n{'='*75}")
    print(f"PDCA サイクル #{cycle_num} 完了")
    print("=" * 75)

    return results_df


def generate_report(cycle_num, results_df, passed, knowledge,
                    wf_results_df=None):
    """Markdownレポート生成"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    report = f"""# PDCA サイクル #{cycle_num} レポート

実行日時: {ts}
エンジン: やがみメソッド v3.0

## サマリー

| 指標 | 値 |
|------|-----|
| テスト戦略数 | {len(results_df)} |
| 合格戦略数 | {len(passed)} |
| 平均リターン | {results_df['total_return_pct'].mean():.2f}% |
| 平均勝率 | {results_df['win_rate_pct'].mean():.1f}% |
| 平均PF | {results_df['profit_factor'].mean():.2f} |
| 平均DD | {results_df['max_drawdown_pct'].mean():.1f}% |

## やがみメソッド戦略結果

"""
    yagami_results = results_df[results_df['strategy'].str.startswith('Yagami')]
    if len(yagami_results) > 0:
        yagami_with_trades = yagami_results[yagami_results['total_trades'] > 0]
        if len(yagami_with_trades) > 0:
            report += "| 戦略 | PF | 勝率 | RR | DD | N | リターン |\n"
            report += "|------|-----|------|-----|-----|---|--------|\n"
            for _, r in yagami_with_trades.nlargest(10, 'profit_factor').iterrows():
                report += (f"| {r['strategy']} | {r['profit_factor']:.2f} | "
                          f"{r['win_rate_pct']:.0f}% | {r['rr_ratio']:.1f} | "
                          f"{r['max_drawdown_pct']:.1f}% | {r['total_trades']} | "
                          f"{r['total_return_pct']:+.1f}% |\n")

    if len(passed) > 0:
        report += "\n## 合格戦略（バックテスト）\n\n"
        report += "| 戦略 | PF | 勝率 | DD | N | リターン |\n"
        report += "|------|-----|------|-----|---|--------|\n"
        for _, r in passed.iterrows():
            report += (f"| {r['strategy']} | {r['profit_factor']:.2f} | "
                      f"{r['win_rate_pct']:.0f}% | "
                      f"{r['max_drawdown_pct']:.1f}% | {r['total_trades']} | "
                      f"{r['total_return_pct']:+.1f}% |\n")

    if wf_results_df is not None and len(wf_results_df) > 0:
        report += "\n## ウォークフォワードテスト結果\n\n"
        report += "| 戦略 | IS_PF | OOS_PF | PF比率 | OOS勝率 | 一貫性 | 判定 |\n"
        report += "|------|-------|--------|--------|---------|--------|------|\n"
        for _, r in wf_results_df.iterrows():
            status = "PASS" if r['passed'] else "fail"
            report += (f"| {r['strategy']} | {r['avg_is_pf']:.2f} | "
                      f"{r['avg_oos_pf']:.2f} | {r['pf_ratio']:.2f} | "
                      f"{r['avg_oos_wr']:.0f}% | {r['oos_consistency']:.0f}% | "
                      f"{status} |\n")
        report += "\n> PF比率 ≥ 0.6 かつ OOS PF ≥ 1.2 かつ 一貫性 ≥ 60% が合格\n"

    report += f"""
## 蓄積知見

"""
    for insight in knowledge.get('insights', [])[-10:]:
        report += f"- {insight}\n"

    report += """
## 次サイクルへの提言

1. データ期間の延長（より多くのティックデータ取得）
2. やがみA評価戦略のパラメータ微調整
3. セッション別フィルターの最適化
4. MTF（マルチタイムフレーム）分析の強化
"""
    return report


if __name__ == '__main__':
    run_pdca_cycle()
