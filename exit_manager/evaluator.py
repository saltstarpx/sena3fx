"""
Exit Manager — 2週間評価スクリプト
=====================================
exit_manager.jsonl を読み込み、出口管理の効果を評価する。

合否ライン（事前固定）:
  1. 最大損失（1トレード）: -150,000円を超えない → PASS/FAIL
  2. 1時間未満の裁量決済:  0回 → PASS/FAIL
  3. 勝ちトレードの中央値: 30,000円以上 → PASS/FAIL（改善方向ならOK）
  4. PF:                    1.5以上 → PASS/FAIL

追加分析:
  - 平均R（勝ち/負け）
  - TP1到達率
  - Giveback Stop 発動回数と回避できた損失
  - ロックアウトでブロックされた回数
  - SL拡大試行のブロック回数
  - 保有時間の分布
  - Kill Switch 発動回数
  - non_textbook イベント数

Usage:
  python -m exit_manager.evaluator --days 14
  python -m exit_manager.cli report --days 14 --output reports/eval.md
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------ #
#  データ読み込み                                                      #
# ------------------------------------------------------------------ #

def load_events(jsonl_path: str, days: int = 14) -> list[dict]:
    """
    JSONL ファイルから指定日数以内のイベントを読み込む。

    Args:
        jsonl_path: exit_manager.jsonl のパス
        days:       集計期間（日数）

    Returns:
        list[dict]: イベントレコードのリスト（時系列順）
    """
    path = Path(jsonl_path)
    if not path.exists():
        return []

    cutoff = datetime.utcnow() - timedelta(days=days)
    events = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                ts_str = r.get('timestamp', '')
                if ts_str:
                    ts = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%SZ')
                    if ts >= cutoff:
                        events.append(r)
            except (json.JSONDecodeError, ValueError):
                continue

    return sorted(events, key=lambda x: x.get('timestamp', ''))


# ------------------------------------------------------------------ #
#  メトリクス計算                                                      #
# ------------------------------------------------------------------ #

def compute_metrics(events: list[dict]) -> dict:
    """
    JSONL イベントから評価メトリクスを計算する。

    トレードごとに:
      - TRADE_REGISTERED → TRADE_CLOSED のペアを特定
      - 実現P&L を PARTIAL_CLOSE + TRADE_CLOSED から集計
      - 保有時間を計算

    Returns:
        dict: 全評価メトリクス
    """
    if not events:
        return {'error': 'no_events'}

    # trade_id ごとにイベントをグループ化
    by_trade: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        tid = e.get('trade_id', '')
        if tid:
            by_trade[tid].append(e)

    # 完結したトレード（REGISTERED + CLOSED）を抽出
    completed = []
    for tid, tevents in by_trade.items():
        sorted_evts = sorted(tevents, key=lambda x: x.get('timestamp', ''))
        reg = next((e for e in sorted_evts if e['event'] == 'TRADE_REGISTERED'), None)
        closed = next((e for e in sorted_evts if e['event'] == 'TRADE_CLOSED'), None)
        if reg and closed:
            completed.append((tid, sorted_evts, reg, closed))

    if not completed:
        return {'error': 'no_completed_trades', 'total_events': len(events)}

    # P&L と保有時間を計算
    trade_pnls = []
    trade_durations_h = []
    tp1_count = 0
    giveback_count = 0
    lockout_blocked_count = 0
    sl_widen_blocked_count = 0
    kill_switch_count = 0
    non_textbook_count = 0

    for tid, tevents, reg, closed in completed:
        # P&L を合計（PARTIAL_CLOSE + TRADE_CLOSED の pnl_jpy）
        pnl_jpy = 0.0
        for e in tevents:
            if e['event'] in ('PARTIAL_CLOSE', 'TP1_HIT', 'TRADE_CLOSED'):
                pnl_jpy += float(e.get('realized_pnl_jpy', e.get('pnl_jpy', 0.0)))

        # 保有時間
        try:
            entry_ts = datetime.strptime(reg['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
            close_ts = datetime.strptime(closed['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
            hold_h = (close_ts - entry_ts).total_seconds() / 3600.0
        except (ValueError, KeyError):
            hold_h = 0.0

        trade_pnls.append(pnl_jpy)
        trade_durations_h.append(hold_h)

        # イベントカウント
        for e in tevents:
            evt = e['event']
            if evt == 'TP1_HIT':
                tp1_count += 1
            elif evt == 'TRADE_CLOSED' and 'giveback' in e.get('reason', ''):
                giveback_count += 1
            elif evt == 'LOCKOUT_BLOCKED':
                lockout_blocked_count += 1
            elif evt == 'VALIDATION_WARNING' and 'SL拡大' in e.get('warning', ''):
                sl_widen_blocked_count += 1
            elif evt == 'KILL_SWITCH':
                kill_switch_count += 1
            if e.get('non_textbook'):
                non_textbook_count += 1

    # 基本統計
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    max_loss = min(losses) if losses else 0.0
    median_win = statistics.median(wins) if wins else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    # 1時間未満の決済（裁量的損切）
    under_1h_exits = sum(1 for h in trade_durations_h if h < 1.0)

    # 1R（¥150,000）ベースの平均R
    one_r = 150_000.0
    win_rs = [p / one_r for p in wins] if wins else []
    loss_rs = [p / one_r for p in losses] if losses else []

    # 合否判定
    pass_max_loss = (max_loss >= -150_000) or not losses
    pass_lockout  = (under_1h_exits == 0)
    pass_median   = (median_win >= 30_000) or not wins
    pass_pf       = (pf >= 1.5)

    return {
        'period_days':             14,
        'total_trades':            len(completed),
        'win_trades':              len(wins),
        'loss_trades':             len(losses),
        'win_rate_pct':            round(100 * len(wins) / len(completed), 1) if completed else 0,
        'profit_factor':           round(pf, 3) if pf != float('inf') else 'inf',
        'gross_profit_jpy':        round(gross_profit),
        'gross_loss_jpy':          round(gross_loss),
        'net_pnl_jpy':             round(sum(trade_pnls)),
        'max_loss_jpy':            round(max_loss),
        'median_win_jpy':          round(median_win),
        'avg_win_r':               round(statistics.mean(win_rs), 2) if win_rs else 0,
        'avg_loss_r':              round(statistics.mean(loss_rs), 2) if loss_rs else 0,
        'avg_hold_hours':          round(statistics.mean(trade_durations_h), 1) if trade_durations_h else 0,
        'under_1h_exits':          under_1h_exits,
        'tp1_count':               tp1_count,
        'tp1_rate_pct':            round(100 * tp1_count / len(completed), 1) if completed else 0,
        'giveback_stop_count':     giveback_count,
        'lockout_blocked_count':   lockout_blocked_count,
        'sl_widen_blocked_count':  sl_widen_blocked_count,
        'kill_switch_count':       kill_switch_count,
        'non_textbook_count':      non_textbook_count,
        # 合否判定
        'pass_max_loss':           pass_max_loss,
        'pass_lockout_exit':       pass_lockout,
        'pass_median_win':         pass_median,
        'pass_profit_factor':      pass_pf,
        'overall_pass':            all([pass_max_loss, pass_lockout, pass_median, pass_pf]),
    }


# ------------------------------------------------------------------ #
#  レポート生成                                                        #
# ------------------------------------------------------------------ #

def generate_report(
    metrics: dict,
    output_path: Optional[str] = None,
) -> str:
    """
    評価メトリクスから Markdown レポートを生成する。

    Args:
        metrics:     compute_metrics() の返り値
        output_path: 保存先パス（None の場合は保存しない）

    Returns:
        str: Markdown テキスト
    """
    now_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        '# Exit Manager — 評価レポート',
        f'生成: {now_str}',
        '',
    ]

    if 'error' in metrics:
        lines.append(f'⚠️ エラー: {metrics["error"]}')
        report = '\n'.join(lines)
        if output_path:
            Path(output_path).write_text(report, encoding='utf-8')
        return report

    # 合否サマリー
    overall = '✅ PASS' if metrics.get('overall_pass') else '❌ FAIL'
    lines += [
        f'## 総合判定: {overall}',
        '',
        '| チェック項目 | 基準 | 結果 | 判定 |',
        '|-------------|------|------|------|',
        f'| 最大損失（1トレード） | ≥ -¥150,000 | ¥{metrics["max_loss_jpy"]:,} | {"✅" if metrics["pass_max_loss"] else "❌"} |',
        f'| 1時間未満の裁量決済 | 0回 | {metrics["under_1h_exits"]}回 | {"✅" if metrics["pass_lockout_exit"] else "❌"} |',
        f'| 勝ちトレード中央値 | ≥ ¥30,000 | ¥{metrics["median_win_jpy"]:,} | {"✅" if metrics["pass_median_win"] else "❌"} |',
        f'| プロフィットファクター | ≥ 1.5 | {metrics["profit_factor"]} | {"✅" if metrics["pass_profit_factor"] else "❌"} |',
        '',
    ]

    # トレード統計
    lines += [
        '## トレード統計',
        '',
        f'| 項目 | 値 |',
        f'|------|-----|',
        f'| 集計期間 | {metrics["period_days"]}日 |',
        f'| 総トレード数 | {metrics["total_trades"]} |',
        f'| 勝ち | {metrics["win_trades"]} ({metrics["win_rate_pct"]}%) |',
        f'| 負け | {metrics["loss_trades"]} |',
        f'| 総利益 | ¥{metrics["gross_profit_jpy"]:,} |',
        f'| 総損失 | -¥{metrics["gross_loss_jpy"]:,} |',
        f'| 純損益 | ¥{metrics["net_pnl_jpy"]:,} |',
        f'| 平均保有時間 | {metrics["avg_hold_hours"]}h |',
        f'| 平均勝ちR | {metrics["avg_win_r"]:+.2f}R |',
        f'| 平均負けR | {metrics["avg_loss_r"]:+.2f}R |',
        '',
    ]

    # Exit Manager 効果
    lines += [
        '## Exit Manager 効果',
        '',
        f'| 項目 | 回数 |',
        f'|------|------|',
        f'| TP1 到達（50%利確） | {metrics["tp1_count"]}回 ({metrics["tp1_rate_pct"]}%) |',
        f'| Giveback Stop 発動 | {metrics["giveback_stop_count"]}回 |',
        f'| ロックアウトブロック | {metrics["lockout_blocked_count"]}回 |',
        f'| SL拡大ブロック | {metrics["sl_widen_blocked_count"]}回 |',
        f'| Kill Switch 発動 | {metrics["kill_switch_count"]}回 |',
        f'| non_textbook ルール適用 | {metrics["non_textbook_count"]}回 |',
        '',
    ]

    report = '\n'.join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding='utf-8')
        print(f'レポートを保存しました: {output_path}')

    return report


# ------------------------------------------------------------------ #
#  CLI エントリーポイント                                              #
# ------------------------------------------------------------------ #

def main():
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Exit Manager 2週間評価')
    parser.add_argument('--days', type=int, default=14, help='集計期間（日数）')
    parser.add_argument('--output', default=None, help='出力Markdownファイルパス')
    parser.add_argument(
        '--jsonl',
        default='./logs/exit_manager.jsonl',
        help='JSONL ログファイルパス',
    )
    args = parser.parse_args()

    events = load_events(args.jsonl, days=args.days)
    metrics = compute_metrics(events)
    report = generate_report(metrics, output_path=args.output)
    print(report)


if __name__ == '__main__':
    main()
