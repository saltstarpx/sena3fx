"""
exit_manager の evaluator テスト

load_events / compute_metrics / generate_report の動作を検証。
仕様書 Section 8 の合否判定基準に準拠。
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from exit_manager.evaluator import load_events, compute_metrics, generate_report


# ------------------------------------------------------------------ #
#  ヘルパー                                                            #
# ------------------------------------------------------------------ #

def _write_jsonl(path: Path, records: list[dict]) -> None:
    """テスト用 JSONL ファイルを書き出す"""
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def _ts(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def _make_trade_events(
    trade_id: str,
    registered_at: datetime,
    closed_at: datetime,
    pnl_jpy: float,
    close_reason: str = 'sl_hit',
    with_tp1: bool = False,
    tp1_pnl_jpy: float = 0.0,
) -> list[dict]:
    """完結した1トレード分のイベントリストを生成する"""
    events = [
        {
            'timestamp': _ts(registered_at),
            'event': 'TRADE_REGISTERED',
            'trade_id': trade_id,
            'symbol': 'XAU_USD',
            'direction': 'LONG',
            'entry_price': 2890.0,
            'initial_sl': 2875.0,
            'units': 65,
            'risk_jpy': 148000,
        },
    ]
    if with_tp1:
        tp1_at = registered_at + timedelta(hours=2)
        events.append({
            'timestamp': _ts(tp1_at),
            'event': 'TP1_HIT',
            'trade_id': trade_id,
            'realized_pnl_jpy': tp1_pnl_jpy,
        })
    events.append({
        'timestamp': _ts(closed_at),
        'event': 'TRADE_CLOSED',
        'trade_id': trade_id,
        'reason': close_reason,
        'pnl_jpy': pnl_jpy,
        'hold_hours': (closed_at - registered_at).total_seconds() / 3600,
    })
    return events


# ------------------------------------------------------------------ #
#  load_events テスト                                                  #
# ------------------------------------------------------------------ #

class TestLoadEvents:

    def test_loads_recent_events(self, tmp_path):
        """14日以内のイベントを読み込む"""
        now = datetime.utcnow()
        records = [
            {'timestamp': _ts(now - timedelta(days=1)), 'event': 'TRADE_CLOSED', 'trade_id': 'A'},
            {'timestamp': _ts(now - timedelta(days=7)), 'event': 'TRADE_CLOSED', 'trade_id': 'B'},
        ]
        p = tmp_path / 'test.jsonl'
        _write_jsonl(p, records)
        events = load_events(str(p), days=14)
        assert len(events) == 2

    def test_excludes_old_events(self, tmp_path):
        """14日より古いイベントは除外される"""
        now = datetime.utcnow()
        records = [
            {'timestamp': _ts(now - timedelta(days=5)), 'event': 'TRADE_CLOSED', 'trade_id': 'A'},
            {'timestamp': _ts(now - timedelta(days=20)), 'event': 'TRADE_CLOSED', 'trade_id': 'B'},
        ]
        p = tmp_path / 'test.jsonl'
        _write_jsonl(p, records)
        events = load_events(str(p), days=14)
        assert len(events) == 1
        assert events[0]['trade_id'] == 'A'

    def test_returns_empty_if_file_missing(self, tmp_path):
        """ファイルが存在しない場合は空リストを返す"""
        result = load_events(str(tmp_path / 'nonexistent.jsonl'), days=14)
        assert result == []

    def test_skips_malformed_lines(self, tmp_path):
        """不正なJSON行をスキップする"""
        now = datetime.utcnow()
        p = tmp_path / 'test.jsonl'
        with open(p, 'w') as f:
            f.write('NOT VALID JSON\n')
            f.write(json.dumps({'timestamp': _ts(now), 'event': 'TEST', 'trade_id': 'X'}) + '\n')
        events = load_events(str(p), days=14)
        assert len(events) == 1
        assert events[0]['trade_id'] == 'X'

    def test_returns_sorted_by_timestamp(self, tmp_path):
        """タイムスタンプ順に並ぶ"""
        now = datetime.utcnow()
        records = [
            {'timestamp': _ts(now - timedelta(hours=2)), 'event': 'B', 'trade_id': '2'},
            {'timestamp': _ts(now - timedelta(hours=5)), 'event': 'A', 'trade_id': '1'},
        ]
        p = tmp_path / 'test.jsonl'
        _write_jsonl(p, records)
        events = load_events(str(p), days=14)
        assert events[0]['trade_id'] == '1'
        assert events[1]['trade_id'] == '2'

    def test_empty_file_returns_empty(self, tmp_path):
        """空ファイルは空リストを返す"""
        p = tmp_path / 'empty.jsonl'
        p.write_text('')
        result = load_events(str(p), days=14)
        assert result == []


# ------------------------------------------------------------------ #
#  compute_metrics テスト                                              #
# ------------------------------------------------------------------ #

class TestComputeMetrics:

    def test_empty_events_returns_error(self):
        """イベントがない場合はエラーを返す"""
        result = compute_metrics([])
        assert 'error' in result

    def test_no_completed_trades(self, tmp_path):
        """完結したトレードがない場合はエラーを返す"""
        events = [
            {'timestamp': _ts(datetime.utcnow()), 'event': 'TRADE_REGISTERED', 'trade_id': 'X'},
        ]
        result = compute_metrics(events)
        assert 'error' in result

    def test_single_win_trade(self):
        """勝ちトレード1件の基本集計"""
        now = datetime.utcnow()
        events = _make_trade_events(
            trade_id='001',
            registered_at=now - timedelta(hours=4),
            closed_at=now - timedelta(hours=1),
            pnl_jpy=160_000,
            close_reason='tp_trailing',
        )
        metrics = compute_metrics(events)
        assert metrics['total_trades'] == 1
        assert metrics['win_trades'] == 1
        assert metrics['loss_trades'] == 0
        assert metrics['net_pnl_jpy'] == 160_000
        assert metrics['profit_factor'] == 'inf'

    def test_single_loss_trade(self):
        """負けトレード1件の基本集計"""
        now = datetime.utcnow()
        events = _make_trade_events(
            trade_id='002',
            registered_at=now - timedelta(hours=8),
            closed_at=now - timedelta(hours=2),
            pnl_jpy=-148_000,
            close_reason='sl_hit',
        )
        metrics = compute_metrics(events)
        assert metrics['total_trades'] == 1
        assert metrics['win_trades'] == 0
        assert metrics['loss_trades'] == 1
        assert metrics['max_loss_jpy'] == -148_000
        assert metrics['net_pnl_jpy'] == -148_000

    def test_pass_all_criteria(self):
        """全合否基準PASSのシナリオ"""
        now = datetime.utcnow()
        # 3勝1敗（PF > 1.5, 最大損失 ≥ -150k, 中央値 ≥ 30k, 1h未満 = 0）
        events = []
        events += _make_trade_events('T1', now - timedelta(hours=30), now - timedelta(hours=20), 200_000)
        events += _make_trade_events('T2', now - timedelta(hours=25), now - timedelta(hours=15), 150_000)
        events += _make_trade_events('T3', now - timedelta(hours=20), now - timedelta(hours=10), 100_000)
        events += _make_trade_events('T4', now - timedelta(hours=10), now - timedelta(hours=3), -100_000)
        metrics = compute_metrics(events)
        assert metrics['total_trades'] == 4
        assert metrics['pass_max_loss'] is True
        assert metrics['pass_lockout_exit'] is True
        assert metrics['pass_median_win'] is True
        assert metrics['pass_profit_factor'] is True
        assert metrics['overall_pass'] is True

    def test_fail_max_loss(self):
        """最大損失が -150k を超えると FAIL"""
        now = datetime.utcnow()
        events = _make_trade_events(
            trade_id='BIG_LOSS',
            registered_at=now - timedelta(hours=10),
            closed_at=now - timedelta(hours=2),
            pnl_jpy=-200_000,  # -150k 超
        )
        metrics = compute_metrics(events)
        assert metrics['pass_max_loss'] is False
        assert metrics['overall_pass'] is False

    def test_fail_under_1h_exit(self):
        """1時間未満の決済があると FAIL"""
        now = datetime.utcnow()
        # 保有時間 = 30分
        events = _make_trade_events(
            trade_id='QUICK_EXIT',
            registered_at=now - timedelta(minutes=90),
            closed_at=now - timedelta(minutes=60),  # 30分保有
            pnl_jpy=-50_000,
        )
        metrics = compute_metrics(events)
        assert metrics['under_1h_exits'] == 1
        assert metrics['pass_lockout_exit'] is False
        assert metrics['overall_pass'] is False

    def test_fail_low_pf(self):
        """PF < 1.5 → FAIL"""
        now = datetime.utcnow()
        events = []
        events += _make_trade_events('W1', now - timedelta(hours=20), now - timedelta(hours=15), 100_000)
        events += _make_trade_events('L1', now - timedelta(hours=15), now - timedelta(hours=10), -80_000)
        events += _make_trade_events('L2', now - timedelta(hours=10), now - timedelta(hours=5), -80_000)
        # PF = 100k / 160k = 0.625
        metrics = compute_metrics(events)
        assert metrics['pass_profit_factor'] is False

    def test_fail_low_median_win(self):
        """勝ちトレード中央値 < 30k → FAIL"""
        now = datetime.utcnow()
        events = []
        events += _make_trade_events('W1', now - timedelta(hours=20), now - timedelta(hours=15), 20_000)
        events += _make_trade_events('W2', now - timedelta(hours=15), now - timedelta(hours=10), 25_000)
        events += _make_trade_events('L1', now - timedelta(hours=10), now - timedelta(hours=5), -100_000)
        # 中央値 = 22.5k < 30k
        metrics = compute_metrics(events)
        assert metrics['pass_median_win'] is False

    def test_tp1_count(self):
        """TP1 到達カウント"""
        now = datetime.utcnow()
        events = _make_trade_events(
            'TP1_TRADE',
            now - timedelta(hours=10),
            now - timedelta(hours=2),
            pnl_jpy=180_000,
            with_tp1=True,
            tp1_pnl_jpy=80_000,
        )
        metrics = compute_metrics(events)
        assert metrics['tp1_count'] == 1
        assert metrics['tp1_rate_pct'] == 100.0

    def test_giveback_stop_count(self):
        """Giveback Stop 発動カウント"""
        now = datetime.utcnow()
        events = _make_trade_events(
            'GIVEBACK',
            now - timedelta(hours=10),
            now - timedelta(hours=2),
            pnl_jpy=120_000,
            close_reason='giveback_stop',
        )
        metrics = compute_metrics(events)
        assert metrics['giveback_stop_count'] == 1

    def test_lockout_blocked_count(self):
        """ロックアウトブロックカウント"""
        now = datetime.utcnow()
        # まず完結したトレードを含むイベント
        events = _make_trade_events('TRD1', now - timedelta(hours=10), now - timedelta(hours=2), 50_000)
        # ロックアウトブロックイベントを同じtrade_idに追加
        events.append({
            'timestamp': _ts(now - timedelta(hours=9)),
            'event': 'LOCKOUT_BLOCKED',
            'trade_id': 'TRD1',
            'attempted_action': 'check_tp1',
        })
        metrics = compute_metrics(events)
        assert metrics['lockout_blocked_count'] == 1

    def test_non_textbook_count(self):
        """non_textbook フラグカウント"""
        now = datetime.utcnow()
        events = _make_trade_events('NTB', now - timedelta(hours=10), now - timedelta(hours=2), 50_000)
        events.append({
            'timestamp': _ts(now - timedelta(hours=8)),
            'event': 'LOCKOUT_BLOCKED',
            'trade_id': 'NTB',
            'non_textbook': True,
        })
        metrics = compute_metrics(events)
        assert metrics['non_textbook_count'] == 1

    def test_multiple_trades_avg_hold(self):
        """平均保有時間の計算"""
        now = datetime.utcnow()
        events = []
        events += _make_trade_events('H1', now - timedelta(hours=20), now - timedelta(hours=16), 50_000)
        # 4時間保有
        events += _make_trade_events('H2', now - timedelta(hours=20), now - timedelta(hours=8), -50_000)
        # 12時間保有 → 平均8h
        metrics = compute_metrics(events)
        assert metrics['avg_hold_hours'] == 8.0


# ------------------------------------------------------------------ #
#  generate_report テスト                                              #
# ------------------------------------------------------------------ #

class TestGenerateReport:

    def test_returns_markdown_string(self):
        """Markdownレポートを文字列で返す"""
        now = datetime.utcnow()
        events = _make_trade_events('T1', now - timedelta(hours=10), now - timedelta(hours=2), 200_000)
        metrics = compute_metrics(events)
        report = generate_report(metrics)
        assert isinstance(report, str)
        assert '# Exit Manager' in report

    def test_pass_shown_in_report(self):
        """PASS が含まれる"""
        now = datetime.utcnow()
        events = []
        events += _make_trade_events('W1', now - timedelta(hours=30), now - timedelta(hours=20), 300_000)
        events += _make_trade_events('L1', now - timedelta(hours=20), now - timedelta(hours=10), -100_000)
        metrics = compute_metrics(events)
        report = generate_report(metrics)
        assert 'PASS' in report or '✅' in report

    def test_fail_shown_when_criteria_not_met(self):
        """基準未達のとき FAIL が含まれる"""
        now = datetime.utcnow()
        events = _make_trade_events(
            'BIG_LOSS',
            now - timedelta(hours=10),
            now - timedelta(hours=2),
            pnl_jpy=-300_000,
        )
        metrics = compute_metrics(events)
        report = generate_report(metrics)
        assert 'FAIL' in report or '❌' in report

    def test_saves_to_file(self, tmp_path):
        """output_path を指定するとファイルに保存する"""
        now = datetime.utcnow()
        events = _make_trade_events('T1', now - timedelta(hours=10), now - timedelta(hours=2), 100_000)
        metrics = compute_metrics(events)
        output = str(tmp_path / 'report.md')
        generate_report(metrics, output_path=output)
        content = Path(output).read_text(encoding='utf-8')
        assert '# Exit Manager' in content

    def test_error_metrics_still_returns_string(self):
        """エラー状態のメトリクスでも文字列を返す"""
        report = generate_report({'error': 'no_events'})
        assert isinstance(report, str)
        assert 'no_events' in report

    def test_report_contains_key_sections(self):
        """レポートにトレード統計セクションが含まれる"""
        now = datetime.utcnow()
        events = []
        events += _make_trade_events('T1', now - timedelta(hours=30), now - timedelta(hours=20), 200_000)
        events += _make_trade_events('T2', now - timedelta(hours=20), now - timedelta(hours=10), -100_000)
        metrics = compute_metrics(events)
        report = generate_report(metrics)
        assert 'トレード統計' in report
        assert 'Exit Manager 効果' in report


# ------------------------------------------------------------------ #
#  統合シナリオ: load → compute → report                              #
# ------------------------------------------------------------------ #

class TestEndToEndEvaluation:

    def test_full_pipeline_pass(self, tmp_path):
        """ファイル読み込みから合否判定まで一気通貫"""
        now = datetime.utcnow()
        all_events = []
        all_events += _make_trade_events(
            'E2E_W1',
            now - timedelta(hours=40),
            now - timedelta(hours=30),
            pnl_jpy=200_000,
            with_tp1=True, tp1_pnl_jpy=80_000,
        )
        all_events += _make_trade_events(
            'E2E_W2',
            now - timedelta(hours=30),
            now - timedelta(hours=20),
            pnl_jpy=180_000,
        )
        all_events += _make_trade_events(
            'E2E_L1',
            now - timedelta(hours=20),
            now - timedelta(hours=10),
            pnl_jpy=-120_000,
        )

        p = tmp_path / 'exit_manager.jsonl'
        _write_jsonl(p, all_events)

        events = load_events(str(p), days=14)
        metrics = compute_metrics(events)
        report = generate_report(metrics)

        assert metrics['total_trades'] == 3
        assert metrics['win_trades'] == 2
        assert metrics['tp1_count'] == 1
        assert metrics['pass_max_loss'] is True
        assert metrics['pass_lockout_exit'] is True
        assert '✅' in report or 'PASS' in report

    def test_full_pipeline_fail_on_big_loss(self, tmp_path):
        """大損失 → 総合FAIL"""
        now = datetime.utcnow()
        all_events = _make_trade_events(
            'BIG',
            now - timedelta(hours=10),
            now - timedelta(hours=2),
            pnl_jpy=-250_000,
        )
        p = tmp_path / 'exit_manager.jsonl'
        _write_jsonl(p, all_events)

        events = load_events(str(p), days=14)
        metrics = compute_metrics(events)
        report = generate_report(metrics)

        assert metrics['overall_pass'] is False
        assert '❌' in report or 'FAIL' in report
