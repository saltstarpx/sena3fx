"""
review_manager.py - YAGAMI改 PDCA知見蓄積システム
==================================================
毎日トレードの振り返り、毎週フィードバック改善案の蓄積、
毎月末に知見をまとめてDiscord経由でユーザーに共有する。

スケジュール:
  毎日   21:00 UTC: 当日クローズトレードのサマリー + Discord通知
  毎週日曜 22:00 UTC: 週次フィードバック（改善案）+ Discord通知
  毎月末   23:00 UTC: 月次サマリー + Discord通知（ユーザー確認依頼）
"""
from __future__ import annotations

import calendar
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# ── ディレクトリ ─────────────────────────────────────────────────
_BASE = Path(__file__).parent / "trade_logs"
REVIEWS_DIR = _BASE / "reviews"
DAILY_DIR   = REVIEWS_DIR / "daily"
WEEKLY_DIR  = REVIEWS_DIR / "weekly"
MONTHLY_DIR = REVIEWS_DIR / "monthly"
TRADES_CSV  = _BASE / "mt5_trades.csv"

TRADE_FIELDS = [
    "ticket", "symbol", "dir", "lots",
    "entry_price", "exit_price", "sl", "tp",
    "entry_time", "exit_time", "exit_type",
    "pnl_pips", "pnl_jpy",
]

# バックテスト基準値（ドリフト検出用）
BASELINES = {
    "XAUUSD": {"wr": 0.731, "pf": 3.44},
    "SPX500":  {"wr": 0.698, "pf": 2.03},
    "GBPUSD":  {"wr": 0.694, "pf": 2.29},
    "AUDUSD":  {"wr": 0.648, "pf": 2.19},
    "NZDUSD":  {"wr": 0.623, "pf": 1.78},
}


class ReviewManager:
    """毎日・毎週・毎月のPDCAレビューを管理するクラス。"""

    def __init__(self, notify_func: Callable[[str, int], None]):
        self.notify = notify_func
        self._last_daily:   object = None
        self._last_weekly:  object = None
        self._last_monthly: object = None

        for d in [DAILY_DIR, WEEKLY_DIR, MONTHLY_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # CSVヘッダー初期化
        if not TRADES_CSV.exists():
            with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()

    # ── トレード記録 ──────────────────────────────────────────────

    def log_closed_trade(self, trade: dict):
        """クローズしたトレードをCSVに追記。mt5_bot.pyから呼び出す。"""
        with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
            writer.writerow(trade)
        log.info(f"[ReviewManager] トレード記録: {trade.get('symbol')} "
                 f"{trade.get('exit_type')} pnl={trade.get('pnl_pips')}pips")

    # ── データ読み込み ────────────────────────────────────────────

    def _load_trades(self, filter_fn) -> list[dict]:
        """条件に合うトレードをCSVから読み込む。"""
        result: list[dict] = []
        if not TRADES_CSV.exists():
            return result
        with open(TRADES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    exit_dt = datetime.fromisoformat(row["exit_time"]).replace(
                        tzinfo=timezone.utc
                    )
                    if filter_fn(exit_dt):
                        result.append(row)
                except (KeyError, ValueError):
                    pass
        return result

    def _load_today_trades(self, now: datetime) -> list[dict]:
        return self._load_trades(lambda dt: dt.date() == now.date())

    def _load_week_trades(self, year: int, week: int) -> list[dict]:
        return self._load_trades(
            lambda dt: dt.isocalendar()[:2] == (year, week)
        )

    def _load_month_trades(self, year: int, month: int) -> list[dict]:
        return self._load_trades(
            lambda dt: dt.year == year and dt.month == month
        )

    # ── 統計計算 ──────────────────────────────────────────────────

    def _calc_stats(self, trades: list[dict]) -> dict:
        if not trades:
            return {"total": 0, "wins": 0, "losses": 0, "wr": 0.0,
                    "pf": 0.0, "net_pips": 0.0, "net_jpy": 0.0,
                    "by_symbol": {}, "by_hour": {}}

        wins   = [t for t in trades if t.get("exit_type") == "TP"]
        losses = [t for t in trades if t.get("exit_type") == "SL"]
        total  = len(trades)

        gross_profit = sum(float(t.get("pnl_jpy", 0)) for t in wins)
        gross_loss   = abs(sum(float(t.get("pnl_jpy", 0)) for t in losses))
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

        net_pips = sum(float(t.get("pnl_pips", 0)) for t in trades)
        net_jpy  = sum(float(t.get("pnl_jpy",  0)) for t in trades)

        # 銘柄別
        by_sym: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "wins": 0, "net_pips": 0.0, "net_jpy": 0.0}
        )
        for t in trades:
            sym = t.get("symbol", "?")
            by_sym[sym]["count"] += 1
            if t.get("exit_type") == "TP":
                by_sym[sym]["wins"] += 1
            by_sym[sym]["net_pips"] += float(t.get("pnl_pips", 0))
            by_sym[sym]["net_jpy"]  += float(t.get("pnl_jpy",  0))
        for ss in by_sym.values():
            ss["wr"] = round(ss["wins"] / ss["count"], 3) if ss["count"] else 0.0

        # 時間帯別
        by_hour: dict[int, dict] = defaultdict(lambda: {"total": 0, "wins": 0})
        for t in trades:
            try:
                h = datetime.fromisoformat(t["entry_time"]).replace(
                    tzinfo=timezone.utc
                ).hour
                by_hour[h]["total"] += 1
                if t.get("exit_type") == "TP":
                    by_hour[h]["wins"] += 1
            except (KeyError, ValueError):
                pass

        return {
            "total":      total,
            "wins":       len(wins),
            "losses":     len(losses),
            "wr":         round(len(wins) / total, 3),
            "pf":         pf,
            "net_pips":   round(net_pips, 1),
            "net_jpy":    round(net_jpy, 0),
            "by_symbol":  dict(by_sym),
            "by_hour":    {h: v for h, v in sorted(by_hour.items())},
        }

    # ── 改善案自動生成 ────────────────────────────────────────────

    def _generate_insights(self, stats: dict) -> list[str]:
        """統計データから改善案・観察事項を自動生成する。"""
        insights: list[str] = []
        if stats["total"] == 0:
            insights.append("トレードなし")
            return insights

        # 全体勝率チェック（バックテスト基準: 65%以上）
        wr = stats["wr"]
        if wr < 0.55:
            insights.append(
                f"⚠️ 勝率{wr*100:.1f}%: 基準(65%)を大幅下回る → 相場環境変化を確認"
            )
        elif wr < 0.62:
            insights.append(
                f"注意: 勝率{wr*100:.1f}%: やや低調 → エントリー条件の再確認を推奨"
            )
        else:
            insights.append(f"✅ 勝率{wr*100:.1f}%: 良好")

        # PFチェック
        if 0 < stats["pf"] < 1.5:
            insights.append(f"⚠️ PF{stats['pf']}: 採用基準(2.0)を大幅下回る")

        # 銘柄別ドリフト検出
        for sym, ss in stats["by_symbol"].items():
            if ss["count"] < 3:
                continue
            bl = BASELINES.get(sym)
            if bl and ss["wr"] < bl["wr"] - 0.10:
                insights.append(
                    f"🔴 {sym}: 勝率{ss['wr']*100:.0f}%"
                    f"（基準{bl['wr']*100:.0f}%から-{(bl['wr']-ss['wr'])*100:.0f}pp）"
                    f" → ポジションサイズ削減を検討"
                )

        # 低勝率時間帯
        bad_hours = [
            h for h, v in stats["by_hour"].items()
            if v["total"] >= 3 and v["wins"] / v["total"] < 0.40
        ]
        if bad_hours:
            insights.append(
                f"⏰ 低勝率時間帯(UTC): {sorted(bad_hours)} → セッションフィルター検討"
            )

        return insights if insights else ["特筆すべき問題なし"]

    # ── スケジュール確認（メインループから毎分呼ぶ） ──────────────

    def check_schedule(self, now: datetime):
        """メインループから毎分呼び出す。時刻条件を確認して各レビューを実行。"""
        date_key = now.date()

        # 日次レビュー: 毎日21:00 UTC
        if now.hour == 21 and now.minute < 2:
            if self._last_daily != date_key:
                self._last_daily = date_key
                try:
                    self.run_daily_review(now)
                except Exception:
                    log.exception("日次レビュー失敗")

        # 週次レビュー: 日曜22:00 UTC
        if now.weekday() == 6 and now.hour == 22 and now.minute < 2:
            week_key = now.isocalendar()[:2]
            if self._last_weekly != week_key:
                self._last_weekly = week_key
                try:
                    self.run_weekly_review(now)
                except Exception:
                    log.exception("週次レビュー失敗")

        # 月次レビュー: 月末23:00 UTC
        last_day = calendar.monthrange(now.year, now.month)[1]
        month_key = (now.year, now.month)
        if now.day == last_day and now.hour == 23 and now.minute < 2:
            if self._last_monthly != month_key:
                self._last_monthly = month_key
                try:
                    self.run_monthly_review(now)
                except Exception:
                    log.exception("月次レビュー失敗")

    # ── 日次レビュー ──────────────────────────────────────────────

    def run_daily_review(self, now: datetime):
        """当日クローズトレードのサマリーを保存・Discord通知。"""
        log.info(f"[日次レビュー] {now.date()}")
        trades = self._load_today_trades(now)
        stats  = self._calc_stats(trades)
        date_str = now.strftime("%Y-%m-%d")

        out = DAILY_DIR / f"{date_str}.json"
        out.write_text(
            json.dumps({"date": date_str, "stats": stats},
                       ensure_ascii=False, indent=2)
        )

        if stats["total"] == 0:
            self.notify(
                f"📅 **日次レビュー {date_str}**\nトレードなし",
                color=0x95a5a6
            )
            return

        lines = [
            f"📅 **日次レビュー {date_str}**",
            f"トレード数: {stats['total']}件  （{stats['wins']}勝 {stats['losses']}敗）",
            f"勝率: {stats['wr']*100:.1f}%　PF: {stats['pf']}",
            f"損益: {stats['net_jpy']:+,.0f} JPY（{stats['net_pips']:+.1f}pips）",
        ]
        for sym, ss in stats["by_symbol"].items():
            lines.append(
                f"  {sym}: {ss['count']}件 WR{ss['wr']*100:.0f}%"
                f" {ss['net_jpy']:+,.0f}JPY"
            )
        self.notify("\n".join(lines), color=0x3498db)
        log.info(f"日次レビュー完了 → {out}")

    # ── 週次レビュー ──────────────────────────────────────────────

    def run_weekly_review(self, now: datetime):
        """週次フィードバック・改善案を生成して保存・Discord通知。"""
        year, week, _ = now.isocalendar()
        log.info(f"[週次レビュー] {year}-W{week:02d}")
        trades   = self._load_week_trades(year, week)
        stats    = self._calc_stats(trades)
        insights = self._generate_insights(stats)
        week_str = f"{year}-W{week:02d}"

        out = WEEKLY_DIR / f"{week_str}.json"
        out.write_text(
            json.dumps(
                {"week": week_str, "stats": stats, "insights": insights},
                ensure_ascii=False, indent=2
            )
        )

        lines = [
            f"📊 **週次フィードバック {week_str}**",
            f"トレード数: {stats['total']}件  勝率: {stats['wr']*100:.1f}%  PF: {stats['pf']}",
            f"週間損益: {stats['net_jpy']:+,.0f} JPY",
            "",
            "**今週の改善案:**",
        ] + [f"・{i}" for i in insights]
        self.notify("\n".join(lines), color=0xe67e22)
        log.info(f"週次レビュー完了 → {out}")

    # ── 月次レビュー ──────────────────────────────────────────────

    def run_monthly_review(self, now: datetime):
        """月次サマリーを生成し、Discord経由でユーザーに確認依頼する。"""
        year, month = now.year, now.month
        log.info(f"[月次レビュー] {year}-{month:02d}")
        trades   = self._load_month_trades(year, month)
        stats    = self._calc_stats(trades)
        insights = self._generate_insights(stats)

        # 月内の週次インサイトを集約（重複除去）
        weekly_accumulated: list[str] = []
        for wf in sorted(WEEKLY_DIR.glob(f"{year}-W*.json")):
            try:
                data = json.loads(wf.read_text())
                weekly_accumulated.extend(data.get("insights", []))
            except Exception:
                pass
        unique_insights = list(dict.fromkeys(weekly_accumulated))

        month_str = f"{year}-{month:02d}"
        out = MONTHLY_DIR / f"{month_str}.json"
        out.write_text(
            json.dumps(
                {
                    "month":              month_str,
                    "stats":              stats,
                    "monthly_insights":   insights,
                    "weekly_accumulated": unique_insights,
                },
                ensure_ascii=False, indent=2
            )
        )

        lines = [
            f"📋 **月次サマリー {month_str}**",
            "━━━━━━━━━━━━━━━━━━━━",
            f"トレード数: {stats['total']}件",
            f"勝率: {stats['wr']*100:.1f}%　PF: {stats['pf']}",
            f"月間損益: {stats['net_jpy']:+,.0f} JPY",
            "",
        ]
        all_insights = unique_insights or insights
        if all_insights:
            lines.append(f"**今月の改善案（{len(all_insights)}件）:**")
            for i, ins in enumerate(all_insights[:8], 1):
                lines.append(f"{i}. {ins}")
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            "⬆️ 上記の改善案を確認してください。",
            f"実装する場合は Claude に「{month_str}の改善案を実装して」と伝えてください。",
            f"詳細: `production/trade_logs/reviews/monthly/{month_str}.json`",
        ]
        self.notify("\n".join(lines), color=0x9b59b6)
        log.info(f"月次レビュー完了 → {out}")
