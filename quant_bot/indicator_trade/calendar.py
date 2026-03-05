"""
経済カレンダー取得モジュール (Investing.com スクレイピング)。

Investing.com の経済カレンダーページから高重要度の指標を取得し、
JSON キャッシュとして保存する。

注意事項:
  - robots.txt を遵守: /economic-calendar/ は許可されている
  - レート制限: リクエスト間に 5 秒待機
  - User-Agent スプーフィング（ブラウザ偽装）を使用
  - キャッシュ: data/calendar_cache/YYYY-MM-DD.json
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("quant_bot.indicator_trade")

# ----------------------------------------
# 定数
# ----------------------------------------

INVESTING_URL = "https://www.investing.com/economic-calendar/"
REQUEST_DELAY_SEC = 5.0  # robots.txt 遵守のためのウェイト
CACHE_DIR = Path("data/calendar_cache")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/",
}

# 高重要度フィルター（Bull's Eye アイコン 3 つ）
HIGH_IMPORTANCE_KEYWORDS = {
    "NFP", "Non-Farm", "FOMC", "Fed", "CPI", "PCE",
    "GDP", "Retail Sales", "ISM", "PMI", "Payroll",
    "Unemployment", "PPI", "Housing Starts", "JOLTS",
}


@dataclass
class EconomicEvent:
    """経済指標イベント。"""

    datetime_utc: str
    """イベント日時 (ISO 8601 UTC)"""

    currency: str
    """影響通貨 (例: 'USD', 'EUR')"""

    event_name: str
    """指標名称"""

    importance: str
    """重要度: 'high' / 'medium' / 'low'"""

    actual: Optional[str] = None
    """実際の結果 (発表後のみ)"""

    forecast: Optional[str] = None
    """予想値"""

    previous: Optional[str] = None
    """前回値"""

    surprise_direction: Optional[str] = None
    """予想比: 'beat' / 'miss' / 'in_line' / None"""


class EconomicCalendar:
    """Investing.com 経済カレンダースクレイパー。"""

    def __init__(self, cache_dir: str | Path = CACHE_DIR):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_events(
        self,
        target_date: Optional[date] = None,
        use_cache: bool = True,
        currency_filter: Optional[list[str]] = None,
        min_importance: str = "high",
    ) -> list[EconomicEvent]:
        """
        指定日の経済指標イベントを取得。

        Args:
            target_date:     対象日 (None の場合は今日)
            use_cache:       キャッシュを使用するか
            currency_filter: 取得する通貨リスト (例: ['USD', 'EUR'])
                             None の場合は全通貨
            min_importance:  最低重要度 'high' / 'medium' / 'low'

        Returns:
            EconomicEvent のリスト
        """
        if target_date is None:
            target_date = date.today()

        # キャッシュ確認
        if use_cache:
            cached = self._load_cache(target_date)
            if cached is not None:
                log.info(f"カレンダーキャッシュ使用: {target_date}")
                events = [EconomicEvent(**e) for e in cached]
                return self._filter_events(events, currency_filter, min_importance)

        # スクレイピング実行
        events = self._scrape(target_date)

        # キャッシュ保存
        if events:
            self._save_cache(target_date, events)

        return self._filter_events(events, currency_filter, min_importance)

    def fetch_week_events(
        self,
        start_date: Optional[date] = None,
        currency_filter: Optional[list[str]] = None,
        min_importance: str = "high",
    ) -> list[EconomicEvent]:
        """
        1週間分のイベントを取得（月〜金）。

        Args:
            start_date: 週の開始日 (None の場合は今週の月曜)
        """
        if start_date is None:
            today = date.today()
            start_date = today - timedelta(days=today.weekday())

        all_events: list[EconomicEvent] = []
        for i in range(5):  # 月〜金
            d = start_date + timedelta(days=i)
            try:
                events = self.fetch_events(d, currency_filter=currency_filter,
                                          min_importance=min_importance)
                all_events.extend(events)
                if i < 4:
                    time.sleep(REQUEST_DELAY_SEC)
            except Exception as e:
                log.warning(f"カレンダー取得失敗 ({d}): {e}")

        return all_events

    # ------------------------------------------------------------------ #
    #  スクレイピング                                                      #
    # ------------------------------------------------------------------ #

    def _scrape(self, target_date: date) -> list[EconomicEvent]:
        """Investing.com からスクレイピング。"""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError(
                f"スクレイピングには {e.name} が必要です。"
                "pip install requests beautifulsoup4 を実行してください。"
            ) from e

        log.info(f"Investing.com スクレイピング開始: {target_date}")

        date_str = target_date.strftime("%Y-%m-%d")
        params = {
            "dateFrom": date_str,
            "dateTo": date_str,
        }

        try:
            resp = requests.get(
                INVESTING_URL,
                headers=HEADERS,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as e:
            log.error(f"HTTP リクエスト失敗: {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        events = self._parse_calendar_table(soup, target_date)

        log.info(f"スクレイピング完了: {len(events)} イベント取得")
        return events

    def _parse_calendar_table(
        self, soup, target_date: date
    ) -> list[EconomicEvent]:
        """カレンダーテーブルをパース。"""
        events: list[EconomicEvent] = []

        # Investing.com のカレンダーテーブル
        table = soup.find("table", {"id": "economicCalendarData"})
        if table is None:
            # 別のセレクターを試行
            table = soup.find("table", class_="genTbl")

        if table is None:
            log.warning("カレンダーテーブルが見つかりませんでした。")
            return events

        rows = table.find_all("tr", class_=lambda c: c and "js-event-item" in c)
        for row in rows:
            event = self._parse_row(row, target_date)
            if event is not None:
                events.append(event)

        return events

    def _parse_row(self, row, target_date: date) -> Optional[EconomicEvent]:
        """テーブル行から EconomicEvent を生成。"""
        try:
            cells = row.find_all("td")
            if len(cells) < 5:
                return None

            # 時刻
            time_cell = cells[0].get_text(strip=True)
            try:
                event_dt = datetime.strptime(
                    f"{target_date} {time_cell}", "%Y-%m-%d %H:%M"
                )
            except ValueError:
                event_dt = datetime(target_date.year, target_date.month, target_date.day)

            # 通貨
            currency_cell = cells[1]
            currency_img = currency_cell.find("span", class_="ceFlags")
            currency = currency_img.get("title", "") if currency_img else cells[1].get_text(strip=True)

            # 重要度（星アイコン数で判定）
            importance_cell = cells[2]
            bull_icons = importance_cell.find_all("i", class_="grayFullBullishIcon")
            imp_count = len(bull_icons)
            if imp_count >= 3:
                importance = "high"
            elif imp_count == 2:
                importance = "medium"
            else:
                importance = "low"

            # イベント名
            event_name = cells[3].get_text(strip=True)

            # 実績・予想・前回
            actual = cells[4].get_text(strip=True) if len(cells) > 4 else None
            forecast = cells[5].get_text(strip=True) if len(cells) > 5 else None
            previous = cells[6].get_text(strip=True) if len(cells) > 6 else None

            # サプライズ方向
            surprise_direction = self._calc_surprise(actual, forecast)

            return EconomicEvent(
                datetime_utc=event_dt.isoformat(),
                currency=currency or "USD",
                event_name=event_name,
                importance=importance,
                actual=actual or None,
                forecast=forecast or None,
                previous=previous or None,
                surprise_direction=surprise_direction,
            )
        except Exception as e:
            log.debug(f"行パースエラー: {e}")
            return None

    @staticmethod
    def _calc_surprise(actual: Optional[str], forecast: Optional[str]) -> Optional[str]:
        """実績と予想を比較してサプライズ方向を判定。"""
        if not actual or not forecast:
            return None
        try:
            # "K", "M", "B" 等の単位を除去
            def parse_val(s: str) -> float:
                s = s.strip().replace(",", "").replace("%", "")
                mult = 1.0
                if s.endswith("K"):
                    s, mult = s[:-1], 1_000
                elif s.endswith("M"):
                    s, mult = s[:-1], 1_000_000
                elif s.endswith("B"):
                    s, mult = s[:-1], 1_000_000_000
                return float(s) * mult

            a_val = parse_val(actual)
            f_val = parse_val(forecast)
            diff_pct = abs(a_val - f_val) / max(abs(f_val), 0.001)

            if diff_pct < 0.01:
                return "in_line"
            elif a_val > f_val:
                return "beat"
            else:
                return "miss"
        except (ValueError, ZeroDivisionError):
            return None

    # ------------------------------------------------------------------ #
    #  キャッシュ                                                          #
    # ------------------------------------------------------------------ #

    def _cache_path(self, d: date) -> Path:
        return self._cache_dir / f"{d.isoformat()}.json"

    def _load_cache(self, d: date) -> Optional[list[dict]]:
        path = self._cache_path(d)
        if path.exists():
            try:
                with path.open(encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, d: date, events: list[EconomicEvent]) -> None:
        path = self._cache_path(d)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump([asdict(e) for e in events], f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"キャッシュ保存失敗: {e}")

    # ------------------------------------------------------------------ #
    #  フィルタリング                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _filter_events(
        events: list[EconomicEvent],
        currency_filter: Optional[list[str]],
        min_importance: str,
    ) -> list[EconomicEvent]:
        importance_rank = {"low": 0, "medium": 1, "high": 2}
        min_rank = importance_rank.get(min_importance, 2)

        filtered = []
        for ev in events:
            if importance_rank.get(ev.importance, 0) < min_rank:
                continue
            if currency_filter and ev.currency not in currency_filter:
                continue
            filtered.append(ev)

        return filtered
