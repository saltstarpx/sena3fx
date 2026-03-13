"""
エントリー禁止期間フィルター
=============================
以下の期間は新規エントリーを全て停止する:

  1. 毎年12月15日 〜 翌年1月5日 (年末年始)
     理由: 流動性低下・スプレッド拡大・異常値リスク

  2. 米国の主要祝日
     理由: CME Gold/Silver は米国市場に連動

  3. 日本の主要祝日
     理由: 円建てトレーダーの流動性低下

  4. CMEクローズ時間帯 = 土曜日は全時間帯禁止
     理由: CME Gold/Silver 市場は土曜に休場

使用例:
  >>> from live.time_filter import is_trading_allowed
  >>> if not is_trading_allowed(datetime.utcnow()):
  ...     print("エントリー禁止期間")
"""

import calendar
from datetime import date, datetime, timedelta
from typing import Union


# ------------------------------------------------------------------ #
#  祝日計算ヘルパー                                                   #
# ------------------------------------------------------------------ #

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """
    指定月の第N weekday を返す。
    weekday: 0=月曜, 1=火曜, ..., 6=日曜
    """
    first_day = date(year, month, 1)
    diff = (weekday - first_day.weekday()) % 7
    first_occurrence = first_day.replace(day=1 + diff)
    return first_occurrence.replace(day=first_occurrence.day + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """指定月の最後の weekday を返す"""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    diff = (d.weekday() - weekday) % 7
    return d.replace(day=last_day - diff)


def _good_friday(year: int) -> date:
    """
    グッドフライデー（復活祭の2日前）を計算する。
    CME Gold/Silver は休場。
    使用アルゴリズム: Oudin's method
    """
    G = year % 19
    C = year // 100
    H = (C - C // 4 - (8 * C + 13) // 25 + 19 * G + 15) % 30
    I = H - (H // 28) * (1 - (H // 28) * (29 // (H + 1)) * ((21 - G) // 11))
    J = (year + year // 4 + I + 2 - C + C // 4) % 7
    L = I - J
    month = 3 + (L + 40) // 44
    day   = L + 28 - 31 * (month // 4)
    easter = date(year, month, day)
    return easter - timedelta(days=2)


# ------------------------------------------------------------------ #
#  米国祝日                                                           #
# ------------------------------------------------------------------ #

def us_holidays(year: int) -> set:
    """
    CME Gold/Silver の市場休場日 (米国主要祝日) を返す。
    CME は固定の休場スケジュールを公表しているが、
    概ね以下の米国連邦祝日に準ずる。
    """
    h = set()

    # 元日 (New Year's Day)
    h.add(date(year, 1, 1))

    # MLK デー (第3月曜 / 1月)
    h.add(_nth_weekday(year, 1, 0, 3))

    # プレジデントデー (第3月曜 / 2月)
    h.add(_nth_weekday(year, 2, 0, 3))

    # グッドフライデー (復活祭-2日 / 3-4月)
    h.add(_good_friday(year))

    # メモリアルデー (5月最終月曜)
    h.add(_last_weekday(year, 5, 0))

    # ジューンティーンス (6/19)
    h.add(date(year, 6, 19))

    # 独立記念日 (7/4)
    h.add(date(year, 7, 4))

    # レイバーデー (9月第1月曜)
    h.add(_nth_weekday(year, 9, 0, 1))

    # コロンブスデー (10月第2月曜) ※CMEは通常営業の場合もあるが保守的に休とする
    h.add(_nth_weekday(year, 10, 0, 2))

    # 退役軍人の日 (11/11)
    h.add(date(year, 11, 11))

    # サンクスギビング (11月第4木曜)
    h.add(_nth_weekday(year, 11, 3, 4))

    # クリスマス (12/25)
    h.add(date(year, 12, 25))

    return h


# ------------------------------------------------------------------ #
#  日本祝日                                                           #
# ------------------------------------------------------------------ #

def japan_holidays(year: int) -> set:
    """
    日本の主要祝日を返す (概算日付)。
    振替休日は考慮していないが、年末年始フィルターで補完される。
    """
    h = set()

    h.add(date(year, 1, 1))   # 元日
    h.add(date(year, 1, 2))   # 元日振替休暇 (一般的に休場)
    h.add(date(year, 1, 3))   # 松の内
    h.add(date(year, 2, 11))  # 建国記念の日
    h.add(date(year, 2, 23))  # 天皇誕生日
    h.add(date(year, 3, 20))  # 春分の日 (概算 ±1日)
    h.add(date(year, 4, 29))  # 昭和の日
    h.add(date(year, 5, 3))   # 憲法記念日
    h.add(date(year, 5, 4))   # みどりの日
    h.add(date(year, 5, 5))   # こどもの日
    h.add(date(year, 7, 20))  # 海の日 (第3月曜 概算)
    h.add(date(year, 8, 11))  # 山の日
    h.add(date(year, 9, 15))  # 敬老の日 (第3月曜 概算)
    h.add(date(year, 9, 23))  # 秋分の日 (概算 ±1日)
    h.add(date(year, 10, 14)) # スポーツの日 (第2月曜 概算)
    h.add(date(year, 11, 3))  # 文化の日
    h.add(date(year, 11, 23)) # 勤労感謝の日
    h.add(date(year, 12, 31)) # 大晦日 (実質的に市場薄い)

    return h


# ------------------------------------------------------------------ #
#  個別フィルター関数                                                  #
# ------------------------------------------------------------------ #

def is_yearend_period(dt: Union[datetime, date]) -> bool:
    """
    年末年始期間の判定 (12/15 〜 翌年1/5)。

    この期間は:
    - 欧米機関投資家の決算・リバランスで相場が歪む
    - 流動性が低下してスプレッドが拡大する
    - 予測困難な価格変動が起きやすい
    """
    d = dt.date() if isinstance(dt, datetime) else dt
    if d.month == 12 and d.day >= 15:
        return True
    if d.month == 1 and d.day <= 5:
        return True
    return False


def is_cme_closed(dt: Union[datetime, date]) -> bool:
    """
    CME Gold/Silver クローズ判定。
    土曜日は全時間帯で新規エントリー禁止。
    """
    d = dt.date() if isinstance(dt, datetime) else dt
    return d.weekday() == 5  # Saturday


def is_holiday(dt: Union[datetime, date]) -> bool:
    """米国または日本の主要祝日の判定"""
    d = dt.date() if isinstance(dt, datetime) else dt
    year = d.year

    if d in us_holidays(year):
        return True
    if d in japan_holidays(year):
        return True
    return False


# ------------------------------------------------------------------ #
#  メインフィルター (総合判定)                                         #
# ------------------------------------------------------------------ #

def is_trading_allowed(dt: Union[datetime, date]) -> bool:
    """
    エントリー可否の総合判定。

    全条件をチェックし、1つでも禁止条件に引っかかれば False を返す。

    Returns:
        True  = エントリー可
        False = エントリー禁止
    """
    if is_yearend_period(dt):
        return False
    if is_cme_closed(dt):
        return False
    if is_holiday(dt):
        return False
    return True


def get_block_reason(dt: Union[datetime, date]) -> str:
    """
    エントリーが禁止される理由を返す (ログ用)。

    Returns:
        '' = 禁止なし
        その他 = 禁止理由
    """
    if is_yearend_period(dt):
        return '年末年始 (12/15〜1/5)'
    if is_cme_closed(dt):
        return 'CMEクローズ (土曜日)'
    if is_holiday(dt):
        return '祝日 (米国/日本)'
    return ''


# ------------------------------------------------------------------ #
#  バックテスト用: Series に一括フィルター適用                        #
# ------------------------------------------------------------------ #

def filter_signals_by_time(signals: 'pd.Series') -> 'pd.Series':
    """
    バックテスト用: シグナル Series に時間フィルターを一括適用する。

    Args:
        signals: 'long' | 'short' | None の pd.Series (DatetimeIndex 必須)

    Returns:
        フィルター適用後の pd.Series (禁止期間のシグナルは None に置換)
    """
    import pandas as pd

    filtered = signals.copy()
    blocked = pd.Series(
        [not is_trading_allowed(dt) for dt in signals.index],
        index=signals.index,
        dtype=bool,
    )
    filtered[blocked] = None
    return filtered
