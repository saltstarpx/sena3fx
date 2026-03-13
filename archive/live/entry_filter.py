"""
エントリーフィルター
====================
薄いゾーン判定・XAUT過熱度フラグ・追いかけエントリー制御を
ライブトレードシステムに統合するモジュール。

bot_v2.py での利用例:
    from live.entry_filter import EntryFilter
    ef = EntryFilter()
    ef.reload_thin_zones()   # 週1回程度で再ロード

    # エントリー前チェック
    ctx = ef.check(current_price, side='long')
    if ctx.allow:
        units = int(base_units * ctx.lot_scale)
        sl    = entry + sl_dist * ctx.stop_scale * (-1 if side=='long' else 1)
"""
import os
import logging
from typing import Optional
from dataclasses import dataclass, field

from price_zone_analyzer import load_thin_zones, is_thin_zone, get_thin_zone_params

log = logging.getLogger('sena3fx.entry_filter')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ #
#  エントリー判定結果                                                 #
# ------------------------------------------------------------------ #
@dataclass
class EntryContext:
    """
    1回のエントリー判定の結果を保持するデータクラス。

    Attributes:
        allow:             True = エントリー許可
        lot_scale:         ロット倍率 (1.0=通常 / 0.7=薄いゾーン縮小)
        stop_scale:        SL距離倍率 (1.0=通常 / 1.3=薄いゾーン拡張)
        allow_chase_entry: True = ブレイク後の後乗りエントリー許可
        block_reason:      エントリー禁止の理由 (許可時は空文字)
        is_thin_zone:      True = 薄いゾーン（ボラ拡大警戒）
        is_overheated:     True = XAUT過熱中（ロング停止）
    """
    allow:             bool  = True
    lot_scale:         float = 1.0
    stop_scale:        float = 1.0
    allow_chase_entry: bool  = False
    block_reason:      str   = ''
    is_thin_zone:      bool  = False
    is_overheated:     bool  = False


# ================================================================== #
#  EntryFilter クラス                                                 #
# ================================================================== #
class EntryFilter:
    """
    エントリーフィルター統合クラス。

    フラグ:
        pause_long_entries:  True にするとロングエントリーを一時停止
                             (OverheatMonitor から外部更新)
        allow_chase_entry:   True にすると後乗りエントリーを全体で許可

    週次タスク:
        ef.reload_thin_zones()  — thin_zones.json を再読み込み
    """

    def __init__(self,
                 thin_zones_path: Optional[str] = None,
                 lot_scale_thin: float = 0.7,
                 stop_scale_thin: float = 1.3,
                 chase_in_thin: bool = True):
        """
        Args:
            thin_zones_path:  thin_zones.json のパス (省略時は自動検出)
            lot_scale_thin:   薄いゾーンでのロット倍率
            stop_scale_thin:  薄いゾーンでのSL拡張倍率
            chase_in_thin:    薄いゾーンで後乗りを許可するか
        """
        self.thin_zones_path = thin_zones_path or os.path.join(
            ROOT, 'data', 'thin_zones.json'
        )
        self.lot_scale_thin  = lot_scale_thin
        self.stop_scale_thin = stop_scale_thin
        self.chase_in_thin   = chase_in_thin

        # ---- 外部から更新されるフラグ ----
        self.pause_long_entries: bool = False   # OverheatMonitor → True で更新
        self.allow_chase_entry:  bool = False   # グローバル後乗り許可フラグ

        # ---- 内部状態 ----
        self._thin_zones: list = []
        self.reload_thin_zones()

    # ------------------------------------------------------------------ #
    #  薄いゾーンデータ管理                                              #
    # ------------------------------------------------------------------ #
    def reload_thin_zones(self) -> int:
        """
        thin_zones.json を再読み込みする。
        週1回程度のタイミングで呼び出すこと。

        Returns:
            int: ゾーン数 (ファイルがない場合は 0)
        """
        self._thin_zones = load_thin_zones(self.thin_zones_path)
        n = len(self._thin_zones)
        if n:
            log.info(f"薄いゾーン再読み込み: {n} ゾーン")
        else:
            log.warning(
                f"thin_zones.json が見つかりません: {self.thin_zones_path}\n"
                "  → python price_zone_analyzer.py を実行して生成してください"
            )
        return n

    @property
    def thin_zones(self) -> list:
        return self._thin_zones

    # ------------------------------------------------------------------ #
    #  エントリー判定                                                     #
    # ------------------------------------------------------------------ #
    def check(self, current_price: float, side: str = 'long') -> EntryContext:
        """
        エントリー可否を判定し、パラメータ調整値を返す。

        Args:
            current_price: 現在の市場価格
            side:          'long' または 'short'

        Returns:
            EntryContext: エントリー判定結果
        """
        ctx = EntryContext()

        # --- XAUT過熱フラグチェック (ロングのみ適用) ---
        if side == 'long' and self.pause_long_entries:
            ctx.allow        = False
            ctx.is_overheated = True
            ctx.block_reason = 'XAUT過熱中: ロングエントリー停止'
            log.info(f"エントリーブロック: {ctx.block_reason}")
            return ctx

        # --- 薄いゾーンチェック ---
        thin_params = get_thin_zone_params(
            current_price,
            self._thin_zones,
            lot_scale  = self.lot_scale_thin,
            stop_scale = self.stop_scale_thin,
            allow_chase = self.chase_in_thin,
        )
        ctx.is_thin_zone      = thin_params['is_thin']
        ctx.lot_scale         = thin_params['lot_scale']
        ctx.stop_scale        = thin_params['stop_scale']
        ctx.allow_chase_entry = (thin_params['allow_chase_entry']
                                 or self.allow_chase_entry)

        if ctx.is_thin_zone:
            log.info(
                f"薄いゾーン検出: ${current_price:.2f} "
                f"→ lot_scale={ctx.lot_scale:.1f} "
                f"stop_scale={ctx.stop_scale:.1f} "
                f"chase={ctx.allow_chase_entry}"
            )

        return ctx

    def apply_to_position(self, units: int, sl_distance: float,
                          current_price: float, side: str = 'long') -> dict:
        """
        エントリー判定結果をポジションパラメータに適用する。

        Args:
            units:         基本ポジションサイズ (未調整)
            sl_distance:   基本SL距離 (価格差)
            current_price: 現在価格
            side:          'long' / 'short'

        Returns:
            dict: {
                'allow': bool,
                'units': int,           # 調整済みユニット数
                'sl_distance': float,   # 調整済みSL距離
                'allow_chase': bool,
                'block_reason': str,
            }
        """
        ctx = self.check(current_price, side)
        if not ctx.allow:
            return {
                'allow':       False,
                'units':       0,
                'sl_distance': sl_distance,
                'allow_chase': False,
                'block_reason': ctx.block_reason,
            }
        return {
            'allow':       True,
            'units':       max(1, int(units * ctx.lot_scale)),
            'sl_distance': sl_distance * ctx.stop_scale,
            'allow_chase': ctx.allow_chase_entry,
            'block_reason': '',
        }

    # ------------------------------------------------------------------ #
    #  OverheatMonitor との連携                                          #
    # ------------------------------------------------------------------ #
    def update_from_overheat_monitor(self, monitor) -> None:
        """
        OverheatMonitor の状態をフィルターに反映する。

        bot_v2.py でのループ内で呼び出す:
            ef.update_from_overheat_monitor(overheat_monitor)

        Args:
            monitor: OverheatMonitor インスタンス
        """
        was_paused = self.pause_long_entries
        self.pause_long_entries = monitor.is_overheated()

        if self.pause_long_entries and not was_paused:
            log.warning(
                f"⚠ XAUT過熱: ロングエントリー停止 "
                f"(乖離={monitor.divergence_pct:+.4f}%)"
            )
        elif not self.pause_long_entries and was_paused:
            log.info(
                f"✅ XAUT正常化: ロングエントリー再開 "
                f"(乖離={monitor.divergence_pct:+.4f}%)"
            )

    def get_status(self) -> dict:
        """現在のフィルター状態を辞書で返す"""
        return {
            'pause_long_entries': self.pause_long_entries,
            'allow_chase_entry':  self.allow_chase_entry,
            'thin_zone_count':    len(self._thin_zones),
            'thin_zones_path':    self.thin_zones_path,
        }
