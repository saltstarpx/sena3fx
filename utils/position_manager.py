"""
utils/position_manager.py
=========================
ポジション管理システム（yagami_position8_risk10_v2）

【バージョン履歴】
  yagami_position8_risk10    : 最大8ポジ / リスク10% / 相関グループフィルター
  yagami_position8_risk10_v2 : 上記に加え、含み益ポジション枠外ルール・
                                損失転換時追加ポジション全カット・4時間クールタイムを追加

【設計仕様】
  全体ポジション上限    : 8ポジ
  全体トータルリスク上限 : 10%
  グループ内上限        : 2ポジ（FX / 貴金属 / 指数）
  サブグループ内上限    : 1ポジ（相関の高い銘柄群）
  相関グループ内上限    : 1ポジ（最優先フィルター）

【v2 追加ルール】
  ① 含み益ポジションは枠カウント外
      → update_pnl() で現在損益を更新し、含み益（pnl > 0）のポジションは
        全体ポジション数・グループ数・サブグループ数のカウントから除外する
      → 損失を防ぐためのルールであり、利益中の追加エントリーを妨げない

  ② 損失転換時に追加ポジションを全カット
      → update_pnl() 呼び出し時、含み益だったポジションが含み損に転じた場合、
        そのポジションが「損失転換トリガー」となる
      → トリガー発生時、check_and_cut() が「追加ポジション」（含み益フラグで
        エントリーされたポジション）を全て強制クローズ対象として返す
      → バックテスト本体側でこのリストを受け取り、即座に成行決済する

  ③ 4時間クールタイム
      → 追加ポジション全カット後、4時間は新規エントリーを禁止する
      → can_enter() でクールタイム中は False を返す

【グループ × サブグループ定義】
  FX
    sub_usd_buy  : USDJPY, USDCAD, USDCHF
    sub_usd_sell : EURUSD, GBPUSD, AUDUSD, NZDUSD
    sub_cross_jpy: EURJPY, GBPJPY
    sub_cross    : EURGBP
  貴金属 (metals)
    sub_metals   : XAUUSD, XAGUSD
  指数 (index)
    sub_us_index : US30, SPX500, NAS100

【相関グループ定義（最優先フィルター）】
  equity_risk : US30, SPX500, NAS100
  risk_on_fx  : EURJPY, GBPJPY, AUDJPY
  eur_pairs   : EURUSD, EURJPY, EURGBP
  gbp_pairs   : GBPUSD, GBPJPY, EURGBP

  → 同グループ内では1ポジション限定
  → US30保有中はSPX500/NAS100の新規エントリー禁止
  → EURJPYとEURGBPは同時保有禁止（eur_pairs）

【使い方】
    from utils.position_manager import PositionManager

    pm = PositionManager()

    # エントリー可否チェック（現在時刻を渡す）
    ok, reason = pm.can_enter("EURUSD", risk_pct=0.02, now=current_time)
    if ok:
        is_extra = pm.is_extra_entry()  # 含み益ポジション枠外で入る追加エントリーか
        pm.open_position("EURUSD", risk_pct=0.02, entry_price=ep,
                         entry_time=current_time, is_extra=is_extra)

    # 損益更新（毎バー呼び出し）
    pm.update_pnl("EURUSD", current_price=current_price)

    # 損失転換チェック → 追加ポジション全カット対象を取得
    to_cut = pm.check_and_cut(now=current_time)
    for sym in to_cut:
        # バックテスト本体で成行決済処理
        pm.close_position(sym)

    # 通常決済
    pm.close_position("EURUSD")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd

# ── グループ × サブグループ定義 ─────────────────────────────
SYMBOL_GROUP: dict[str, str] = {}
SYMBOL_SUBGROUP: dict[str, str] = {}

_GROUPS: dict[str, dict[str, list[str]]] = {
    "fx": {
        "usd_buy":   ["USDJPY", "USDCAD", "USDCHF"],
        "usd_sell":  ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
        "cross_jpy": ["EURJPY", "GBPJPY"],
        "cross":     ["EURGBP"],
    },
    "metals": {
        "metals":    ["XAUUSD", "XAGUSD"],
    },
    "index": {
        "us_index":  ["US30", "SPX500", "NAS100"],
    },
}

for _grp, _subs in _GROUPS.items():
    for _sub, _syms in _subs.items():
        for _sym in _syms:
            SYMBOL_GROUP[_sym]    = _grp
            SYMBOL_SUBGROUP[_sym] = _sub

# ── 相関グループ定義（最優先フィルター） ──────────────────────
CORRELATION_GROUPS: dict[str, list[str]] = {
    "equity_risk": ["US30", "SPX500", "NAS100"],
    "risk_on_fx":  ["EURJPY", "GBPJPY", "AUDJPY"],
    "eur_pairs":   ["EURUSD", "EURJPY", "EURGBP"],
    "gbp_pairs":   ["GBPUSD", "GBPJPY", "EURGBP"],
}

SYMBOL_CORR_GROUPS: dict[str, list[str]] = {}
for _cgrp, _syms in CORRELATION_GROUPS.items():
    for _sym in _syms:
        if _sym not in SYMBOL_CORR_GROUPS:
            SYMBOL_CORR_GROUPS[_sym] = []
        SYMBOL_CORR_GROUPS[_sym].append(_cgrp)

# ── 制約定数 ────────────────────────────────────────────────
MAX_TOTAL_POSITIONS    = 8      # 全体ポジション上限
MAX_TOTAL_RISK_PCT     = 0.10   # 全体トータルリスク上限（10%）
MAX_GROUP_POSITIONS    = 2      # グループ内上限
MAX_SUBGROUP_POSITIONS = 1      # サブグループ内上限
MAX_CORR_POSITIONS     = 1      # 相関グループ内上限（最優先）
COOLTIME_HOURS         = 4      # 追加ポジション全カット後のクールタイム（時間）


@dataclass
class Position:
    """保有中のポジション情報"""
    symbol:      str
    risk_pct:    float
    entry_time:  datetime
    entry_price: float
    group:       str
    subgroup:    str
    is_extra:    bool  = False   # 含み益枠外で追加エントリーされたポジション
    pnl:         float = 0.0     # 現在の含み損益（円）
    in_profit:   bool  = False   # 含み益フラグ（True = 枠カウント外）
    meta:        dict  = field(default_factory=dict)


class PositionManager:
    """
    全銘柄のポジションを一元管理するクラス。
    yagami_position8_risk10_v2 ルールを実装。
    """

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}
        self._cooltime_until: Optional[datetime] = None  # クールタイム終了時刻

    # ── 含み益フラグを考慮したカウント ────────────────────────

    @property
    def total_positions(self) -> int:
        """含み損ポジションのみカウント（含み益は枠外）"""
        return sum(1 for p in self._positions.values() if not p.in_profit)

    @property
    def total_risk_pct(self) -> float:
        """含み損ポジションのリスク合計のみ"""
        return sum(p.risk_pct for p in self._positions.values() if not p.in_profit)

    @property
    def total_positions_all(self) -> int:
        """含み益含む全ポジション数（参照用）"""
        return len(self._positions)

    def group_positions(self, group: str) -> int:
        """グループ内の含み損ポジション数"""
        return sum(1 for p in self._positions.values()
                   if p.group == group and not p.in_profit)

    def subgroup_positions(self, subgroup: str) -> int:
        """サブグループ内の含み損ポジション数"""
        return sum(1 for p in self._positions.values()
                   if p.subgroup == subgroup and not p.in_profit)

    def corr_group_positions(self, corr_group: str) -> int:
        """相関グループ内の保有ポジション数（含み益・含み損両方カウント）"""
        members = CORRELATION_GROUPS.get(corr_group, [])
        return sum(1 for s in self._positions if s in members)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def is_in_cooltime(self, now: Optional[datetime] = None) -> bool:
        """クールタイム中かどうか"""
        if self._cooltime_until is None:
            return False
        t = now or datetime.now(timezone.utc)
        if not t.tzinfo:
            t = t.replace(tzinfo=timezone.utc)
        return t < self._cooltime_until

    def is_extra_entry(self) -> bool:
        """
        次のエントリーが「追加エントリー」かどうかを返す。
        含み益ポジションが1つ以上あり、通常枠（含み損カウント）が
        上限未満の場合に True を返す。
        バックテスト本体側で open_position の is_extra 引数に渡す。
        """
        has_profit_pos = any(p.in_profit for p in self._positions.values())
        return has_profit_pos

    # ── エントリー可否チェック ───────────────────────────────

    def can_enter(
        self,
        symbol: str,
        risk_pct: float,
        now: Optional[datetime] = None,
    ) -> tuple[bool, str]:
        """
        エントリー可否を判定する。

        チェック順序:
          0. クールタイム中チェック（最優先）
          1. 同銘柄保有中チェック
          2. 相関グループ内上限（最優先フィルター）
          3. 全体ポジション上限（含み益は除外してカウント）
          4. 全体トータルリスク上限（含み益は除外）
          5. グループ内上限（含み益は除外）
          6. サブグループ内上限（含み益は除外）
        """
        sym = symbol.upper()

        # 0. クールタイムチェック
        if self.is_in_cooltime(now):
            remaining = self._cooltime_until - (now or datetime.now(timezone.utc))
            h = int(remaining.total_seconds() // 3600)
            m = int((remaining.total_seconds() % 3600) // 60)
            return False, f"{sym}: クールタイム中（残り {h}時間{m}分）"

        # 1. 既に同銘柄を保有中
        if self.has_position(sym):
            return False, f"{sym}: 既に保有中"

        # 2. 相関グループ内上限チェック（最優先）
        corr_groups = SYMBOL_CORR_GROUPS.get(sym, [])
        for cgrp in corr_groups:
            if self.corr_group_positions(cgrp) >= MAX_CORR_POSITIONS:
                members = CORRELATION_GROUPS[cgrp]
                conflicting = [s for s in self._positions if s in members]
                return False, (
                    f"{sym}: 相関グループ [{cgrp}] の上限 {MAX_CORR_POSITIONS}ポジ に達しています "
                    f"（競合: {', '.join(conflicting)}）"
                )

        # 銘柄がグループ定義に存在するか確認
        if sym not in SYMBOL_GROUP:
            return False, f"{sym}: グループ定義なし（SYMBOL_GROUPに追加してください）"

        grp = SYMBOL_GROUP[sym]
        sub = SYMBOL_SUBGROUP[sym]

        # 3. 全体ポジション上限（含み益除外カウント）
        if self.total_positions >= MAX_TOTAL_POSITIONS:
            return False, (
                f"{sym}: 全体ポジション上限 {MAX_TOTAL_POSITIONS}ポジ に達しています "
                f"（含み損カウント {self.total_positions}ポジ / 全{self.total_positions_all}ポジ）"
            )

        # 4. 全体トータルリスク上限（含み益除外）
        if self.total_risk_pct + risk_pct > MAX_TOTAL_RISK_PCT:
            return False, (
                f"{sym}: 全体リスク上限 {MAX_TOTAL_RISK_PCT*100:.0f}% を超えます "
                f"（現在 {self.total_risk_pct*100:.1f}% + {risk_pct*100:.1f}%）"
            )

        # 5. グループ内上限（含み益除外）
        if self.group_positions(grp) >= MAX_GROUP_POSITIONS:
            return False, (
                f"{sym}: グループ [{grp}] の上限 {MAX_GROUP_POSITIONS}ポジ に達しています "
                f"（含み損カウント {self.group_positions(grp)}ポジ）"
            )

        # 6. サブグループ内上限（含み益除外）
        if self.subgroup_positions(sub) >= MAX_SUBGROUP_POSITIONS:
            return False, (
                f"{sym}: サブグループ [{sub}] の上限 {MAX_SUBGROUP_POSITIONS}ポジ に達しています "
                f"（含み損カウント {self.subgroup_positions(sub)}ポジ）"
            )

        return True, "OK"

    # ── ポジション操作 ───────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        risk_pct: float,
        entry_price: float,
        entry_time: Optional[datetime] = None,
        is_extra: bool = False,
        meta: Optional[dict] = None,
    ) -> Position:
        """
        ポジションを登録する。
        can_enter() で True を確認してから呼ぶこと。

        Parameters
        ----------
        is_extra : bool
            含み益ポジション枠外で追加エントリーされた場合 True。
            損失転換時の全カット対象になる。
        """
        sym = symbol.upper()
        if sym not in SYMBOL_GROUP:
            raise ValueError(f"{sym} はグループ定義に存在しません")

        pos = Position(
            symbol=sym,
            risk_pct=risk_pct,
            entry_time=entry_time or datetime.now(timezone.utc),
            entry_price=entry_price,
            group=SYMBOL_GROUP[sym],
            subgroup=SYMBOL_SUBGROUP[sym],
            is_extra=is_extra,
            pnl=0.0,
            in_profit=False,
            meta=meta or {},
        )
        self._positions[sym] = pos
        return pos

    def update_pnl(self, symbol: str, pnl_jpy: float) -> None:
        """
        ポジションの含み損益を更新し、in_profit フラグを更新する。

        Parameters
        ----------
        symbol   : 銘柄名
        pnl_jpy  : 現在の含み損益（円）。プラスが含み益、マイナスが含み損。
        """
        sym = symbol.upper()
        if sym not in self._positions:
            return
        self._positions[sym].pnl = pnl_jpy
        self._positions[sym].in_profit = pnl_jpy > 0

    def check_and_cut(
        self,
        now: Optional[datetime] = None,
    ) -> list[str]:
        """
        損失転換チェックを行い、追加ポジション全カット対象の銘柄リストを返す。

        ロジック:
          - 含み損に転じたポジション（in_profit=False かつ pnl < 0）が存在する場合、
            is_extra=True のポジションを全て強制クローズ対象として返す
          - クールタイムを設定する
          - バックテスト本体側でリストを受け取り、close_position() を呼ぶこと

        Returns
        -------
        list[str]
            強制クローズ対象の銘柄名リスト（空リストの場合はカットなし）
        """
        # 含み損に転じたポジションが存在するか確認
        has_loss = any(
            p.pnl < 0 and not p.in_profit
            for p in self._positions.values()
        )
        if not has_loss:
            return []

        # 追加ポジション（is_extra=True）を全カット対象に
        to_cut = [sym for sym, p in self._positions.items() if p.is_extra]
        if not to_cut:
            return []

        # クールタイム設定
        t = now or datetime.now(timezone.utc)
        if not t.tzinfo:
            t = t.replace(tzinfo=timezone.utc)
        self._cooltime_until = t + timedelta(hours=COOLTIME_HOURS)

        return to_cut

    def set_cooltime(self, now: Optional[datetime] = None) -> None:
        """手動でクールタイムを設定する（テスト・手動操作用）"""
        t = now or datetime.now(timezone.utc)
        if not t.tzinfo:
            t = t.replace(tzinfo=timezone.utc)
        self._cooltime_until = t + timedelta(hours=COOLTIME_HOURS)

    def close_position(self, symbol: str) -> Optional[Position]:
        """ポジションを削除して返す。存在しない場合は None を返す。"""
        return self._positions.pop(symbol.upper(), None)

    def close_all(self) -> list[Position]:
        """全ポジションを削除して返す。"""
        closed = list(self._positions.values())
        self._positions.clear()
        return closed

    # ── 状態確認 ────────────────────────────────────────────

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def summary(self) -> str:
        lines = [
            f"=== PositionManager [yagami_position8_risk10_v2] ===",
            f"全体: {self.total_positions}/{MAX_TOTAL_POSITIONS}ポジ（含み益除外）"
            f"  全ポジ: {self.total_positions_all}  "
            f"リスク合計: {self.total_risk_pct*100:.1f}/{MAX_TOTAL_RISK_PCT*100:.0f}%",
        ]
        if self._cooltime_until:
            lines.append(f"クールタイム終了: {self._cooltime_until.strftime('%Y-%m-%d %H:%M UTC')}")
        for sym, p in self._positions.items():
            profit_tag = "【含み益・枠外】" if p.in_profit else "【含み損・枠内】"
            extra_tag  = "【追加】" if p.is_extra else ""
            lines.append(
                f"  {sym:8s} {profit_tag}{extra_tag} "
                f"pnl={p.pnl:+,.0f}円  risk={p.risk_pct*100:.1f}%"
            )
        if not self._positions:
            lines.append("  （保有なし）")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PositionManager("
            f"total={self.total_positions}/{MAX_TOTAL_POSITIONS}, "
            f"risk={self.total_risk_pct*100:.1f}%/{MAX_TOTAL_RISK_PCT*100:.0f}%, "
            f"cooltime={'ON' if self._cooltime_until else 'OFF'})"
        )


# ── 動作確認テスト ───────────────────────────────────────────
if __name__ == "__main__":
    from datetime import timezone
    now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    pm = PositionManager()
    print("=== yagami_position8_risk10_v2 動作確認テスト ===\n")

    # --- 基本エントリー ---
    print("--- 基本エントリー ---")
    ok, reason = pm.can_enter("US30", 0.02, now=now)
    print(f"US30: ok={ok}  {reason}")
    pm.open_position("US30", 0.02, entry_price=40000.0, entry_time=now, is_extra=False)

    ok, reason = pm.can_enter("AUDUSD", 0.02, now=now)
    print(f"AUDUSD: ok={ok}  {reason}")
    pm.open_position("AUDUSD", 0.02, entry_price=0.63, entry_time=now, is_extra=False)

    print(f"\n{pm.summary()}\n")

    # --- 含み益になったら枠外 ---
    print("--- US30が含み益に転換 → 枠外になり追加エントリー可能 ---")
    pm.update_pnl("US30", pnl_jpy=+50000)
    print(f"US30 in_profit={pm._positions['US30'].in_profit}")
    print(f"total_positions（含み益除外）: {pm.total_positions}")

    ok, reason = pm.can_enter("EURJPY", 0.02, now=now)
    print(f"EURJPY: ok={ok}  {reason}")
    is_extra = pm.is_extra_entry()
    print(f"is_extra_entry()={is_extra}")
    pm.open_position("EURJPY", 0.02, entry_price=160.0, entry_time=now, is_extra=is_extra)

    print(f"\n{pm.summary()}\n")

    # --- 損失転換 → 追加ポジション全カット ---
    print("--- US30が含み損に転換 → EURJPYを全カット ---")
    pm.update_pnl("US30", pnl_jpy=-10000)
    to_cut = pm.check_and_cut(now=now)
    print(f"カット対象: {to_cut}")
    for sym in to_cut:
        pm.close_position(sym)
    print(f"クールタイム終了: {pm._cooltime_until}")

    # --- クールタイム中はエントリー禁止 ---
    print("\n--- クールタイム中（2時間後）---")
    now2 = now + timedelta(hours=2)
    ok, reason = pm.can_enter("EURGBP", 0.02, now=now2)
    print(f"EURGBP（2時間後）: ok={ok}  {reason}")

    # --- クールタイム後はエントリー可能 ---
    print("\n--- クールタイム後（5時間後）---")
    now3 = now + timedelta(hours=5)
    ok, reason = pm.can_enter("EURGBP", 0.02, now=now3)
    print(f"EURGBP（5時間後）: ok={ok}  {reason}")

    print(f"\n{pm.summary()}")
    print("\n=== テスト完了 ===")
