"""
utils/position_manager.py
=========================
ポジション管理システム（全体・グループ・サブグループ・相関グループ制約）

【設計仕様】
  全体ポジション上限    : 8ポジ
  全体トータルリスク上限 : 10%
  グループ内上限        : 2ポジ（FX / 貴金属 / 指数）
  サブグループ内上限    : 1ポジ（相関の高い銘柄群）
  相関グループ内上限    : 1ポジ（最優先フィルター）

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

    # エントリー可否チェック
    ok, reason = pm.can_enter("EURUSD", risk_pct=0.02)
    if ok:
        pm.open_position("EURUSD", risk_pct=0.02, entry_time=..., meta={...})

    # 決済
    pm.close_position("EURUSD")

    # 現在の状態確認
    print(pm.summary())
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

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
# 同グループ内では1ポジション限定
# 1銘柄が複数グループに属する場合、いずれかのグループで1ポジ保有中なら
# 同グループの他銘柄はエントリー禁止
CORRELATION_GROUPS: dict[str, list[str]] = {
    "equity_risk": ["US30", "SPX500", "NAS100"],
    "risk_on_fx":  ["EURJPY", "GBPJPY", "AUDJPY"],
    "eur_pairs":   ["EURUSD", "EURJPY", "EURGBP"],
    "gbp_pairs":   ["GBPUSD", "GBPJPY", "EURGBP"],
}

# 銘柄 → 所属する相関グループ名リスト（複数可）
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


@dataclass
class Position:
    """保有中のポジション情報"""
    symbol:     str
    risk_pct:   float
    entry_time: datetime
    group:      str
    subgroup:   str
    meta:       dict = field(default_factory=dict)


class PositionManager:
    """
    全銘柄のポジションを一元管理するクラス。
    バックテストと本番EA の両方から使用可能。
    """

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}  # symbol -> Position

    # ── 現在の集計値 ────────────────────────────────────────

    @property
    def total_positions(self) -> int:
        return len(self._positions)

    @property
    def total_risk_pct(self) -> float:
        return sum(p.risk_pct for p in self._positions.values())

    def group_positions(self, group: str) -> int:
        return sum(1 for p in self._positions.values() if p.group == group)

    def subgroup_positions(self, subgroup: str) -> int:
        return sum(1 for p in self._positions.values() if p.subgroup == subgroup)

    def corr_group_positions(self, corr_group: str) -> int:
        """相関グループ内の保有ポジション数を返す"""
        members = CORRELATION_GROUPS.get(corr_group, [])
        return sum(1 for s in self._positions if s in members)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    # ── エントリー可否チェック ───────────────────────────────

    def can_enter(self, symbol: str, risk_pct: float) -> tuple[bool, str]:
        """
        エントリー可否を判定する。

        チェック順序（最優先から）:
          1. 同銘柄保有中チェック
          2. 相関グループ内上限（最優先フィルター）
          3. 全体ポジション上限
          4. 全体トータルリスク上限
          5. グループ内上限
          6. サブグループ内上限

        Parameters
        ----------
        symbol   : 銘柄名（例: "EURUSD"）
        risk_pct : このトレードのリスク率（例: 0.02 = 2%）

        Returns
        -------
        tuple[bool, str]
            (可否, 理由)
        """
        sym = symbol.upper()

        # 1. 既に同銘柄を保有中
        if self.has_position(sym):
            return False, f"{sym}: 既に保有中"

        # 2. 相関グループ内上限チェック（最優先）
        corr_groups = SYMBOL_CORR_GROUPS.get(sym, [])
        for cgrp in corr_groups:
            if self.corr_group_positions(cgrp) >= MAX_CORR_POSITIONS:
                # どの銘柄が競合しているか特定
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

        # 3. 全体ポジション上限
        if self.total_positions >= MAX_TOTAL_POSITIONS:
            return False, (
                f"{sym}: 全体ポジション上限 {MAX_TOTAL_POSITIONS}ポジ に達しています "
                f"（現在 {self.total_positions}ポジ）"
            )

        # 4. 全体トータルリスク上限
        if self.total_risk_pct + risk_pct > MAX_TOTAL_RISK_PCT:
            return False, (
                f"{sym}: 全体リスク上限 {MAX_TOTAL_RISK_PCT*100:.0f}% を超えます "
                f"（現在 {self.total_risk_pct*100:.1f}% + {risk_pct*100:.1f}% = "
                f"{(self.total_risk_pct+risk_pct)*100:.1f}%）"
            )

        # 5. グループ内上限
        if self.group_positions(grp) >= MAX_GROUP_POSITIONS:
            return False, (
                f"{sym}: グループ [{grp}] の上限 {MAX_GROUP_POSITIONS}ポジ に達しています "
                f"（現在 {self.group_positions(grp)}ポジ）"
            )

        # 6. サブグループ内上限
        if self.subgroup_positions(sub) >= MAX_SUBGROUP_POSITIONS:
            return False, (
                f"{sym}: サブグループ [{sub}] の上限 {MAX_SUBGROUP_POSITIONS}ポジ に達しています "
                f"（現在 {self.subgroup_positions(sub)}ポジ）"
            )

        return True, "OK"

    # ── ポジション操作 ───────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        risk_pct: float,
        entry_time: Optional[datetime] = None,
        meta: Optional[dict] = None,
    ) -> Position:
        """
        ポジションを登録する。
        can_enter() で True を確認してから呼ぶこと。
        """
        sym = symbol.upper()
        if sym not in SYMBOL_GROUP:
            raise ValueError(f"{sym} はグループ定義に存在しません")

        pos = Position(
            symbol=sym,
            risk_pct=risk_pct,
            entry_time=entry_time or datetime.utcnow(),
            group=SYMBOL_GROUP[sym],
            subgroup=SYMBOL_SUBGROUP[sym],
            meta=meta or {},
        )
        self._positions[sym] = pos
        return pos

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
        """現在の全ポジションを返す（読み取り専用）。"""
        return dict(self._positions)

    def summary(self) -> str:
        """現在の状態をテキストで返す。"""
        lines = [
            f"=== PositionManager ===",
            f"全体: {self.total_positions}/{MAX_TOTAL_POSITIONS}ポジ  "
            f"リスク合計: {self.total_risk_pct*100:.1f}/{MAX_TOTAL_RISK_PCT*100:.0f}%",
        ]
        for grp in _GROUPS:
            n = self.group_positions(grp)
            if n > 0:
                syms = [s for s, p in self._positions.items() if p.group == grp]
                lines.append(f"  [{grp}] {n}/{MAX_GROUP_POSITIONS}ポジ: {', '.join(syms)}")
        # 相関グループの状態
        active_corr = []
        for cgrp, members in CORRELATION_GROUPS.items():
            held = [s for s in self._positions if s in members]
            if held:
                active_corr.append(f"{cgrp}:{','.join(held)}")
        if active_corr:
            lines.append(f"  [相関グループ] {' | '.join(active_corr)}")
        if not self._positions:
            lines.append("  （保有なし）")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PositionManager("
            f"total={self.total_positions}/{MAX_TOTAL_POSITIONS}, "
            f"risk={self.total_risk_pct*100:.1f}%/{MAX_TOTAL_RISK_PCT*100:.0f}%)"
        )


# ── 動作確認テスト ───────────────────────────────────────────
if __name__ == "__main__":
    pm = PositionManager()
    print("=== PositionManager 動作確認テスト ===\n")

    print("--- 相関グループフィルターテスト ---")
    corr_tests = [
        # (銘柄, リスク%, 期待結果, 説明)
        ("US30",   0.02, True,  "OK: equity_risk 1ポジ目"),
        ("SPX500", 0.02, False, "NG: equity_risk 競合（US30保有中）"),
        ("NAS100", 0.02, False, "NG: equity_risk 競合（US30保有中）"),
        ("EURJPY", 0.02, True,  "OK: eur_pairs/risk_on_fx 1ポジ目（FX 1ポジ目）"),
        ("EURGBP", 0.02, False, "NG: eur_pairs 競合（EURJPY保有中）"),
        ("EURUSD", 0.02, False, "NG: eur_pairs 競合（EURJPY保有中）"),
        ("GBPJPY", 0.02, False, "NG: risk_on_fx 競合（EURJPY保有中）"),
        ("AUDUSD", 0.02, True,  "OK: 相関グループ無関係 + FX 2ポジ目"),
        ("USDJPY", 0.02, False, "NG: FXグループ上限（2ポジ達成）"),
    ]

    for sym, risk, expected, desc in corr_tests:
        ok, reason = pm.can_enter(sym, risk)
        status = "✓" if ok == expected else "✗ FAIL"
        print(f"  {status}  {sym:8s}  ok={ok}  {desc}")
        if ok:
            pm.open_position(sym, risk)

    print(f"\n{pm.summary()}")

    print("\n--- 決済後の再エントリーテスト ---")
    pm.close_position("US30")
    ok, reason = pm.can_enter("SPX500", 0.02)
    print(f"  US30決済後 SPX500: ok={ok}  {reason}")

    pm.close_position("EURJPY")
    ok, reason = pm.can_enter("EURGBP", 0.02)
    print(f"  EURJPY決済後 EURGBP: ok={ok}  {reason}")

    print("\n=== テスト完了 ===")
