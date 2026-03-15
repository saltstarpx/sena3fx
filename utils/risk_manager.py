"""
utils/risk_manager.py
=====================
全通貨ペア対応の共通資金管理ユーティリティ

【使い方】
    from utils.risk_manager import RiskManager

    rm = RiskManager("EURUSD")
    lot = rm.calc_lot(equity=1_000_000, sl_distance=0.0005, ref_price=1.0850)
    print(f"ロットサイズ: {lot:.0f}通貨")

【設計思想】
- 銘柄名を渡すだけで pip_size / spread / 円換算ロジックが自動設定される
- SL距離はv77本体のATRベース（ATR × 0.15）で決まるため固定値最適化なし
- 損切額 = 総資産 × risk_pct（デフォルト3%、資産規模で逓減）を厳守
- バックテストでも本番EAでも同一コードで動作する

【通貨ペアタイプと円換算】
  Type A: XXX/JPY  (USDJPY, EURJPY, GBPJPY)
    → 1通貨の損益 = 価格差（円）× ロット数
    → lot = risk_jpy / sl_distance

  Type B: XXX/USD  (EURUSD, GBPUSD, AUDUSD, NZDUSD)
    → 1通貨の損益 = 価格差（USD）× USDJPY × ロット数
    → lot = risk_jpy / (sl_distance × usdjpy_rate)

  Type C: USD/XXX  (USDCAD, USDCHF)
    → 1通貨の損益 = 価格差（USD/XXX）× USDJPY / 現在価格 × ロット数
    → lot = risk_jpy / (sl_distance × usdjpy_rate / ref_price)
    ※ ref_price = エントリー時の現在価格（raw_ep）

  Type D: 指数（US30, SPX500, NAS100）
    → 1ptの損益 = USDJPY × ロット数（コントラクトサイズ依存）
    → lot = risk_jpy / (sl_distance × usdjpy_rate)
    ※ US30/SPX500はcfd_multiplier=1として扱う（Exness標準）

【スプレッド設定（fxfan.club / Exness 2026.3.8 計測値）】
  採用ルール:
    FX主要ペア: ロースプレッド口座 最小スプレッド
    クロス円（EURJPY等）: スタンダード口座 最小スプレッド
    貴金属（XAUUSD, XAGUSD）: ロースプレッド口座 最小スプレッド
    指数（US30, SPX500, NAS100）: ゼロ口座 平均スプレッド
  出典: https://www.fxfan.club/?p=59656
  USDJPY=0.0,  EURUSD=0.0,  GBPUSD=0.1,  AUDUSD=0.0,  NZDUSD=0.5
  USDCAD=0.1,  USDCHF=0.2
  EURJPY=2.4,  GBPJPY=2.2,  AUDJPY=1.9,  NZDJPY=4.3,  CADJPY=3.8
  CHFJPY=2.4,  HKDJPY=3.0
  US30=0.8pt,  SPX500=0.1pt, NAS100=8.3pt
  XAUUSD=5.2,  XAGUSD=2.6
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

# ── 銘柄マスタ ────────────────────────────────────────────
# key: 銘柄名（大文字）
# pip_size  : 1pip = 何価格単位か（SL距離の最小単位）
# spread    : Exness ゼロ口座の生スプレッド（pips）
# quote_type: 円換算タイプ（A=円建て / B=USD建て / C=逆USD建て / D=指数）
# color     : チャート用カラーコード

# スプレッド採用ルール（fxfan.club / Exness 2026.3.8 計測値）:
# FX主要ペア: ロースプレッド口座 最小スプレッド
# クロス円（EURJPY等）: スタンダード口座 最小スプレッド
# 貴金属: ロースプレッド口座 最小スプレッド
# 指数: ゼロ口座 平均スプレッド
# ── Exness ロット上限（1注文あたり） ────────────────────────────
# 日中 (UTC 07:00-20:59): max_lots_day   標準ロット
# 夜間 (UTC 21:00-06:59): max_lots_night 標準ロット
# contract_size: 1標準ロットあたりの通貨数（FX=100,000, XAUUSD=100oz）
# エキゾチック通貨は50ロット上限（該当銘柄なし）
EXNESS_LOT_DAY   = 200   # 日中: 最大200ロット/注文
EXNESS_LOT_NIGHT = 20    # 夜間: 最大20ロット/注文
EXNESS_DAY_START = 7     # UTC 07:00〜
EXNESS_DAY_END   = 21    # 〜UTC 20:59

SYMBOL_CONFIG: dict[str, dict] = {
    # FX主要ペア（ロースプレッド口座 / fxfan.club 2026.3.8）
    "USDJPY": {"pip": 0.01,   "spread": 0.0,  "quote_type": "A", "color": "#ef4444", "account": "raw_spread", "contract_size": 100_000},
    "EURUSD": {"pip": 0.0001, "spread": 0.0,  "quote_type": "B", "color": "#f97316", "account": "raw_spread", "contract_size": 100_000},
    "GBPUSD": {"pip": 0.0001, "spread": 0.1,  "quote_type": "B", "color": "#eab308", "account": "raw_spread", "contract_size": 100_000},
    "AUDUSD": {"pip": 0.0001, "spread": 0.0,  "quote_type": "B", "color": "#22c55e", "account": "raw_spread", "contract_size": 100_000},
    "USDCAD": {"pip": 0.0001, "spread": 0.1,  "quote_type": "C", "color": "#14b8a6", "account": "raw_spread", "contract_size": 100_000},
    "USDCHF": {"pip": 0.0001, "spread": 0.2,  "quote_type": "C", "color": "#3b82f6", "account": "raw_spread", "contract_size": 100_000},
    "NZDUSD": {"pip": 0.0001, "spread": 0.5,  "quote_type": "B", "color": "#8b5cf6", "account": "raw_spread", "contract_size": 100_000},
    # クロス円（スタンダード口座 / fxfan.club 2026.3.8）
    "EURJPY": {"pip": 0.01,   "spread": 2.4,  "quote_type": "A", "color": "#ec4899", "account": "standard", "contract_size": 100_000},
    "GBPJPY": {"pip": 0.01,   "spread": 2.2,  "quote_type": "A", "color": "#f43f5e", "account": "standard", "contract_size": 100_000},
    "AUDJPY": {"pip": 0.01,   "spread": 1.9,  "quote_type": "A", "color": "#10b981", "account": "standard", "contract_size": 100_000},
    "NZDJPY": {"pip": 0.01,   "spread": 4.3,  "quote_type": "A", "color": "#6366f1", "account": "standard", "contract_size": 100_000},
    "CADJPY": {"pip": 0.01,   "spread": 3.8,  "quote_type": "A", "color": "#f472b6", "account": "standard", "contract_size": 100_000},
    "CHFJPY": {"pip": 0.01,   "spread": 2.4,  "quote_type": "A", "color": "#a78bfa", "account": "raw_spread", "contract_size": 100_000},
    "HKDJPY": {"pip": 0.001,  "spread": 3.0,  "quote_type": "A", "color": "#fb923c", "account": "raw_spread", "contract_size": 100_000},
    "EURGBP": {"pip": 0.0001, "spread": 0.40, "quote_type": "B_GBP", "color": "#a855f7", "account": "zero", "contract_size": 100_000},
    # 指数（ゼロ口座 平均スプレッド / fxfan.club 2026.3.8）
    "US30":   {"pip": 1.0,    "spread": 0.8,  "quote_type": "D", "color": "#f59e0b", "account": "zero", "contract_size": 1},
    "SPX500": {"pip": 0.1,    "spread": 0.1,  "quote_type": "D", "color": "#06b6d4", "account": "zero", "contract_size": 1},
    "NAS100": {"pip": 1.0,    "spread": 8.3,  "quote_type": "D", "color": "#84cc16", "account": "zero", "contract_size": 1},
    # 貴金属（ロースプレッド口座 平均スプレッド / fxfan.club 2026.3.8）
    "XAUUSD": {"pip": 0.01,   "spread": 5.2,  "quote_type": "B", "color": "#d97706", "account": "raw_spread", "contract_size": 100},
    "XAGUSD": {"pip": 0.001,  "spread": 2.6,  "quote_type": "B", "color": "#6b7280", "account": "raw_spread", "contract_size": 5_000},
}


class RiskManager:
    """
    銘柄名を渡すだけで全通貨ペア対応のロットサイズを計算するクラス。

    Parameters
    ----------
    symbol : str
        銘柄名（例: "EURUSD", "USDJPY"）。大文字・小文字どちらでも可。
    risk_pct : float
        1トレードあたりのリスク割合（デフォルト: 0.03 = 3%、資産規模で逓減）
    """

    def __init__(self, symbol: str, risk_pct: float = 0.03):
        self.symbol    = symbol.upper()
        self.risk_pct  = risk_pct

        if self.symbol not in SYMBOL_CONFIG:
            raise ValueError(
                f"銘柄 '{self.symbol}' はサポートされていません。"
                f"対応銘柄: {list(SYMBOL_CONFIG.keys())}"
            )

        cfg = SYMBOL_CONFIG[self.symbol]
        self.pip_size   = cfg["pip"]
        self.spread_pips = cfg["spread"]
        self.quote_type = cfg["quote_type"]
        self.color      = cfg["color"]

    @property
    def spread(self) -> float:
        """スプレッドを価格単位で返す（pips × pip_size）"""
        return self.spread_pips * self.pip_size

    def calc_lot(
        self,
        equity: float,
        sl_distance: float,
        ref_price: float,
        usdjpy_rate: Optional[float] = None,
        gbpjpy_rate: Optional[float] = None,
    ) -> float:
        """
        ロットサイズ（通貨数）を計算する。

        Parameters
        ----------
        equity : float
            現在の総資産（円）
        sl_distance : float
            SLまでの価格差（チャートレベル、raw_ep - sl の絶対値）
        ref_price : float
            エントリー時の現在価格（raw_ep）。Type C（USDCAD等）の換算に使用。
        usdjpy_rate : float, optional
            USDJPY の現在レート。Type B/C/D の換算に必要。
            バックテスト時は data_1m の直近終値から自動取得を推奨。
            None の場合はデフォルト値（150.0）を使用。
        gbpjpy_rate : float, optional
            GBPJPY の現在レート。Type B_GBP（EURGBP）の換算に必要。
            None の場合は usdjpy_rate × ref_price で近似。

        Returns
        -------
        float
            ロットサイズ（通貨数）。0以下の場合は0を返す。
        """
        if sl_distance <= 0 or equity <= 0:
            return 0.0

        risk_jpy = equity * self.risk_pct  # 許容損失額（円）

        # USDJPY レートのデフォルト（バックテスト時は実データから取得を推奨）
        if usdjpy_rate is None or usdjpy_rate <= 0:
            usdjpy_rate = 150.0

        qt = self.quote_type

        if qt == "A":
            # XXX/JPY: 価格差がそのまま円
            # lot = risk_jpy / sl_distance
            lot = risk_jpy / sl_distance

        elif qt == "B":
            # XXX/USD: 価格差（USD）× USDJPY → 円
            # lot = risk_jpy / (sl_distance × usdjpy_rate)
            lot = risk_jpy / (sl_distance * usdjpy_rate)

        elif qt == "B_GBP":
            # EUR/GBP: 価格差（GBP）× GBPJPY → 円
            # GBPJPY ≈ USDJPY × ref_price（EURGBP価格）で近似
            if gbpjpy_rate is None or gbpjpy_rate <= 0:
                gbpjpy_rate = usdjpy_rate * ref_price
            lot = risk_jpy / (sl_distance * gbpjpy_rate)

        elif qt == "C":
            # USD/XXX: 損益はUSD建て（価格差 / 現在価格 × USDJPY）
            # 例: USDCAD で SL距離0.005、現在価格1.35、USDJPY=150
            # 1通貨の損失 = 0.005 / 1.35 × 150 = 0.556円
            # lot = risk_jpy / (sl_distance / ref_price × usdjpy_rate)
            lot = risk_jpy / (sl_distance / ref_price * usdjpy_rate)

        elif qt == "D":
            # 指数（US30, SPX500, NAS100）: 1pt = USDJPY 円
            # Exness の CFD はコントラクトサイズ1として計算
            lot = risk_jpy / (sl_distance * usdjpy_rate)

        else:
            lot = 0.0

        return max(lot, 0.0)

    def max_units(self, entry_hour_utc: Optional[int] = None) -> float:
        """
        Exnessの1注文あたり最大ユニット数（通貨数/oz数）を返す。

        Parameters
        ----------
        entry_hour_utc : int, optional
            エントリー時刻のUTC時（0-23）。
            日中(7-20): 200ロット、夜間(21-6): 20ロット。
            None の場合は夜間上限（保守的）を返す。

        Returns
        -------
        float
            最大ユニット数（通貨数 or oz数）
        """
        cfg = SYMBOL_CONFIG[self.symbol]
        cs  = cfg.get("contract_size", 100_000)
        if entry_hour_utc is not None and EXNESS_DAY_START <= entry_hour_utc < EXNESS_DAY_END:
            return cs * EXNESS_LOT_DAY
        return cs * EXNESS_LOT_NIGHT

    def cap_lot(self, lot: float, entry_hour_utc: Optional[int] = None) -> tuple:
        """
        ロットサイズをExness上限でキャップする。

        Returns
        -------
        tuple[float, bool]
            (capped_lot, was_capped)
        """
        mx = self.max_units(entry_hour_utc)
        if lot > mx:
            return mx, True
        return lot, False

    def calc_pnl_jpy(
        self,
        direction: int,
        ep: float,
        exit_price: float,
        lot: float,
        usdjpy_rate: Optional[float] = None,
        ref_price: Optional[float] = None,
    ) -> float:
        """
        損益を円で計算する。

        Parameters
        ----------
        direction : int
            1=ロング, -1=ショート
        ep : float
            エントリー価格（スプレッド込みの実約定価格）
        exit_price : float
            決済価格
        lot : float
            ロットサイズ（通貨数）
        usdjpy_rate : float, optional
            USDJPY レート（Type B/C/D で使用）
        ref_price : float, optional
            エントリー時の現在価格（Type C で使用）

        Returns
        -------
        float
            損益（円）。プラスが利益、マイナスが損失。
        """
        if usdjpy_rate is None or usdjpy_rate <= 0:
            usdjpy_rate = 150.0
        if ref_price is None or ref_price <= 0:
            ref_price = ep

        price_diff = (exit_price - ep) * direction  # プラスが利益方向
        qt = self.quote_type

        if qt == "A":
            return price_diff * lot
        elif qt == "B":
            return price_diff * lot * usdjpy_rate
        elif qt == "B_GBP":
            gbpjpy = usdjpy_rate * ref_price
            return price_diff * lot * gbpjpy
        elif qt == "C":
            return price_diff / ref_price * lot * usdjpy_rate
        elif qt == "D":
            return price_diff * lot * usdjpy_rate
        else:
            return 0.0

    def get_usdjpy_rate(self, data_1m: pd.DataFrame, entry_time) -> float:
        """
        バックテスト時にエントリー時点の USDJPY レートを 1分足データから取得する。

        Parameters
        ----------
        data_1m : pd.DataFrame
            USDJPY の 1分足データ（close カラムが必要）
        entry_time : pd.Timestamp
            エントリー時刻

        Returns
        -------
        float
            USDJPY レート。取得できない場合は 150.0 を返す。
        """
        try:
            before = data_1m[data_1m.index <= entry_time]
            if len(before) > 0:
                return float(before.iloc[-1]["close"])
        except Exception:
            pass
        return 150.0

    def calc_commission_jpy(
        self,
        lot_size: float,
        usdjpy_rate: float = 150.0,
        side: str = "open",
    ) -> float:
        """
        ゼロ口座の片道手数料を円沿いで返す。

        ゼロ口座の手数料規定: 片道0.2ドル / 100,000通貨（1標準ロット）
        つまり 0.2ドル / 100,000通貨 × lot_size通貨 × USDJPYレート

        Parameters
        ----------
        lot_size : float
            ロットサイズ（通貨数）
        usdjpy_rate : float
            USDJPYレート（円換算用）
        side : str
            "open"（エントリー）または "close"（クローズ）→ 片道分のみ

        Returns
        -------
        float
            手数料（円）。常に正の値（資産から引く金額）。
        """
        # ゼロ口座: 0.2ドル / 100,000通貨
        commission_usd = 0.2 * (lot_size / 100_000)
        commission_jpy = commission_usd * usdjpy_rate
        return commission_jpy

    def calc_roundtrip_commission_jpy(
        self,
        lot_size: float,
        usdjpy_rate: float = 150.0,
    ) -> float:
        """
        ゼロ口座の往復手数料合計（円）を返す。
        エントリー時 + クローズ時の両方をまとめて計算する際に使用。
        """
        return self.calc_commission_jpy(lot_size, usdjpy_rate) * 2

    def __repr__(self) -> str:
        return (
            f"RiskManager({self.symbol}, risk={self.risk_pct*100:.0f}%, "
            f"spread={self.spread_pips}pips, type={self.quote_type})"
        )


# ── 便利関数: 銘柄名からRiskManagerを生成 ─────────────────
def get_risk_manager(symbol: str, risk_pct: float = 0.03) -> RiskManager:
    """銘柄名からRiskManagerインスタンスを返す。"""
    return RiskManager(symbol, risk_pct)


# ── 便利関数: 銘柄設定の一覧を返す ───────────────────────
def list_symbols() -> list[str]:
    """サポートされている銘柄名の一覧を返す。"""
    return list(SYMBOL_CONFIG.keys())


def get_symbol_config(symbol: str) -> dict:
    """銘柄の設定（pip_size, spread, quote_type）を返す。"""
    return SYMBOL_CONFIG.get(symbol.upper(), {})


# ── 動作確認 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RiskManager 動作確認")
    print("=" * 60)

    equity     = 1_000_000  # 100万円
    usdjpy     = 150.0

    test_cases = [
        # (symbol,   sl_distance, ref_price, 説明)
        ("USDJPY",  0.50,   150.0,  "SL50pips"),
        ("EURUSD",  0.0005, 1.085,  "SL5pips"),
        ("GBPUSD",  0.0010, 1.270,  "SL10pips"),
        ("AUDUSD",  0.0005, 0.650,  "SL5pips"),
        ("USDCAD",  0.0010, 1.350,  "SL10pips"),
        ("USDCHF",  0.0010, 0.900,  "SL10pips"),
        ("NZDUSD",  0.0005, 0.600,  "SL5pips"),
        ("EURJPY",  0.50,   163.0,  "SL50pips"),
        ("GBPJPY",  0.50,   192.0,  "SL50pips"),
        ("EURGBP",  0.0005, 0.850,  "SL5pips"),
        ("US30",    50.0,   38000,  "SL50pt"),
        ("SPX500",  5.0,    5000,   "SL5pt"),
    ]

    print(f"{'銘柄':<10} {'タイプ':<8} {'スプレッド':<12} {'SL距離':<12} "
          f"{'ロット(通貨)':<14} {'リスク額(円)':<14} {'説明'}")
    print("-" * 90)

    for sym, sl_dist, ref, desc in test_cases:
        rm  = RiskManager(sym)
        lot = rm.calc_lot(equity, sl_dist, ref, usdjpy_rate=usdjpy)

        # 損失額の検証（SLまで動いたときの損失が2万円になるか確認）
        pnl = rm.calc_pnl_jpy(
            direction=1, ep=ref, exit_price=ref - sl_dist,
            lot=lot, usdjpy_rate=usdjpy, ref_price=ref
        )

        print(f"{sym:<10} {rm.quote_type:<8} {rm.spread_pips:<12} {sl_dist:<12} "
              f"{lot:<14.1f} {abs(pnl):<14.0f} {desc}")

    print("-" * 90)
    print(f"※ リスク額は全て {equity*0.03:,.0f}円（総資産{equity:,}円の3%）になるはず")


# ══════════════════════════════════════════════════════════════
# AdaptiveRiskManager: 資産規模 × DD 連動型リスク逓減マネージャー
# ══════════════════════════════════════════════════════════════
class AdaptiveRiskManager(RiskManager):
    """
    資産規模とドローダウンの両方に連動してリスク%を自動調整するクラス。

    【ルール】
    - 資産規模テーブルと DD テーブルをそれぞれ評価し、
      より低い方のリスク% を採用する（保守的な方を優先）。

    【資産規模テーブル（加速成長 → 逓減）】
    資産 〜 1,000万円:       base_risk_pct × 1.00  (3.0%)
    資産 1,000万〜3,000万円: base_risk_pct × 0.833 (2.5%)
    資産 3,000万〜5,000万円: base_risk_pct × 0.667 (2.0%)
    資産 5,000万〜1億円:     base_risk_pct × 0.50  (1.5%)
    資産 1億円〜:            base_risk_pct × 0.333 (1.0%)

    【DD テーブル】
    DD  0%未満:  × 1.00  (通常)
    DD  5%以上:  × 0.75
    DD 10%以上:  × 0.50
    DD 15%以上:  × 0.25  (最小)

    【採用ルール】
    effective_risk = min(資産規模テーブル値, DDテーブル値)
    """

    # 資産規模テーブル: (資産下限, リスク乗数)
    # base_risk_pct=0.03 との組合せで 3.0% → 2.5% → 2.0% → 1.5% → 1.0%
    EQUITY_STEPS = [
        (0,           1.000),  # 〜1,000万円:       3.0%
        (10_000_000,  0.833),  # 1,000万〜3,000万円: 2.5%
        (30_000_000,  0.667),  # 3,000万〜5,000万円: 2.0%
        (50_000_000,  0.500),  # 5,000万〜1億円:     1.5%
        (100_000_000, 0.333),  # 1億円〜:           1.0%
    ]

    # 絶対下限リスク率（1億円超 + DD15%でも0.5%を下回らない）
    RISK_FLOOR = 0.005

    # DD テーブル: (DD閾値, リスク乗数)
    DD_STEPS = [
        (0.00,  1.00),   # DD  0%未満 → 通常
        (0.05,  0.75),   # DD  5%以上 → ×0.75
        (0.10,  0.50),   # DD 10%以上 → ×0.50
        (0.15,  0.25),   # DD 15%以上 → ×0.25（最小）
    ]

    def __init__(self, symbol: str, base_risk_pct: float = 0.03):
        super().__init__(symbol, risk_pct=base_risk_pct)
        self.base_risk_pct = base_risk_pct
        self.peak_equity   = None   # 過去最高資産額
        self._current_dd   = 0.0   # 現在のDD率（参照用）

    def update_peak(self, equity: float) -> None:
        """資産の最高値を更新する（毎トレード後・半利確後に呼ぶ）"""
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity

    def current_dd_rate(self, equity: float) -> float:
        """現在のドローダウン率を返す"""
        if self.peak_equity is None or self.peak_equity <= 0:
            return 0.0
        dd = (self.peak_equity - equity) / self.peak_equity
        self._current_dd = max(dd, 0.0)
        return self._current_dd

    def equity_risk_multiplier(self, equity: float) -> float:
        """資産規模テーブルから乗数を返す"""
        mult = 1.00
        for threshold, m in self.EQUITY_STEPS:
            if equity >= threshold:
                mult = m
        return mult

    def dd_risk_multiplier(self, equity: float) -> float:
        """DDテーブルから乗数を返す"""
        dd   = self.current_dd_rate(equity)
        mult = 1.00
        for threshold, m in self.DD_STEPS:
            if dd >= threshold:
                mult = m
        return mult

    def effective_risk_pct(self, equity: float) -> tuple:
        """
        資産規模テーブルと DD テーブルの両方を評価し、
        より低い方（保守的な方）のリスク% を返す。
        RISK_FLOOR で絶対下限を保証する。

        Returns
        -------
        tuple[float, str]
            (effective_risk_pct, reason)
            reason: 'equity' / 'dd' / 'floor' / 'base'
        """
        eq_mult = self.equity_risk_multiplier(equity)
        dd_mult = self.dd_risk_multiplier(equity)

        eq_risk = self.base_risk_pct * eq_mult
        dd_risk = self.base_risk_pct * dd_mult

        if eq_risk <= dd_risk:
            raw    = eq_risk
            reason = 'equity' if eq_mult < 1.0 else 'base'
        else:
            raw    = dd_risk
            reason = 'dd' if dd_mult < 1.0 else 'base'

        # 絶対下限保証
        final = max(raw, self.RISK_FLOOR)
        if final > raw:
            reason = 'floor'

        return final, reason

    def calc_lot_adaptive(
        self,
        equity: float,
        sl_distance: float,
        ref_price: float,
        usdjpy_rate: float = 150.0,
    ) -> tuple:
        """
        資産規模 × DD 連動でロットを計算する。

        Returns
        -------
        tuple[float, float, str]
            (lot, effective_risk_pct, reason)
            reason: 'equity' / 'dd' / 'floor' / 'base'
        """
        eff_risk, reason = self.effective_risk_pct(equity)
        self.risk_pct    = eff_risk  # 親クラスの calc_lot に渡す値を一時変更

        lot = self.calc_lot(
            equity=equity,
            sl_distance=sl_distance,
            ref_price=ref_price,
            usdjpy_rate=usdjpy_rate,
        )

        self.risk_pct = self.base_risk_pct  # 元に戻す
        return lot, eff_risk, reason

    def __repr__(self) -> str:
        return (
            f"AdaptiveRiskManager({self.symbol}, base_risk={self.base_risk_pct*100:.0f}%, "
            f"peak={self.peak_equity}, dd={self._current_dd:.1%}, "
            f"spread={self.spread_pips}pips, type={self.quote_type})"
        )


# ── 動作確認（AdaptiveRiskManager） ──────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AdaptiveRiskManager 動作確認")
    print("=" * 60)

    arm = AdaptiveRiskManager("NZDUSD", base_risk_pct=0.03)
    usdjpy = 150.0
    sl_dist = 0.00074
    ref_price = 0.565

    test_equities = [
        (1_000_000,   0.00, "初期資産100万・DD0%"),
        (950_000,     0.05, "資産95万・DD5%"),
        (900_000,     0.10, "資産90万・DD10%"),
        (850_000,     0.15, "資産85万・DD15%"),
        (10_000_000,  0.00, "資産1,000万・DD0%"),
        (30_000_000,  0.00, "資産3,000万・DD0%"),
        (30_000_000,  0.08, "資産3,000万・DD8%"),
        (100_000_000, 0.00, "資産1億・DD0%"),
    ]

    print(f"\n{'資産':>14} {'DD':>6} {'資産乗数':>8} {'DD乗数':>8} "
          f"{'実効リスク':>10} {'損切額':>10} {'説明'}")
    print("-" * 80)

    for equity, dd_rate, desc in test_equities:
        # ピークを設定してDDを再現
        arm.peak_equity = equity / (1 - dd_rate) if dd_rate < 1 else equity
        eq_mult = arm.equity_risk_multiplier(equity)
        dd_mult = arm.dd_risk_multiplier(equity)
        eff, reason = arm.effective_risk_pct(equity)
        risk_amt = equity * eff
        lot, _, _  = arm.calc_lot_adaptive(equity, sl_dist, ref_price, usdjpy)
        print(f"{equity:>14,.0f} {dd_rate:>6.0%} {eq_mult:>8.2f} {dd_mult:>8.2f} "
              f"{eff:>10.2%} {risk_amt:>10,.0f}円  {desc}")
