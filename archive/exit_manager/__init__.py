"""
Exit Manager — OANDA v20 出口管理システム
==========================================
エントリーは人間（水原）が行う。
botは出口（損切・利確・ポジション管理）だけを強制する。

主要モジュール:
  oanda_client   - OANDA API 拡張クライアント
  position_manager - TradeState / TradeRegistry
  lot_calculator - JPYベースのロットサイズ計算
  exit_rules     - 出口ルールエンジン（Priority 1-9）
  logger         - JSONL 構造化ログ
  notifier       - Discord 通知
  main           - メインループ
  cli            - コマンドラインインターフェース
  evaluator      - 2週間評価スクリプト
"""

__version__ = '1.0.0'
