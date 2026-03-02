"""
quant_bot — クオンツ最強卍bot
==============================
sena3fx プロジェクトの自動取引エンジン。
既存の lib/ を土台として、以下のモジュールで構成される:

  data_pipeline/    Module 4: データ取得・整形・Parquet保存
  entry_engine/     Module 1: やがみ5条件スコアリングエンジン
  indicator_trade/  Module 2: 指標トレード支援
  backtest/         Module 3: イベント駆動バックテスト + JSONL
  rules/            Module 5: 教材準拠ルール管理 (YAML)

注意: lib/ / strategies/ / live/ は一切変更しない。
      quant_bot/ は import のみで既存コードを参照する。
"""
__version__ = "1.0.0"
