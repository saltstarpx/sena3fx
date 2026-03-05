# quant_bot — クオンツ最強卍bot

sena3fx プロジェクトの自動取引エンジン（v1.0）。
既存の `lib/` コードを基盤として、やがみ5条件スコアリング + 指標トレード支援を統合。

## モジュール構成

| ディレクトリ | モジュール | 概要 |
|---|---|---|
| `data_pipeline/` | Module 4 | データ取得・整形・Parquet保存 |
| `entry_engine/` | Module 1 | やがみ5条件スコアリングエンジン |
| `indicator_trade/` | Module 2 | 指標トレード支援（マクロレジーム・カレンダー） |
| `backtest/` | Module 3 | バックテスト + JSONL ロガー |
| `rules/` | Module 5 | 教材準拠ルール管理 (YAML) |

## クイックスタート

```bash
# 依存パッケージのインストール
pip install beautifulsoup4 lxml pyarrow pyyaml pytest

# テスト全体実行（まずリーケージテストから）
cd /home/user/sena3fx
pytest quant_bot/tests/entry_engine/test_base_leakage.py -v
pytest quant_bot/tests/ -v --tb=short

# XAUUSDスキャン実行
python -c "
import sys, yaml, pandas as pd
sys.path.insert(0, '.')
from quant_bot.entry_engine.scanner import EntryScanner
with open('quant_bot/entry_engine/config.yaml') as f:
    config = yaml.safe_load(f)
df = pd.read_csv('data/ohlc/XAUUSD_4h.csv', index_col='datetime', parse_dates=True)
scanner = EntryScanner(config)
records = list(scanner.scan_dataframe(df, 'XAU_USD', 'H4'))
print(f'Signals found: {len(records)}')
for r in records[-3:]:
    import json; print(json.dumps(r, ensure_ascii=False, indent=2))
"

# マクロレジーム確認
python -c "
from quant_bot.indicator_trade.macro_regime import classify_regime
print(classify_regime(cpi_trend='rising', gdp_growth='positive'))
"
```

## 設計原則

1. **ゼロリーケージ**: `ConditionBase._confirmed()` が iloc[-1]（ライブバー）を常に除外
2. **ライブラリ不変**: `lib/` / `strategies/` / `live/` は一切変更しない
3. **教材準拠**: 全シグナルに `non_textbook: bool` フラグ付与
4. **設定外だし**: ハードコード禁止、全パラメータは `config.yaml` で管理
5. **APIキー**: `OANDA_API_KEY`, `OANDA_ACCOUNT_ID` は環境変数のみ

## 環境変数

```bash
export OANDA_API_KEY="your-api-key"
export OANDA_ACCOUNT_ID="your-account-id"
export OANDA_ENVIRONMENT="practice"  # or "live"
```

## ディレクトリ依存関係

```
data_pipeline/  ← 独立（scripts/fetch_data.py のみ参照）
     ↓
entry_engine/   ← lib/levels, lib/candle, lib/patterns, lib/timing を参照
     ↓
backtest/       ← lib/backtest, lib/risk_manager を参照
indicator_trade/ ← 独立（外部 API のみ参照）
rules/          ← YAML のみ（コード依存なし）
```
