---
name: "data-fetch"
description: "OANDA APIからOHLCデータを取得してdata/ohlc/に保存する。バックテスト用データ整備スキル。"
tags: [data, oanda, fetch, ohlc]
trigger: "データ取得"
---

# YAGAMI改 データ取得スキル

## データ仕様

| 系統 | パス | 銘柄名 | 用途 |
|---|---|---|---|
| **全期間** | `data/ohlc/` | 大文字（AUDUSD_1m.csv） | バックテスト主流 |
| IS/OOS分割 | `data/` | 小文字（audusd_is_15m.csv） | 旧スクリプト用 |

## 利用可能スクリプト

```bash
cd /home/user/sena3fx

# 全銘柄OHLC取得（推奨）
python scripts/fetch_all_ohlc.py

# OOSデータのみ取得
python scripts/fetch_oos_data.py

# FXデータ（リトライ付き）
python scripts/fetch_fx_robust.py

# 1m足から上位足を生成（4h/15mが欠けている場合）
python scripts/generate_htf_from_1m.py
```

## 銘柄別データ状況

| 銘柄 | 1m | 15m | 4h | 1d |
|---|---|---|---|---|
| XAUUSD | ✅ 全期間 | ✅ | ✅ | ✅ |
| AUDUSD | ✅ 全期間 | ✅ | ✅ | ✅ |
| EURUSD | ✅ 全期間 | ✅ | ✅ | ✅ |
| GBPUSD | ✅ 全期間 | ✅ | ✅ | ✅ |
| NZDUSD | ✅ 全期間 | ✅ | ✅ | — |
| USDJPY | ❌（1mなし）| ✅ | ✅ | ✅ |
| SPX500 | ✅ IS+OOS | ✅ | ✅ | — |

## 注意事項
- 1m足は大容量（20MB超）のため `.gitignore` 対象（`data/ohlc/*_1m.csv`）
- USDJPY に 1m足データは存在しない（バックテストは15mで代用）
- データ再取得で「データがない」と言われた場合は実際にファイルを確認すること:
  ```bash
  ls data/ohlc/ | head -20
  ```

## ロールバック手順
- API接続エラー → OANDA_TOKEN / ACCOUNT_ID を確認
- ファイル権限エラー → `chmod 644 data/ohlc/*.csv`
- データが空 → OANDA APIの取得期間上限（H4は5000本まで）を確認して分割取得
