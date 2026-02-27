# XAUUSD OHLC Market Data

本物のマーケットデータ。Yahoo Finance (GC=F: Gold Futures) から取得。
合成データ（GBM）ではない。

## ファイル一覧

| ファイル | 時間足 | 期間 | バー数 | ソース |
|:---|:---|:---|:---|:---|
| `XAUUSD_1d.csv` | 日足 | 2019-01-02 〜 2026-02-27 | 1,801 | Yahoo Finance |
| `XAUUSD_1h.csv` | 1時間足 | 2023-10-06 〜 2026-02-27 | 13,693 | Yahoo Finance |
| `XAUUSD_4h.csv` | 4時間足 | 2023-10-06 〜 2026-02-27 | 3,714 | 1Hからリサンプリング |
| `XAUUSD_8h.csv` | 8時間足 | 2023-10-06 〜 2026-02-27 | 1,937 | 1Hからリサンプリング |

## CSV形式

```
datetime,open,high,low,close,volume
```

- `datetime`: ISO 8601形式
- `open/high/low/close`: USD建て価格
- `volume`: 取引量

## 注意事項

- GC=F（金先物）はXAUUSD（スポット金）と微小な価格差がある
- バックテスト用途としては十分に近似している
- 取得日: 2026-02-27 by Manus AI
