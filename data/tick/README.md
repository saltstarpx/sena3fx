# ティックデータ管理

## 概要

このディレクトリには、XAUUSD（金スポット）のティックデータをgzip圧縮形式（`.csv.gz`）で保管します。全ファイルはGit LFSで管理されています。

## ファイル命名規則

```
ticks_{銘柄}_{YYYY-MM}.csv.gz
```

例: `ticks_XAUUSD_2024-11.csv.gz`

## データ形式

元データのCSVフォーマットは以下の通りです（タブ区切り）。

| カラム | 説明 |
|:---|:---|
| `<DATE>` | 日付（YYYY.MM.DD形式） |
| `<TIME>` | 時刻（HH:MM:SS.mmm形式） |
| `<BID>` | ビッド価格 |
| `<ASK>` | アスク価格 |
| `<LAST>` | 最終価格 |
| `<VOLUME>` | 出来高 |

## 収録データ一覧

| ファイル名 | 期間 | 圧縮後サイズ | 行数（概算） |
|:---|:---|:---:|:---:|
| `ticks_XAUUSD_2024-11.csv.gz` | 2024年11月 | 56MB | 約912万行 |
| `ticks_XAUUSD_2025-12.csv.gz` | 2025年12月 | 55MB | 約794万行 |
| `ticks_XAUUSD_2026-02.csv.gz` | 2026年2月 | 66MB | 約909万行 |

## 解凍方法

```bash
# 単一ファイルの解凍
gunzip -c ticks_XAUUSD_2024-11.csv.gz > ticks_XAUUSD_2024-11.csv

# Pythonでの読み込み（解凍不要）
import pandas as pd
df = pd.read_csv('ticks_XAUUSD_2024-11.csv.gz', sep='\t', compression='gzip')
```

## 今後の追加方針

毎月末に当月分のティックデータをgzip圧縮してこのディレクトリに追加します。Git LFSの無料枠（1GB）を超えた場合は、古いデータをアーカイブストレージに移行することを検討します。
