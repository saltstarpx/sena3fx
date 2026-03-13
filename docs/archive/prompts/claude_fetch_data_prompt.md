# Claude向け：USDJPYバックテスト用データ収集プロンプト

---

## プロンプト本文（ここからコピーして使用）

---

以下の仕様に従って、USDJPYのバックテスト用OHLCデータを取得・保存するPythonスクリプトを作成・実行してください。

---

### 取得対象

| 時間軸 | 保存ファイル名 | 取得期間 | 用途 |
|--------|--------------|---------|------|
| 1分足 | `usdjpy_is_1m.csv` | 2024-07-01 〜 2025-02-28 (UTC) | エントリー判定・SL/TP判定 |
| 15分足 | `usdjpy_is_15m.csv` | 2024-07-01 〜 2025-02-28 (UTC) | 1時間足の集約元 |
| 4時間足 | `usdjpy_is_4h.csv` | 2024-07-01 〜 2025-02-28 (UTC) | トレンド判定フィルター |
| 1分足 | `usdjpy_oos_1m.csv` | 2025-03-03 〜 2026-02-27 (UTC) | OOS検証用 |
| 15分足 | `usdjpy_oos_15m.csv` | 2025-03-03 〜 2026-02-27 (UTC) | OOS検証用 |
| 4時間足 | `usdjpy_oos_4h.csv` | 2025-03-03 〜 2026-02-27 (UTC) | OOS検証用 |

保存先: `/home/ubuntu/sena3fx/data/`

---

### データソース

**OANDA v20 REST API**（`price=M`、ミッドポイント価格）を使用すること。
APIキーは環境変数 `OANDA_API_KEY`、アカウントタイプは `practice` を使用。

```
エンドポイント: https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles
instrument: USD_JPY
granularity: M1 / M15 / H4
price: M  （ミッドポイント）
```

5000本ずつ分割取得して結合すること（OANDA APIの1リクエスト上限が5000本のため）。

---

### CSVファイル仕様（厳守）

```
カラム構成（6カラム固定）:
  timestamp, open, high, low, close, volume

タイムゾーン: UTC（例: 2024-07-01 00:00:00+00:00）
インデックス: なし（timestamp は列として保存）
保存方法: df.to_csv(path, index=False)
```

**カラム数は必ず6〜7以内**にすること。8を超えた場合はバグとして扱う。

---

### 禁止事項（バグ防止）

以下のコードパターンは**絶対に使用しないこと**。DataFrameに対して複数集計関数リストを渡すとMultiIndexが生成され、カラムが27個に崩れるバグが発生する。

```python
# NG（絶対禁止）
bars = df.resample('15min').agg(['first', 'max', 'min', 'last'])
```

必ず以下の形式を使うこと。

```python
# OK（正しい書き方）
bars = df['mid_close'].resample('15min').agg(
    open='first', high='max', low='min', close='last'
)
bars['volume'] = df['volume'].resample('15min').sum()
```

---

### 品質チェック（保存前に必ず実行）

保存前に以下を全て確認・修正すること。

**① タイムスタンプ重複の除去**

OANDA APIを月またぎで分割取得すると、月末16:00 UTCに同一タイムスタンプが2行生成されることがある。必ず除去すること。

```python
df = df.drop_duplicates(subset=['timestamp'], keep='first')
df = df.sort_values('timestamp').reset_index(drop=True)
```

**② OHLC整合性チェック**

```python
assert (df['high'] >= df['low']).all(),   "high < low の異常行あり"
assert (df['open'] >= df['low']).all(),   "open < low の異常行あり"
assert (df['open'] <= df['high']).all(),  "open > high の異常行あり"
assert (df['close'] >= df['low']).all(),  "close < low の異常行あり"
assert (df['close'] <= df['high']).all(), "close > high の異常行あり"
```

**③ ゼロ値・欠損値チェック**

```python
assert (df['close'] > 0).all(), "close=0 の異常値あり"
assert df[['open','high','low','close']].isnull().sum().sum() == 0, "欠損値あり"
```

**④ カラム数チェック**

```python
assert len(df.columns) <= 7, f"カラム数異常: {len(df.columns)}個"
assert not isinstance(df.columns, pd.MultiIndex), "MultiIndexカラムが存在する"
```

**⑤ 未確定足の除外**

OANDAの `complete=False` の足は取得しないこと。

```python
# APIレスポンスのパース時
for c in candles:
    if not c.get('complete', True):
        continue  # 未確定足はスキップ
```

---

### 保存後の確認出力

各ファイルを保存したら、以下の情報を出力すること。

```
[OK] usdjpy_is_1m.csv: 248731行, 2024-07-01 00:00:00+00:00 〜 2025-02-28 23:59:00+00:00, カラム数=6
[OK] usdjpy_is_15m.csv: 16629行, 2024-07-01 00:00:00+00:00 〜 2025-02-28 16:45:00+00:00, カラム数=6
[OK] usdjpy_is_4h.csv: 1042行, 2024-07-01 00:00:00+00:00 〜 2025-02-28 16:00:00+00:00, カラム数=6
（OOSも同様）
```

---

### 参考：既存ファイルの正常な行数（照合用）

| ファイル | 期待行数 | 期間 |
|---------|---------|------|
| `usdjpy_is_15m.csv` | 16,629行 | 2024-07-01 〜 2025-02-28 |
| `usdjpy_is_4h.csv` | 1,042行 | 2024-07-01 〜 2025-02-28 |
| `usdjpy_oos_15m.csv` | 24,814行 | 2025-03-03 〜 2026-02-27 |
| `usdjpy_oos_4h.csv` | 1,553行 | 2025-03-03 〜 2026-02-27 |

1分足は週末・祝日の休場時間帯（金曜22時〜日曜22時 UTC）はデータなしが正常。

---

### 保存後に実行するコマンド

```bash
cd /home/ubuntu/sena3fx
python3.11 scripts/check_ohlc_quality.py
```

`CRITICAL` と `ERROR` が **0件** であることを確認すること。1件でも出た場合は修正してから再チェックを行うこと。

---

## プロンプト終わり

---

## 補足メモ（水原さん向け）

- **1時間足は不要**。v77の `generate_signals()` が内部で15分足から `resample("1h")` して生成するため。
- 期間の区切りは IS（In-Sample）が 2024/7〜2025/2、OOS（Out-of-Sample）が 2025/3〜2026/2。
- `check_ohlc_quality.py` が最終的な品質の砦。Claudeに「このスクリプトを最後に必ず実行して結果を見せて」と付け加えると確実。
