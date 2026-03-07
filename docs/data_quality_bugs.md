# ローソク足データ品質バグ — 詳細調査報告書

**調査日:** 2026年3月7日  
**対象システム:** sena3fx v76 ペーパートレーディングBot  
**対象データ:** USDJPY / EURJPY / GBPJPY（OANDA v20 APIから取得）

---

## 概要

2026年3月のフォルダ整理作業中に、`scripts/check_ohlc_quality.py` による品質チェックを実施した結果、以下の2種類のバグが発見された。いずれもバックテスト結果の信頼性に影響を与える可能性があるため、即時修正を行った。

| バグID | 名称 | 深刻度 | 影響ファイル数 | 修正日 |
|--------|------|--------|--------------|--------|
| BUG-001 | 27カラムバグ（MultiIndexカラム崩れ） | CRITICAL | 2ファイル | 2026-03-07 |
| BUG-002 | 月末タイムスタンプ重複 | ERROR | 4ファイル | 2026-03-07 |

---

## BUG-001: 27カラムバグ（MultiIndexカラム崩れ）

### 症状

`USDJPY_15m.csv` および `USDJPY_4h.csv` を読み込むと、カラム数が正常な6〜7個ではなく **27個** になっていた。カラム名は `open`, `high`, `low`, `close`, `open.1`, `high.1`, `low.1`, `close.1`, ..., `open.5`, `high.5`, `low.5`, `close.5`, `volume`, `tick_count` という異常な構造になっており、ほとんどの行が `NaN` で埋まっていた。

```
# 異常ファイルのカラム構造（実際の出力）
['timestamp', 'open', 'high', 'low', 'close',
 'open.1', 'high.1', 'low.1', 'close.1',
 'open.2', 'high.2', 'low.2', 'close.2',
 'open.3', 'high.3', 'low.3', 'close.3',
 'open.4', 'high.4', 'low.4', 'close.4',
 'open.5', 'high.5', 'low.5', 'close.5',
 'volume', 'tick_count']
```

さらに、先頭行以外はほぼすべて `NaN` であり、データとして機能していなかった。

### 根本原因

pandas の `resample().agg()` を **DataFrame 全体**に対して複数集計関数リストで適用した際に、**MultiIndex カラム**が生成される。このMultiIndexを持つDataFrameをそのまま `to_csv()` で保存すると、pandas がカラム名を自動的に連番サフィックス（`.1`, `.2`, ...）付きで平坦化してしまう。

**問題のあるコードパターン（再現例）:**

```python
# NG: DataFrameに対してagg(['first','max','min','last'])を適用
df_ticks = pd.DataFrame({
    'open': [...], 'high': [...], 'low': [...], 'close': [...],
    'volume': [...], 'tick_count': [...]
}, index=datetime_index)

# これにより (open,first),(open,max),...,(close,last) のMultiIndexが生成される
bars = df_ticks.resample('15min').agg(['first', 'max', 'min', 'last'])
bars.to_csv('USDJPY_15m.csv')  # ← カラム名が崩れた状態で保存される
```

**生成されるMultiIndexの構造:**

```
(open, first), (open, max), (open, min), (open, last),
(high, first), (high, max), (high, min), (high, last),
(low,  first), (low,  max), (low,  min), (low,  last),
(close,first), (close,max), (close,min), (close,last),
(volume,first),(volume,max),(volume,min),(volume,last),
(tick_count,first),...
```

これを `to_csv()` で保存すると、同名カラムに連番が振られ `open`, `open.1`, `open.2`, `open.3`, `high`, `high.1`, ... という27カラムのCSVが生成される。

### バックテストへの影響

このファイルを読み込んだバックテストスクリプトは、`open` カラムには最初の1行（2024-07-01 00:00）の値のみが入り、それ以外はすべて `NaN` であるため、**ほぼすべてのシグナル判定がスキップされる**。結果として、バックテストのトレード数が極端に少なくなるか、エラーで停止する。

実際に `usdjpy_15m_old.csv`（旧形式、669行）との比較から、このバグが発生する前は正常に約670本のデータが存在していたことが確認できる。

### 修正内容

異常ファイルを `data/archive/` へ移動し、正しい形式で生成された `usdjpy_is_15m.csv`（16,629行）および `usdjpy_is_4h.csv`（1,042行）を正規データとして採用した。

### 再発防止策

**正しいコードパターン:**

```python
# OK: bidPrice Series に対して named aggregation を使用
bars = df_ticks['bidPrice'].resample('15min').agg(
    open='first',
    high='max',
    low='min',
    close='last'
)
bars['spread']     = (df_ticks['askPrice'] - df_ticks['bidPrice']).resample('15min').mean()
bars['tick_count'] = df_ticks['bidPrice'].resample('15min').count()
```

**チェックポイント（データ保存前に必ず確認）:**

```python
# カラム数の事前検証
assert len(df.columns) <= 8, f"カラム数異常: {len(df.columns)}個 - {df.columns.tolist()}"
# MultiIndexの検出
assert not isinstance(df.columns, pd.MultiIndex), "MultiIndexカラムが存在します"
```

品質チェックスクリプト `scripts/check_ohlc_quality.py` は、カラム数が8を超えるファイルを `[CRITICAL]` として検出する。

---

## BUG-002: 月末タイムスタンプ重複

### 症状

4時間足データ（`usdjpy_is_4h.csv`, `usdjpy_oos_4h.csv`, `eurjpy_4h.csv`, `gbpjpy_4h.csv`）において、**毎月末の16:00 UTC** に同一タイムスタンプを持つ2行が存在していた。

**実際の重複データ例（usdjpy_oos_4h.csv, 2025-03-31 16:00 UTC）:**

```
timestamp                   open      high      low       close
2025-03-31 16:00:00+00:00   149.5985  149.8890  149.4075  149.5185  ← 月末最終足
2025-03-31 16:00:00+00:00   149.5175  149.9865  149.4635  149.8955  ← 翌月最初足（誤）
```

2行目の `open`（149.5175）が1行目の `close`（149.5185）とほぼ一致していることから、2行目は実際には **翌月（4月）最初の4時間足**であることが確認できる。同様のパターンがすべての月末に発生しており、1年間のデータで5〜9件の重複が存在していた。

| ファイル | 重複件数 | 修正後行数 |
|---------|---------|----------|
| `usdjpy_is_4h.csv` | 5件 | 1,047 → 1,042行 |
| `usdjpy_oos_4h.csv` | 7件 | 1,560 → 1,553行 |
| `eurjpy_4h.csv` | 9件 | 1,814 → 1,805行 |
| `gbpjpy_4h.csv` | 9件 | 1,814 → 1,805行 |

### 根本原因

OANDA v20 API の `candles` エンドポイントは、**月をまたぐ取得リクエスト**を行うと、月末最終足と翌月最初足に同一のタイムスタンプ（月末 16:00 UTC）を付与して返すことがある。これはOANDA APIの既知の挙動であり、複数の月にまたがるデータを分割取得して `pd.concat()` で結合する際に顕在化する。

**発生メカニズム:**

```
取得リクエスト1: 2025-01-01 〜 2025-03-31
  → 最後の足: 2025-03-31 16:00 (close=149.5185)

取得リクエスト2: 2025-03-31 〜 2025-06-30  ← 境界が重なる
  → 最初の足: 2025-03-31 16:00 (open=149.5175)  ← 実際は4月最初の足
```

`fetch_data.py` の `fetch_oanda_bars()` 関数では `pd.concat(all_dfs)` 後に `drop_duplicates(keep='first')` を呼んでいるが、これはインデックスベースの重複削除であり、タイムスタンプが列として保存されている場合には機能しない。

### バックテストへの影響

4時間足は v76 戦略において**トレンド方向の判定**（上位足フィルター）に使用される。同一タイムスタンプに2本の足が存在すると、以下の問題が発生する可能性がある。

1. **シグナルの二重発火:** バックテストエンジンが同じ4時間足バーで2回シグナルを評価し、同一タイムスタンプに2回エントリーが発生する。
2. **パターン検出の誤動作:** 二番底・二番天井パターンの検出ロジックが、重複行を「連続した2本の足」として誤認識する可能性がある。
3. **統計の歪み:** 月末に集中するため、月末前後のパフォーマンス統計が実態より良く（または悪く）見える可能性がある。

### 修正内容

`drop_duplicates(subset=['timestamp'], keep='first')` を適用し、各タイムスタンプの最初の行（月末最終足）を保持して2行目（翌月最初足）を削除した。

```python
# 修正コード（scripts/fix_ohlc_data.py より）
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df_fixed = df.drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)
df_fixed = df_fixed.sort_values('timestamp').reset_index(drop=True)
df_fixed.to_csv(filepath, index=False)
```

### 再発防止策

**データ取得スクリプトへの組み込み（推奨）:**

```python
def fetch_oanda_bars_safe(instrument, granularity, days, api_key):
    """重複チェック付きOANDAデータ取得"""
    df = fetch_oanda_bars(instrument, granularity, days, api_key)
    if df is None:
        return None
    
    # タイムスタンプ重複チェック・修正
    before = len(df)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    df = df.sort_values('timestamp').reset_index(drop=True)
    after = len(df)
    
    if before != after:
        print(f"[WARN] タイムスタンプ重複を修正: {before - after}件削除 ({instrument} {granularity})")
    
    return df
```

**品質チェックの定期実行:**

4時間足データを新規取得・更新した後は、必ず以下を実行する。

```bash
python scripts/check_ohlc_quality.py
```

`[ERROR]` が出力された場合は `scripts/fix_ohlc_data.py` を参考に修正を行う。

---

## 品質チェックの運用手順

### 新規データ取得後のチェックフロー

```
1. OANDAからデータ取得
       ↓
2. python scripts/check_ohlc_quality.py を実行
       ↓
3. 結果を確認
   ├── CRITICAL/ERROR が 0件 → そのまま使用可
   └── CRITICAL/ERROR が 1件以上 → 修正が必要
           ↓
4. 修正: drop_duplicates / カラム再構築
       ↓
5. 再チェック（CRITICAL/ERROR が 0件になるまで繰り返す）
```

### チェックスクリプトの検出項目

`scripts/check_ohlc_quality.py` は以下の項目を自動チェックする。

| 項目 | 深刻度 | 説明 |
|------|--------|------|
| カラム数 > 8 | CRITICAL | 27カラムバグ等の異常構造を検出 |
| タイムスタンプ重複 | ERROR | 同一タイムスタンプの複数行を検出 |
| high < low | ERROR | OHLC整合性の異常 |
| open/close が high/low の範囲外 | ERROR | OHLC整合性の異常 |
| 欠損値（NaN） | WARN | 必須カラムの欠損 |
| 10%超のスパイク | WARN | 価格の異常変動 |
| close = 0 | ERROR | ゼロ値の異常データ |

---

## 修正済みファイルの保管場所

修正前の異常ファイルは `data/archive/` に保管されており、参照・比較目的で利用できる。

| ファイル | 問題 | 保管場所 |
|---------|------|---------|
| `USDJPY_15m.csv` | 27カラムバグ | `data/archive/USDJPY_15m.csv` |
| `USDJPY_4h.csv` | 27カラムバグ | `data/archive/USDJPY_4h.csv` |
| `usdjpy_15m_fixed.csv` | 旧版（usdjpy_is_15m.csvと重複） | `data/archive/usdjpy_15m_fixed.csv` |
| `usdjpy_15m_old.csv` | 旧形式（tick_count付き） | `data/archive/usdjpy_15m_old.csv` |
| `USDJPY_1m.csv` | 旧版（usdjpy_is_1m.csvと重複） | `data/archive/USDJPY_1m.csv` |

---

## 関連ファイル

- `scripts/check_ohlc_quality.py` — 品質チェックスクリプト
- `scripts/fix_ohlc_data.py` — 2026年3月実施の修正スクリプト（修正内容の記録）
- `data/archive/` — 修正前の異常ファイル保管場所
