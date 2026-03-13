# Teammate A: Yagami Strategy Documentation

## 概要

やがみメソッドに基づく高勝率シグナル追求戦略。
5条件の複合評価でエントリーを判定し、養分トレードを排除する。

**評価基準**: WR (勝率) > 60%, PF (プロフィットファクター) > 1.8

## やがみメソッド 5条件

| # | 条件 | 実装 | 説明 |
|---|------|------|------|
| 1 | レジサポの位置 | `lib/levels.py` | ヒゲ先端・実体からS/R水平線を抽出。`is_at_level()` で判定 |
| 2 | ローソク足の強弱 | `lib/candle.py` | 大陽線・包み足・ハンマー・ピンバー・Doji等のパターン検出 |
| 3 | プライスアクション | `lib/candle.py` | リバーサルロー/ハイ・ダブルボトム・ヒゲ埋め・実体揃い |
| 4 | チャートパターン | `lib/patterns.py` | 逆三尊・フラッグ・ウェッジ・三角持ち合い等 |
| 5 | 足更新タイミング | `lib/timing.py` | 上位足更新時、5分間隔エントリー |

### 評価グレード

- **A評価 (4-5条件)**: 高品質エントリー
- **B評価 (3条件)**: 慎重にエントリー
- **C評価 (2条件以下)**: エントリー禁止 (養分)

### 禁止ルール

- 大陽線/大陰線への逆張り禁止 (「逆らうと死にます」)
- アジア時間のブレイクアウトは見送り
- Doji 4本以上連続 → トレンドレス → ノーポジ
- 二番底/二番天井を待ってからエントリー

## シグナル関数一覧

コア実装は `lib/yagami.py`、戦略ハブは `strategies/yagami_rules.py`。

### 基本シグナル

| 関数 | 説明 |
|------|------|
| `sig_yagami_A(freq)` | A評価のみ (4-5条件) の厳選エントリー |
| `sig_yagami_B(freq)` | B評価以上 (3条件以上) のエントリー |
| `sig_yagami_reversal_only(freq)` | リバーサルロー/ハイのみ (最強シグナル) |
| `sig_yagami_double_bottom(freq)` | ダブルボトム/トップ + 足更新タイミング |
| `sig_yagami_pattern_break(freq)` | チャートパターンブレイク + レジサポ + 足更新 |
| `sig_yagami_london_ny(freq)` | ロンドン/NYセッション限定のB評価以上 |

### 強化フィルター付き

| 関数 | 説明 |
|------|------|
| `sig_yagami_filtered(freq, ...)` | botter AC知見統合 (Vol/Trend/Time/Momentum) |
| `sig_yagami_vol_regime(freq)` | ボラティリティレジーム付き (ATR比率 0.6-2.2) |
| `sig_yagami_trend_regime(freq)` | トレンドレジーム付き (EMA200方向) |
| `sig_yagami_prime_time(freq)` | ロンドン/NYオープン限定 |
| `sig_yagami_full_filter(freq)` | 全フィルター統合 |
| `sig_yagami_A_full_filter(freq)` | A評価 + 全フィルター |

### MTF (マルチタイムフレーム)

| 関数 | 説明 |
|------|------|
| `sig_yagami_mtf_cascade(bars_dict)` | MTFカスケード (複数時間足を辞書で渡す) |
| `sig_yagami_mtf_4h_1h(freq)` | 4H+1H のMTFフィルター |
| `sig_yagami_breakout(freq)` | やがみ式ブレイクアウト |
| `sig_yagami_breakout_filtered(freq)` | フィルター付きブレイクアウト |

### USD強弱フィルター付きバリアント (`strategies/yagami_rules.py`)

| 変数名 | ベース |
|--------|--------|
| `sig_yagami_A_usd` | `sig_yagami_A` + USD threshold=75 |
| `sig_yagami_B_usd` | `sig_yagami_B` + USD threshold=75 |
| `sig_yagami_full_usd` | `sig_yagami_full_filter` + USD threshold=75 |
| `sig_yagami_lonny_usd` | `sig_yagami_london_ny` + USD threshold=75 |

## 薄いゾーン戦略との連携

`price_zone_analyzer.py` が検出した薄いゾーン (滞在本数が少ない価格帯) と連携する。

```python
from price_zone_analyzer import load_thin_zones, get_thin_zone_params

zones = load_thin_zones()
params = get_thin_zone_params(current_price, zones)

if params['is_thin']:
    lot_size *= params['lot_scale']       # 0.7x (ロット縮小)
    stop_distance *= params['stop_scale'] # 1.3x (ストップ拡大)
```

- 薄いゾーンではボラティリティ拡大が予想されるため、ロットを縮小しストップを拡大
- `price_zone_analyzer.py` は4H足データから $10幅ビンの滞在本数ヒストグラムを計算
- 下位15%以下のビンを薄いゾーンとして `data/thin_zones.json` に保存

## overheat_monitor との連携

`overheat_monitor.py` はXAUTとXAUUSDの価格乖離を監視し、市場の過熱/冷却を検知する。

```python
from overheat_monitor import OverheatMonitor

monitor = OverheatMonitor()
monitor.update()

if monitor.is_overheated():
    # 乖離率 >= +0.3% → 新規ロングエントリー停止
    pass
if monitor.is_cooled_down():
    # 乖離率 <= -0.3% → ロング再開検討
    pass
```

- **OVERHEAT** (乖離 >= +0.3%): XAUT割高 = 過熱警戒 → 新規ロング停止推奨
- **COOLDOWN** (乖離 <= -0.3%): XAUT割安 = 冷却サイン → ロング再開検討
- **NORMAL**: 通常運用
- データソース: Bybit公開API (XAUT/USDT) + OANDA (XAU/USD)
- 監視間隔: 15分、ログは `data/overheat_log.csv`

## バックテスト実行例

```python
from strategies.yagami_rules import yagami_variants

variants = yagami_variants(freq='4h')
for name, sig_func in variants:
    result = engine.run(data=df, signal_func=sig_func, name=name)
```
