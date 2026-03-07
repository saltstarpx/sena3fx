# Claude Code 指示書 v7: 通貨強弱分析結果と戦略統合

## 前回からの変更点

通貨強弱ベースのXAGクロスペア動的選択について、徹底的なバックテストを実施した。

## 通貨強弱分析の結論（重要）

### 検証結果サマリー

| アプローチ | 動的累積 | XAGUSD固定 | 改善幅 | 判定 |
|---|---|---|---|---|
| v1: 単純最弱通貨選択 | +150.29% | +154.97% | -4.68% | NG |
| v2: ブレンド強弱+加速度+閾値 | +64.37% | +63.45% | +0.92% | 微妙 |
| v3: テクニカルスコア統合 | +220.01% | +231.76% | -11.75% | NG |
| ポートフォリオ分散 | +44.81% | +119.38% | -74.57% | NG（リスク低減のみ） |

**結論: 通貨強弱による動的ペア切替はXAGUSD固定に勝てない。**

### 理由の分析
1. XAGUSDは最も流動性が高く、テクニカルが最もクリーンに機能する
2. 合成ペア（XAGJPY等）はスプレッドが広く、ノイズが大きい
3. 通貨強弱は遅行指標であり、切替タイミングが遅れる
4. ペア切替のトランザクションコストが改善分を食う

### ただし有用な発見
- **XAGCHF**: CHFが弱い時（下位25%）のロングはWR=56.9%、平均+0.24%/日
- **XAGJPY**: 大きなトレンド時に累積+91.88%（ただし平均は低い）
- **ポートフォリオ効果**: 最大DD -19.52% vs XAGUSD単独 -36.18%（リスク半減）

## 推奨する戦略統合方法

### 1. メイン戦略はXAGUSD固定を維持
既存のNT_4H_UNION2d40_RR5等の戦略をそのまま使用。通貨強弱でペアを切り替えない。

### 2. 通貨強弱を「フィルター」として活用
```python
# USDが極端に強い時（上位25%）はXAGUSDロングを避ける
if usd_strength > usd_strong_threshold:
    skip_xagusd_long = True
```
これにより、逆風の中でのロングエントリーを回避できる。

### 3. リスク管理への活用
```python
# 通貨強弱によるポジションサイズ調整
if usd_strength > 0:  # USD強い = XAGUSDロングに逆風
    position_size *= 0.7  # 30%減
elif usd_strength < -1.0:  # USD弱い = XAGUSDロングに追い風
    position_size *= 1.0  # フルサイズ
```

### 4. 将来的なマルチペア拡張（Phase 2以降）
OANDA APIで実際にXAGCHF、XAGJPYが取引可能か確認後、
- XAGCHFのロング専用戦略（CHF弱い時のみ）
- XAGJPYのトレンドフォロー（大きなトレンド時のみ）
を追加ペアとして検討。

## 既存戦略への統合コード

`scripts/currency_strength_engine.py` に通貨強弱計算ロジックがある。
以下の関数を既存戦略に統合すること：

```python
from scripts.currency_strength_portfolio import calc_usd_strength

# バックテスト内で使用
usd_str = calc_usd_strength('1d', lookback=20)

# エントリーフィルター
def should_enter_long(date, usd_str, threshold=1.343):
    """USDが強すぎる時はロングを避ける"""
    if date in usd_str.index:
        return usd_str.loc[date] < threshold
    return True  # データなしの場合はデフォルトでエントリー許可
```

## 優先タスク

1. **既存のNT_4H戦略にUSD強弱フィルターを追加**してバックテスト
2. **季節フィルター（7月・9月回避）との併用効果**を検証
3. **ポジションサイズ調整ロジック**の実装
4. **OANDA APIでのXAGCHF/XAGJPY取引可否**の確認

## データファイル

以下のファイルがリポジトリに追加済み：
- `scripts/currency_strength_engine.py` - v1通貨強弱計算
- `scripts/currency_strength_v2.py` - v2改良版
- `scripts/currency_strength_v3.py` - v3テクニカル統合版
- `scripts/currency_strength_portfolio.py` - ポートフォリオ版（最終版）
- `results/v3_parameter_sweep.csv` - パラメータスイープ結果
- `results/portfolio_summary.json` - ポートフォリオ分析サマリー
- `data/ohlc/XAGJPY_1d.csv` - 合成XAGJPYデータ
- `data/ohlc/XAGCHF_1d.csv` - 合成XAGCHFデータ
- `data/ohlc/XAGEUR_1d.csv` - 合成XAGEURデータ
- `data/ohlc/XAGGBP_1d.csv` - 合成XAGGBPデータ
- `data/ohlc/XAGAUD_1d.csv` - 合成XAGAUDデータ
- `data/ohlc/XAGNZD_1d.csv` - 合成XAGNZDデータ
- `data/ohlc/XAGCAD_1d.csv` - 合成XAGCADデータ
- FXペアデータ: USDJPY, EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF (1d/1h)
