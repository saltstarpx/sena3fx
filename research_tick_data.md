# ティックデータソース調査結果

## 最有力: Dukascopy Bank SA (無料)
- **dukascopy-python** ライブラリ (PyPI, v4.0.1, 2025年4月リリース)
- ティックレベルデータ対応（bidPrice, askPrice, bidVolume, askVolume）
- Forex, Commodities, Indices対応
- XAUUSDは "INSTRUMENT_COMMODITIES_XAU_USD" として利用可能
- 無料、アカウント不要
- Pythonで直接DataFrameとして取得可能

## 使い方
```python
from datetime import datetime
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_COMMODITIES_XAU_USD

df = dukascopy_python.fetch(
    INSTRUMENT_COMMODITIES_XAU_USD,
    dukascopy_python.INTERVAL_TICK,
    dukascopy_python.OFFER_SIDE_BID,
    datetime(2025, 1, 1),
    datetime(2025, 2, 1),
)
```

## ティックデータの列
- timestamp (UTC)
- bidPrice
- askPrice
- bidVolume
- askVolume

## その他のソース
- Polygon.io: Forexティックデータは有料プラン必要
- Kaggle: 一部XAUUSDティックデータあり（期間限定）
- Tickstory: デスクトップアプリ（GUIのみ）
- ForexTester: GUIソフト（自動操作困難）
