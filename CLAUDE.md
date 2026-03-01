# CLAUDE.md

sena3fx — 戦略ポートフォリオPDCAエージェント v4.0

## プロジェクト概要

XAUUSD（金）のバックテストとPDCA自動化システム。
戦略ポートフォリオ（Yagami / Maedai / Risk Manager）による多角的評価。

- **対象通貨ペア**: XAUUSD（OANDA）
- **データソース**: Yahoo Finance (GC=F) → `data/ohlc/`
- **ブランチ**: `main`
- **リモート**: GitHub `saltstarpx/sena3fx`

## 戦略ポートフォリオ (Agent Teams)

| チーム | 役割 | 評価基準 | 詳細 |
|--------|------|----------|------|
| **A: Yagami** | 高勝率シグナル | WR>60%, PF>1.8 | `docs/strategy_yagami.md` |
| **B: Maedai** | 高RRトレンドフォロー | Sharpe>1.5, Calmar>3.0 | `docs/strategy_maedai.md` |
| **C: Risk Manager** | 市場環境フィルター | MDD低減, Sharpe改善 | `docs/filters_risk_manager.md` |

## ディレクトリ構成

```
sena3fx/
├── CLAUDE.md                # AI開発ガイド（このファイル）
├── README.md
├── .gitignore
├── knowledge.json           # PDCA蓄積知見
│
├── lib/                     # コアエンジン
│   ├── __init__.py
│   ├── candle.py            #   ローソク足パターン検出
│   ├── patterns.py          #   チャートパターン検出
│   ├── levels.py            #   レジサポ自動検出
│   ├── timing.py            #   足更新タイミング・MTF・セッション
│   ├── yagami.py            #   やがみ5条件統合シグナル（コア）
│   ├── backtest.py          #   バックテストエンジン v3.0
│   └── indicators.py        #   インジケーター戦略（SMA/RSI/BB/MACD）
│
├── strategies/              # 戦略ポートフォリオ (v4.0 新設)
│   ├── market_filters.py    #   Teammate C: USD強弱・季節フィルター
│   ├── yagami_rules.py      #   Teammate A: Yagami戦略バリアント
│   └── maedai_breakout.py   #   Teammate B: Maedai戦略バリアント
│
├── scripts/                 # 実行スクリプト
│   ├── main_loop.py         #   PDCA自動実行ループ
│   ├── backtest_portfolio.py#   戦略ポートフォリオ統合バックテスト
│   └── fetch_data.py        #   Dukascopyティックデータ取得
│
├── docs/                    # 戦略ドキュメント + 参考資料
│   ├── strategy_yagami.md   #   Teammate A 詳細
│   ├── strategy_maedai.md   #   Teammate B 詳細
│   ├── filters_risk_manager.md # Teammate C 詳細
│   ├── ローソク足の本｜やがみ.pdf
│   ├── ローソク足の本2｜やがみ.pdf
│   ├── ポジり方の本｜やがみ.pdf
│   └── *.md                 #   要件定義・リサーチメモ
│
├── archive/                 # 旧バージョン（v1/v2コード・旧レポート）
│
├── data/ohlc/               # OHLC CSV (Yahoo Finance GC=F)
├── data/tick/               # ティックデータCSV（gitignore対象）
├── results/                 # バックテスト結果CSV + PNG
├── reports/                 # PDCAレポート
└── trade_logs/              # トレードログ
```

## やがみメソッド 5条件

エントリーは以下5条件の複合評価で判定（`lib/yagami.py`）:

1. **レジサポの位置** (`lib/levels.py`) — ヒゲ先端・実体からS/R水平線を抽出
2. **ローソク足の強弱** (`lib/candle.py`) — 大陽線・包み足・ハンマー・ピンバー・Doji等
3. **プライスアクション** (`lib/candle.py`) — リバーサルロー/ハイ・ダブルボトム・ヒゲ埋め・実体揃い
4. **チャートパターン** (`lib/patterns.py`) — 逆三尊・フラッグ・ウェッジ・三角持ち合い等
5. **足更新タイミング** (`lib/timing.py`) — 上位足更新時・5分間隔エントリー

### 評価グレード
- **A評価（4-5条件）**: 高品質エントリー
- **B評価（3条件）**: 慎重にエントリー
- **C評価（2条件以下）**: エントリー禁止（養分）

### 禁止ルール
- 大陽線/大陰線への逆張り禁止（「逆らうと死にます」）
- アジア時間のブレイクアウトは見送り
- Doji4本以上連続 → トレンドレス → ノーポジ
- 二番底/二番天井を待ってからエントリー

## 実行コマンド

```bash
# PDCAサイクル1回実行
python scripts/main_loop.py

# ティックデータ取得（Dukascopy）
python scripts/fetch_data.py

# 統合テスト
python -c "from lib.candle import *; from lib.yagami import *; print('OK')"
```

## 依存パッケージ

`pip install numpy pandas` で十分（lzma, structは標準ライブラリ）。

## バックテスト合格基準

### Yagami (高勝率)
| 指標 | 基準 |
|------|------|
| プロフィットファクター | ≥ 1.8 |
| 最大ドローダウン | ≤ 15% |
| 勝率 | ≥ 60% |
| トレード数 | ≥ 30 |

### Maedai (高RR)
| 指標 | 基準 |
|------|------|
| Sharpe Ratio | > 1.5 |
| Calmar Ratio | > 3.0 |
| 年間トレード数 | ≥ 50 |
| 最大ドローダウン | ≤ 30% |

## 注意事項

- `archive/`は旧v1/v2コード。新規開発は`lib/`配下 + `strategies/`配下を使用
- `data/ohlc/` にYahoo Finance由来の実データを使用
- やがみPDF（`docs/`）がYagami戦略のソースオブトゥルース
- 各戦略の詳細は`docs/strategy_*.md`を参照（コンテキスト分離）
