# CLAUDE.md

sena3fx — 完全自律型トレードPDCAエージェント v3.0（やがみメソッド準拠）

## プロジェクト概要

XAUUSD（金）のティックレベルバックテストとPDCA自動化システム。
やがみメソッド（ローソク足の本・ポジり方の本）の5条件評価を完全実装。

- **対象通貨ペア**: XAUUSD（OANDA）
- **データソース**: Dukascopyティックデータ
- **ブランチ**: `main`
- **リモート**: GitHub `saltstarpx/sena3fx`

## ディレクトリ構成

```
sena3fx/
├── CLAUDE.md               # このファイル
├── .gitignore
├── lib/                    # コアライブラリ
│   ├── __init__.py
│   ├── candle.py           # ローソク足パターン検出
│   ├── patterns.py         # チャートパターン検出
│   ├── levels.py           # レジサポ自動検出
│   ├── timing.py           # 足更新タイミング・MTF・セッション
│   ├── yagami.py           # やがみ5条件統合シグナルエンジン（コア）
│   └── backtest.py         # バックテストエンジン v3.0
├── scripts/
│   ├── fetch_data.py       # Dukascopyティックデータ取得
│   └── main_loop.py        # PDCA自動実行ループ
├── tick_backtest.py        # v1.0エンジン（レガシー）
├── tick_backtest_fast.py   # v2.0エンジン（レガシー、main_loopが参照）
├── data/tick/              # ティックデータCSV（gitignore対象）
├── results/                # バックテスト結果CSV
├── reports/                # PDCAレポートMarkdown
└── docs/                   # やがみPDF等
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

## 開発ワークフロー

### 実行コマンド

```bash
# PDCAサイクル1回実行
python scripts/main_loop.py

# ティックデータ取得（Dukascopy）
python scripts/fetch_data.py

# 統合テスト
python -c "from lib.candle import *; from lib.yagami import *; print('OK')"
```

### 依存パッケージ

```
numpy, pandas, lzma（標準ライブラリ）, struct（標準ライブラリ）
```

`pip install numpy pandas` で十分。

### バックテスト合格基準

| 指標 | 基準 |
|------|------|
| プロフィットファクター | ≥ 1.5 |
| 最大ドローダウン | ≤ 10% |
| 勝率 | ≥ 50%（RR≥2.0なら35%可） |
| トレード数 | ≥ 30 |

### コミット

- 明確な日本語/英語のコミットメッセージ
- 1コミット = 1つの論理的変更
- `__pycache__`やデータファイルはコミットしない

## 注意事項

- `tick_backtest.py`と`tick_backtest_fast.py`はv1/v2レガシー。新規開発は`lib/`配下を使用
- `main_loop.py`がv2互換のため`tick_backtest_fast.py`をインポートしている
- サンプルデータでのバックテスト結果はランダムデータのため参考にならない。実ティックデータが必要
- やがみPDF（`docs/`）が戦略のソースオブトゥルース
