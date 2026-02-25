# sena3fx

XAUUSD自動トレードPDCAエージェント v3.0 — やがみメソッド準拠

## 概要

やがみメソッド（ローソク足5条件評価）に基づく完全自律型バックテスト＆PDCA自動化システム。

## フォルダ構成

| フォルダ | 内容 |
|---------|------|
| `lib/` | コアエンジン（ローソク足・チャートパターン・レジサポ・やがみシグナル・バックテスト） |
| `scripts/` | 実行スクリプト（PDCA実行・データ取得） |
| `docs/` | 参考資料（やがみPDF・要件定義） |
| `archive/` | 旧バージョン（v1/v2コード・旧レポート） |
| `data/` | ティックデータ |
| `results/` | バックテスト結果 |
| `reports/` | PDCAレポート |

## 使い方

```bash
# PDCAサイクル実行
python scripts/main_loop.py

# ティックデータ取得
python scripts/fetch_data.py
```

## 依存パッケージ

```bash
pip install numpy pandas
```
