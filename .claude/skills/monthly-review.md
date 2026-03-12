---
name: "monthly-review"
description: "月次PDCAレビューを実行。本番取引ログをバックテスト基準値と比較してドリフトを検出する。"
tags: [review, pdca, trading, analysis]
trigger: "月次レビュー"
---

# YAGAMI改 月次レビュースキル

## 概要
毎月末に本番取引結果とバックテスト基準値を比較し、戦略ドリフトを早期検出する。

## 実行コマンド

```bash
cd /home/user/sena3fx

# 特定月のレビュー
python scripts/monthly_review.py --month 2026-03 --csv production/trade_logs/paper_trades.csv

# 全期間レビュー
python scripts/monthly_review.py --all --csv production/trade_logs/paper_trades.csv
```

結果: `results/monthly_review_YYYY-MM.png` と `.txt` に保存

## ドリフトアラート基準
| アラート | 条件 | 対応 |
|---|---|---|
| WR乖離 | \|本番WR - 期待WR\| > 8pp | 取引ログを個別確認 |
| PF乖離 | 本番PF / 期待PF < 0.7 | バックテスト再実行 |

## バックテスト基準値（results/backtest_baseline.json）
| 銘柄 | 期待PF | 期待WR |
|---|---|---|
| XAUUSD | 3.44 | 73.1% |
| SPX500 | 2.03 | 69.8% |
| GBPUSD | 2.29 | 69.4% |
| AUDUSD | 2.19 | 64.8% |
| NZDUSD | 1.78 | 62.3% |

## レビュー手順（PDCA）
1. **Check**: monthly_review.py を実行してドリフト確認
2. **ドリフトあり（WR-8pp超 / PF比<0.7）**:
   - 時間帯別勝率チャートでエントリー時間帯の偏りを確認
   - Long/Short方向の偏りを確認
   - 直近のマーケット環境（トレンド vs レンジ）を確認
3. **Action決定**:
   - 軽微ドリフト: 1〜2ヶ月様子見
   - 中程度ドリフト: バックテスト再実行 → パラメータ見直し
   - 重大ドリフト（PF比<0.5）: 該当銘柄のポジションサイズを半減
4. **改善実装 → 翌月確認**

## 月次チェックリスト
- [ ] monthly_review.py 実行
- [ ] ドリフトアラートの有無確認
- [ ] 銘柄別PF/WR確認
- [ ] UTC時間帯別勝率チャート確認
- [ ] Long/Short比率確認
- [ ] 来月の銘柄別リスク設定見直し（必要な場合）

## ロールバック手順（データ不足時）
- 取引ログが空 → `production/trade_logs/paper_trades.csv` の存在確認
- 銘柄が表示されない → `entry_time` カラムの日付形式確認（ISO8601）
- グラフ生成失敗 → `pip install matplotlib` を確認
