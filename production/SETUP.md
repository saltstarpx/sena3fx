# YAGAMI改 本番セットアップガイド（Exness VPS）

## 前提条件

- Exness リアル口座（ロースプレッド口座 推奨）
- Exness 無料VPS（口座残高 $500以上で利用可）
- MetaTrader5 ターミナル（VPSにインストール済み）
- Python 3.10以上（VPSにインストール）

---

## ステップ1: Exness VPSにMT5をセットアップ

1. Exness マイページ → 「VPS管理」から VPS を有効化
2. VPSにリモートデスクトップ接続
3. MT5ターミナルをダウンロード・インストール
4. MT5 → `ファイル` → `サーバーに接続` → Exnessアカウントでログイン
5. MT5ターミナルを **最小化**（バックグラウンドで起動したまま）

---

## ステップ2: Python環境のセットアップ

```powershell
# PowerShellで実行
python --version   # 3.10以上を確認

# このリポジトリをVPSにコピー（GitかZIPで）
cd C:\yagami

# 依存ライブラリをインストール
pip install -r production\requirements.txt
```

---

## ステップ3: .envを設定

```powershell
# production フォルダに .env を作成
copy production\.env.example production\.env

# メモ帳で編集
notepad production\.env
```

入力する内容:
```
MT5_LOGIN=あなたのMT5口座番号
MT5_PASSWORD=あなたのMT5パスワード
MT5_SERVER=Exness-MT5Real8
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
BASE_RISK_PCT=0.02
MAX_POSITIONS=3
DRY_RUN=false
```

---

## ステップ4: MT5銘柄名の確認と調整

ExnessのMT5では銘柄名に末尾が付く場合があります（例: `XAUUSDm`）。
MT5ターミナルの「気配値表示」で確認してください。

```python
# mt5_bot.py の SYMBOLS 辞書で mt5_sym を調整
SYMBOLS = {
    "XAUUSD": {"mt5_sym": "XAUUSDm", ...},  # ← 確認した名前に変更
    "GBPUSD": {"mt5_sym": "GBPUSDm", ...},
    # ...
}
```

よくある銘柄名パターン:
| 銘柄 | ロースプレッド | ゼロ | スタンダード |
|------|------|------|------|
| XAUUSD | XAUUSDm | XAUUSD | XAUUSDm |
| GBPUSD | GBPUSDm | GBPUSD | GBPUSDm |
| SPX500 | SP500m | SP500 | — |

---

## ステップ5: ドライランで動作確認

```powershell
# DRY_RUN=true に設定した状態でテスト
# production\.env の DRY_RUN=true にしてから:
cd C:\yagami
python production\mt5_bot.py
```

ログ出力を確認し、シグナル検出・ロット計算が正常に動いていればOK。

---

## ステップ6: 本番起動

```powershell
# DRY_RUN=false に変更
# production\.env の DRY_RUN=false

# タスクスケジューラ or コマンドプロンプトで起動
python production\mt5_bot.py
```

VPSを閉じてもバックグラウンドで動かし続けるには:
```powershell
# バックグラウンド起動（PowerShell）
Start-Process python -ArgumentList "production\mt5_bot.py" -WindowStyle Hidden
```

---

## 月次レビュー

毎月末に `scripts/monthly_review.py` で本番とバックテストを比較:

```powershell
python scripts\monthly_review.py --month 2026-03 --csv production\trade_logs\paper_trades.csv
```

結果: `results/monthly_review_YYYY-MM.png` に保存されます。

---

## 採用銘柄・リスク設定（2026/3/9 確定）

| 銘柄 | OOS PF | 基本リスク | 備考 |
|------|:---:|:---:|------|
| XAUUSD | 3.44 | 2% | Gold Logic 主力 |
| GBPUSD | 2.29 | 2% | Gold Logic |
| AUDUSD | 2.19 | 2% | Gold Logic |
| NZDUSD | 1.78 | 1% | 様子見（半リスク）|
| SPX500 | 2.03 | 2% | Gold Logic |

AdaptiveRiskManager による自動調整:
- DD ≥ 5% → リスク×0.75
- DD ≥ 10% → リスク×0.50
- DD ≥ 15% → リスク×0.25（最小）
