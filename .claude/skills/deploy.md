---
name: "deploy"
description: "YAGAMI改をExness VPS（MT5ボット）またはGCP Cloud Runにデプロイする。"
tags: [deploy, production, mt5, cloud-run]
trigger: "デプロイ"
---

# YAGAMI改 デプロイスキル

## デプロイ先は2系統

| 系統 | 対象 | 場所 |
|---|---|---|
| **MT5ボット** | Exness VPS（本番リアル取引） | `production/mt5_bot.py` |
| **Cloud Run** | GCP（旧OANDAデモ、参照用） | `cloud_run/main.py` |

---

## MT5ボット（Exness VPS）デプロイ手順

### Step 1: .env を設定（初回のみ）
```
production/.env.example を .env にコピーして以下を入力:
  MT5_LOGIN=口座番号
  MT5_PASSWORD=パスワード
  MT5_SERVER=Exness-MT5Real8
  DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
  DRY_RUN=false
```

### Step 2: 依存ライブラリインストール（初回のみ）
```powershell
pip install -r production\requirements.txt
```

### Step 3: MT5銘柄名確認・調整
MT5ターミナルの気配値表示で銘柄名を確認。
`production/mt5_bot.py` の SYMBOLS 辞書の `mt5_sym` を合わせる:
```python
"XAUUSD": {"mt5_sym": "XAUUSDm", ...}  # 実際のMT5銘柄名に変更
```

### Step 4: DRY_RUNで動作確認
```powershell
# .env の DRY_RUN=true にしてから:
python production\mt5_bot.py
# ログにシグナル検出・ロット計算が表示されればOK
```

### Step 5: 本番起動
```powershell
# DRY_RUN=false に変更後:
python production\mt5_bot.py
# バックグラウンド起動:
Start-Process python -ArgumentList "production\mt5_bot.py" -WindowStyle Hidden
```

### ロールバック手順（トラブル時）
- MT5接続失敗 → MT5ターミナルが起動・ログイン済みか確認
- 銘柄名エラー → MT5の気配値表示で正確な銘柄名を確認
- 注文拒否 → 証拠金不足 / 最小ロット未満 / 市場クローズを確認
- クラッシュ後の再起動 → `production/trade_logs/bot_state.json` から状態が自動復元される

---

## Cloud Run デプロイ手順（参照用）

```bash
cd /home/user/sena3fx/cloud_run
gcloud run deploy yagami-bot \
  --source . \
  --region asia-northeast1 \
  --set-env-vars "OANDA_TOKEN=xxx,DISCORD_WEBHOOK=yyy"
```

## 注意事項
- `strategies/current/` と `cloud_run/strategies/` は常に同一内容を保つ
- `gcp-key.json` は絶対にコミットしない（.gitignore対象）
- 本番起動前に必ずDRY_RUNで動作確認すること
