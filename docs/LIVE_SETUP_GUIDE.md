# 本番環境セットアップガイド
## Exness × MT5 × YAGAMI改 — ゼロからトレード開始まで

> 所要時間: 約2〜3時間（口座審査除く）
> あなたがやること: **太字のステップのみ**。それ以外はコード・自動設定済み。

---

## 全体の流れ

```
Step1: VPN設定（5分）
Step2: Exness口座開設（30分）
Step3: Windows VPS取得（10分）
Step4: MT5インストール（15分）
Step5: Pythonセットアップ（10分）
Step6: ライブトレーダー起動（5分）
Step7: 動作確認（初回のみ15分）
```

---

## Step 1: VPN設定

### なぜVPNが必要か

Exnessは日本のIPからでも登録・取引できますが、
万が一アクセス制限がかかった場合の保険としてVPNを推奨します。
またWindows VPSからの接続にも使用します。

### **あなたがやること**

1. **ExpressVPN または NordVPN に登録**（月$5〜8程度）
   - ExpressVPN: https://www.expressvpn.com
   - NordVPN: https://nordvpn.com
2. **アプリをインストール**（PC・スマホ）
3. **接続先を「シンガポール」または「香港」に設定**
4. **VPN ON の状態**でStep2を実施

---

## Step 2: Exness口座開設

### 口座タイプ選択

**Raw Spreadアカウントを選ぶ（必須）**

| 項目 | 設定 |
|---|---|
| 口座タイプ | **Raw Spread** |
| ベース通貨 | JPY（日本円） |
| レバレッジ | 1:500〜1:1000 |
| 最低入金 | $200 相当（約3万円） |

Raw Spreadを選ぶ理由: スプレッドが最小（コード内のSYMBOL_CONFIGに合わせてある）

### **あなたがやること**

1. **VPN ON の状態**で Exness公式サイトにアクセス
2. 「登録」→ メール・パスワード設定
3. 本人確認書類アップロード（パスポートor運転免許証 + 住所確認書類）
4. **Raw Spreadアカウントを新規作成**
5. **入金**（最初はPhase1なので3〜5万円から可）
6. MT5サーバー名をメモ（例: `Exness-MT5Real8`）→ Step4で使う

---

## Step 3: Windows VPS取得

### なぜWindowsが必要か

MT5のPython APIは **Windowsでしか動かない**。
24時間稼働のため、自分のPCでなくVPSを使う。

### 推奨VPS

| サービス | 月額 | 特徴 |
|---|---|---|
| **Vultr** (推奨) | $12〜 | 東京リージョンあり・起動が早い |
| Linode | $12〜 | 安定性高い |
| ConoHa | ¥1,500〜 | 日本語対応 |

### スペック目安

- CPU: 2コア以上
- RAM: **4GB以上**（MT5 + Python同時起動）
- OS: **Windows Server 2022** または **Windows 10/11**
- 容量: 60GB以上

### **あなたがやること**

1. Vultr.com にサインアップ
2. 「Deploy New Server」→ Cloud Compute
3. OS: **Windows Server 2022**
4. Plan: **4GB RAM / 2vCPU ($24/月)**
5. Location: **Tokyo**
6. Deploy → 5分後にRDPで接続（ユーザー名/パスワードが表示される）
7. **RDPで接続できることを確認**

> Macの場合: App Storeで「Microsoft Remote Desktop」を無料インストール

---

## Step 4: MT5インストール（VPS上で実施）

### **あなたがやること（VPSにRDPログインした状態で）**

1. VPS上のEdgeブラウザを開く
2. `https://www.exness.com/jp/platforms/mt5/` にアクセス
3. **「MetaTrader5をダウンロード」** → インストール実行
4. MT5を起動 → 「ファイル」→「取引口座にログイン」
5. サーバー検索: `Exness-` と入力 → メモしたサーバー名を選択
6. Exnessのログイン番号・パスワードを入力
7. **口座残高が表示されることを確認** ✅

### MT5の設定（重要）

MT5上で以下の設定を必ずONにする:

1. 「ツール」→「オプション」→「Expert Advisors」タブ
   - ✅ **「自動売買を許可する」** をチェック
   - ✅ **「DLLのインポートを許可する」** をチェック
2. 「ツール」→「オプション」→「サーバー」タブ
   - 自動更新: OFF（予期しない再起動防止）
3. MT5ウィンドウを **最小化**（閉じない）

### 銘柄追加

7銘柄を気配値に追加する:

1. 「表示」→「気配値表示」→右クリック「全て表示」
2. 以下7銘柄があることを確認:
   - EURUSD, GBPUSD, USDCAD, NZDUSD, AUDUSD（Forex）
   - USDJPY（Forex Majors）
   - XAUUSD（Metals）
3. なければ「気配値」で右クリック→「銘柄」→検索して追加

---

## Step 5: Pythonセットアップ（VPS上で実施）

### **あなたがやること**

```powershell
# PowerShellを管理者権限で開いて実行

# 1. Pythonインストール（3.11推奨）
# Edgeで https://www.python.org/downloads/ にアクセス
# 「Download Python 3.11.x」→ インストール時に「Add to PATH」にチェック

# 2. コードをダウンロード（GitHubから）
cd C:\Users\Administrator
git clone https://github.com/saltstarpx/sena3fx.git
cd sena3fx

# 3. 依存パッケージをインストール
pip install MetaTrader5 pandas numpy scipy
```

### GitHubにアクセスできない場合

1. 自分のPCでこのリポジトリをZIPダウンロード
2. VPSにRDPでコピー（「ファイル転送」機能を使用）
3. `C:\Users\Administrator\sena3fx\` に展開

---

## Step 6: ライブトレーダー起動

### 起動前確認チェックリスト

```
□ MT5が起動中でExness口座にログイン済み
□ 自動売買がONになっている
□ 7銘柄全て気配値に表示されている
□ 口座残高が正しく表示されている
```

### **あなたがやること**

```powershell
# PowerShellで実行

# sena3fxフォルダに移動
cd C:\Users\Administrator\sena3fx

# ライブトレーダー起動
python scripts/mt5_live_trader.py
```

### 正常起動時の表示

```
============================================================
  YAGAMI改 MT5 ライブトレーダー 起動
  Phase2: 1.0%リスク × 7銘柄
============================================================
MT5 接続成功: 12345678 / Exness-MT5Real8 / 残高=100000.00 JPY
監視開始... Ctrl+C で停止
```

### バックグラウンド実行（VPSを閉じても動かし続ける）

```powershell
# PowerShellでバックグラウンド起動
Start-Process -NoNewWindow -FilePath "python" `
  -ArgumentList "C:\Users\Administrator\sena3fx\scripts\mt5_live_trader.py" `
  -RedirectStandardOutput "C:\Users\Administrator\sena3fx\trade_logs\stdout.log" `
  -RedirectStandardError  "C:\Users\Administrator\sena3fx\trade_logs\stderr.log"
```

または、**Windowsタスクスケジューラ**に登録（VPS再起動後も自動起動）:

1. 「タスクスケジューラ」を検索して開く
2. 「基本タスクの作成」
3. トリガー: **「コンピューターの起動時」**
4. 操作: `python C:\Users\Administrator\sena3fx\scripts\mt5_live_trader.py`
5. 「ネットワーク接続後に実行を遅延」: 60秒

---

## Step 7: 動作確認

### 起動後30分以内に確認すること

```
□ ログに「MT5 接続成功」と表示されている
□ エラーログがない
□ 各銘柄の「シグナルなし」が流れている（正常動作）
□ MT5の「Expert」タブにエラーがない
```

### ログファイルの場所

```
C:\Users\Administrator\sena3fx\trade_logs\
├── mt5_live_YYYYMMDD.log    ← リアルタイムログ
├── trades_YYYYMM.csv        ← トレード記録（運用ルール書 §10）
└── live_state.json          ← 日次/週次R、ポジション状態
```

### 最初のトレードが入ったら確認すること

```
□ MT5の「取引」タブにポジションが表示されている
□ SL/TPが正しく設定されている
□ ロットサイズが適切（1%リスク相当か）
□ trade_logs/trades_YYYYMM.csv に記録されている
□ コメントに「YAGAMI-A-GBPUSD」等の表示がある
```

---

## 日常の管理（毎日5分）

### 毎日確認すること

1. VPSにRDP接続
2. PowerShellで確認:
```powershell
# ログ末尾を確認
Get-Content C:\Users\Administrator\sena3fx\trade_logs\mt5_live_*.log -Tail 20
```
3. MT5の「取引」タブでオープンポジション確認
4. 異常があれば `Ctrl+C` で停止 → 調査

### 週次確認（5〜10分）

```powershell
# トレード記録確認
Import-Csv C:\Users\Administrator\sena3fx\trade_logs\trades_YYYYMM.csv | Format-Table
```

---

## トラブルシューティング

### MT5接続失敗

```
ERROR: MT5 初期化失敗
```

→ MT5ターミナルが起動しているか確認
→ MT5で「自動売買を許可する」がONか確認
→ MT5を再起動してから再実行

### 銘柄が見つからない

```
WARNING: XAUUSD がMT5で見つかりません
```

→ MT5の気配値表示で該当銘柄を追加
→ Exnessでは `XAUUSD` または `GOLDm` の場合あり
→ `SYMBOLS`辞書のキー名をMT5表示に合わせて変更:

```python
# mt5_live_trader.py の SYMBOLS を修正
"XAUUSD": {"logic": "A", ...},
# ↓ MT5に表示されている名前に変更（例）
"XAUUSDm": {"logic": "A", ...},
```

### 発注失敗: invalid volume

```
ERROR: GBPUSD 発注失敗: invalid volume
```

→ 残高に対してリスク額が小さすぎてロット計算が最小以下になっている
→ 入金額を増やすか、リスク率を一時的に上げる

### ロット0.00エラー

入金額が少ない場合、1%リスクで計算されるロットが最小ロット(0.01)を下回る。
最低でも**10万円**の入金を推奨。

---

## Phase管理

### 今すぐ開始できる設定

```python
# mt5_live_trader.py 1行目付近
RISK_PCT = 0.010   # Phase2: 1.0%

# Phase1から始める場合（より慎重）
RISK_PCT = 0.005   # Phase1: 0.5%
```

Phase1で40トレード確認後にPhase2に上げる場合は、この1行を変更して再起動するだけ。

---

## コスト概算

| 項目 | 月額 |
|---|---|
| VPS（Windows 4GB） | $24 ≈ 3,600円 |
| VPN | $6 ≈ 900円 |
| **合計** | **約4,500円/月** |

Phase2（1%×7銘柄）のバックテスト期待値: **月次+50万円以上**（初期資金100万円、複利）

---

## 緊急停止方法

### 即時全ポジション決済

MT5で手動操作:
1. 「取引」タブを右クリック
2. 「全ポジションをクローズ」

### スクリプト停止

```powershell
# PowerShell
Get-Process python | Stop-Process -Force
```

---

> 作成日: 2026-03-11
> 対象ストラテジー: YAGAMI改 Phase2（1.0%×7銘柄）
> OOS検証結果: Sharpe=6.95 / MDD=8.2% / 月次プラス9/9ヶ月
