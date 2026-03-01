# Claude Code v14 指示書: 5D HMM + Union+Kellyモニター

## ブランチ運用ルール（必須）
- **Claude Code の作業ブランチ**: `claude/add-trading-backtest-ePJat`
- **Manus AI のブランチ**: `main`
- Claude Code は `main` に直接 push できない（403エラー）
- 完成後は Manus AI が `claude/...` → `main` にマージ

---

## プロジェクト背景

XAUUSD (金先物) CFD戦略の自動バックテスト・PDCA システム。

### 現在の実績 (実トレーダー)
- 累計損益: +17,585,449円 (2025-09〜2026-02)
- 勝率: 70.6% (1,369決済トレード)
- 月次: 9月+60万 / 10月+179万 / 11月+159万 / 12月+729万 / 1月-113万 / 2月+745万

### バックテスト最高成績 (v13)
- Union_4H_Base: Sharpe=2.817, Calmar=13.7, MDD=9.8%
- Kelly(f=0.25): Calmar=24.5 (最高効率)
- MetaStrategy v2.0 (3状態HMM): Sharpe=1.366 (v1.0から+135%改善)

---

## v14 タスク一覧

### Task 1: 5D HMM 特徴量強化 → MetaStrategy v3.0

**背景:**
v2.0の2D HMM (log_return, abs_return) では high_trend=1.7%と識別精度が低い。
ATR/ADX/RSIを追加した5次元HMMでレジーム判定精度を向上する。

**実装対象:**
1. `lib/indicators.py` に `Ind.adx(h,l,c,p=14)` 追加（ADX計算）
2. `lib/regime.py` を5D HMM対応に改修:
   - 観測変数: `[log_return, abs_return, ATR(14), ADX(14), RSI(14)]` — z-score正規化
   - `fit(close, ohlc_df=None)` — ohlc_df指定で5D、Noneで2D（後方互換）
   - `predict(close, ohlc_df=None)` — 同上
   - `feature_stats_by_regime(close, ohlc_df)` 追加
3. `strategies/meta_strategy.py` に v3.0 追加:
   - `make_meta_signal_v3(daily_close, daily_ohlc, ...)` 5D HMMシグナル生成
   - `grid_search_meta_v3()` グリッドサーチ
4. `scripts/backtest_v14.py` 作成・実行:
   - Union_4H_Base (ベースライン) vs MetaStrategy_v2 vs MetaStrategy_v3 比較
   - results/v14_equity_curves.csv + v14_hmm_regimes.csv 保存
   - results/performance_log.csv に追記

**期待成果:**
- high_trend検出率 1.7% → 目標5%以上
- MetaStrategy Calmar > 3.0 (v2.0: 2.425)
- MDD < 25% (v2.0: 21.9%)

---

### Task 2: Union+Kelly リアル稼働モニター

**実装対象:**
1. `monitors/monitor_union_kelly.py`:
   - OANDA v20 REST API で XAU_USD 4H足を取得 (urllib使用)
   - `sig_maedai_yagami_union()` でシグナル評価
   - Kelly乗数 2.4x を適用した最終リスク表示
   - ログ形式:
     ```
     [Signal] 2025-10-15 09:00 JST | XAUUSD | LONG | Price: 2650.40
     [Sizing] Kelly Multiplier: 2.4x | Final Risk: 12.0% of Equity
     ```
   - API未設定時はローカルCSVにフォールバック
   - `--once` オプション (単発実行) / デフォルトはループ監視 (30分ポーリング)
   - 発注は絶対に行わない (監視専用)
2. `deploy/union_kelly_monitor.service` (systemdテンプレート):
   - User/WorkingDirectory/ExecStart をプレースホルダー形式で記述
   - EnvironmentFile対応
   - セキュリティ強化 (NoNewPrivileges, PrivateTmp)

---

### Task 3: ダッシュボード最終更新

**実装対象:** `dashboard.html` に以下を追加:
1. v2.0(2D) vs v3.0(5D) HMM レジーム円グラフ対比 (横並び2つ)
2. MetaStrategy v2 vs v3 Sharpe/Calmar/MDD 比較棒グラフ (`chart-v14-meta`)
3. 5D HMM 特徴量統計テーブル (レジーム別 avg ATR/ADX/RSI)

---

## 実行結果 (達成済み)

### v14 バックテスト結果
| 戦略 | Sharpe | Calmar | MDD% | PF | WR% |
|---|---|---|---|---|---|
| Union_4H_Base | 2.817 | 13.709 | 9.8 | 3.624 | 66.7 |
| MetaStrategy v2.0 (2D HMM) | 1.366 | 2.425 | 21.9 | 1.821 | 53.9 |
| **MetaStrategy v3.0 (5D HMM)** | **1.359** | **3.158** | **14.4** | **1.938** | **55.0** |

### 5D HMM 分布
- range: 45.2% (avg ATR=38.7, ADX=26.6, RSI=64.5)
- low_trend: 48.3% (avg ATR=68.1, ADX=28.1, RSI=60.8)
- **high_trend: 6.6%** (avg ATR=172.3, ADX=29.9, RSI=55.5) ← 1.7%→6.6%改善

### 評価
- Calmar 3.158 → 目標 3.0 達成 ✅
- MDD 14.4% → 目標 30%以下 達成 ✅
- high_trend検出 6.6% → 目標 5%以上 達成 ✅

---

## 完了後のコミット・プッシュ

```bash
git add -A
git commit -m "v14: 5D HMM MetaStrategy v3.0 + Union+Kelly monitor"
git push -u origin claude/add-trading-backtest-ePJat
```

その後、Manus AI が main にマージ。
