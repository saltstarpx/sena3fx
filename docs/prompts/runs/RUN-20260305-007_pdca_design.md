# RUN-20260305-007 以降 PDCA設計文書

**作成日**: 2026-03-05  
**作成者**: Manus AI  
**参照EntryID**: 20260305-006（Claude Code Walk-Forward結果）  
**参照RunID**: RUN-20260305-001〜006

---

## 1. これまでの学習成果（RUN-001〜006）

| RunID | 内容 | 結論 |
|-------|------|------|
| RUN-001 | やがみPA基本バックテスト | PA1_Reversal_TightSL のみPF>1（損益分岐点） |
| RUN-002 | 4H ゾーンフィルター | 仮説棄却（フィルターと方向が不整合） |
| RUN-003 | 方向A/B並行 | ボリューム確認のみ有効、EMAトレンドは再設計必要 |
| RUN-004 | P0修正（C1切替） | Yagami_A PF=2.012（N=7）→ 統計的信頼性低 |
| RUN-005 | Walk-Forward検証 | **Yagami_A 偶然と判定**（OOS合格1/11、OOS平均PF=0.47） |
| RUN-006 | Maedai DC30+EMA200 | **DC50+EMA200 PF=3.761、WR=76.5%**（7/9戦略合格） |

### 重要な結論

> **やがみメソッドは「シグナル生成器」として機能しない。Maedai系（DC+EMA200）が真のアルファ源泉である。**

この結論はClaude Codeのハンドオフ文書（P0-1の独自意見）と完全に一致し、RUN-006の定量データで裏付けられた。

---

## 2. RUN-007: Maedai DC50+EMA200 Walk-Forward検証

### 目的
RUN-006でPF=3.761、WR=76.5%を記録したDC50+EMA200（XAUUSD D1）が、Walk-Forward検証でも有意性を示すか確認する。

### 設計

| 項目 | 設定値 |
|------|--------|
| 対象戦略 | DC50+EMA200（XAUUSD D1） |
| データ | XAUUSD_1d.csv（全期間） |
| 訓練期間 | 12ヶ月 |
| 検証期間 | 3ヶ月 |
| fold数 | 推定8〜10 |
| 合格基準 | OOS PF≥1.5、WR≥50%、MDD≤15% |

### 仮説
- H0: DC50+EMA200のOOS平均PF < 1.5（帰無仮説）
- H1: DC50+EMA200のOOS平均PF ≥ 1.5（採択目標）

### 期待値（RUN-006結果から）
- IS PF=3.761 → OOS劣化30%を見込んでも OOS PF≈2.6
- WR=76.5% → OOS劣化10pt見込みでも WR≈66%

---

## 3. RUN-008: Maedai DC50+EMA200 マルチ銘柄ポートフォリオ

### 目的
RUN-006で合格した7戦略を組み合わせ、ポートフォリオとして相関・分散効果を検証する。

### 設計

| 戦略 | PF | WR% | MDD% | 相関グループ |
|------|----|-----|------|------------|
| DC50_EMA200_XAUUSD_D1 | 3.761 | 76.5 | 4.0 | 金属 |
| DC30_EMA200_XAUUSD_D1 | 3.144 | 69.2 | 6.3 | 金属 |
| DC30_EMA200_USDJPY_D1 | 2.367 | 57.9 | 1.8 | JPY |
| DC30_EMA200_confirm_XAUUSD | 2.274 | 66.7 | 9.6 | 金属（確認付き） |
| DC30_EMA100_XAUUSD_D1 | 2.163 | 63.0 | 6.3 | 金属 |
| DC20_EMA200_XAUUSD_D1 | 2.025 | 60.9 | 9.9 | 金属 |
| DC30_EMA200_EURUSD_D1 | 1.643 | 47.4 | 0.0 | EUR |

**注意**: XAUUSD系5戦略は高相関のため、ポートフォリオ分散効果は限定的。USDJPY・EURUSDとの組み合わせが重要。

---

## 4. RUN-009: HMMレジームフィルターの実装

### 目的
Claude Codeのハンドオフ文書で「未実施」と指摘されたHidden Markov Modelによるレジーム判別フィルターを実装し、DC50+EMA200に重ねる。

### 設計
- 状態数: 2（トレンド / レンジ）
- 特徴量: 日次リターン、ATR比率、ボリューム変化率
- フィルター: トレンド状態のみエントリー許可
- 期待効果: MDD削減（現状4.0% → 目標2.0%以下）

---

## 5. RUN-010: OANDA Practice環境でのフォワードテスト

### 目的
DC50+EMA200（Walk-Forward合格後）をOANDA Practice環境で実際に稼働させ、バックテストとの乖離を計測する。

### 前提条件
- RUN-007でWalk-Forward合格（OOS PF≥1.5）
- `resolve_oanda_credentials()` 実装済み（RUN-006並行作業で完了）
- OANDA_API_KEY / OANDA_ACCOUNT_ID が `.env` に設定済み

### 計測指標
- スリッページ（バックテスト想定 vs 実際）
- 約定率（成行注文の充足率）
- 日次PnLのバックテストとの相関

---

## 6. 優先順位マトリクス

| RunID | 優先度 | 依存関係 | 担当推奨 |
|-------|--------|---------|---------|
| RUN-007 | **最高** | RUN-006完了 | Claude Code |
| RUN-008 | 高 | RUN-007完了後 | Manus |
| RUN-009 | 中 | RUN-007完了後 | Claude Code |
| RUN-010 | 高 | RUN-007合格後 | Manus |

---

## 7. Claude Code向け RUN-007 依頼プロンプト

以下をそのままClaude Codeに渡すこと。

```
## タスク: RUN-20260305-007 Maedai DC50+EMA200 Walk-Forward検証

### 背景
- RUN-006でDC50+EMA200（XAUUSD D1）がPF=3.761、WR=76.5%を記録
- RUN-005でYagami_Aは偶然と判定（OOS平均PF=0.47）
- DC50+EMA200の統計的有意性をWalk-Forwardで確認する

### 実装手順
1. `data/ohlc/XAUUSD_1d.csv` を使用
2. `lib/yagami.py` の `sig_maedai_d1_dc30(50, 200)` を使用
3. 訓練12ヶ月/検証3ヶ月のウォークフォワード（RUN-005の `scripts/walk_forward_run005.py` を参考に実装）
4. 合格基準: OOS PF≥1.5、WR≥50%、MDD≤15%
5. 結果を `results/walk_forward_run007.csv` に保存
6. `results/run007_wf_report.md` にサマリーを記述
7. `docs/logbook.md` に EntryID `20260305-007` として追記
8. ブランチ `claude/run007-maedai-wf` でPR作成

### 注意事項
- BacktestEngineのキー名: `win_rate_pct`（%単位）、`max_drawdown_pct`（%単位）
- `tick_count` KeyErrorは修正済み（`lib/backtest.py`）
- コンフリクト防止のためPRで提出すること（mainへの直接pushは禁止）
```
