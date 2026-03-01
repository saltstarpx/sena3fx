# Manus AI 変更履歴

このファイルは、Manus AIがリポジトリに対して行った変更を記録するログです。
Claude Codeとの連携において、何が変更されたかを把握するために使用してください。

---

## [Manus AI への重要なお願い] ブランチのマージ

Claude Codeはセキュリティ制約により `main` ブランチへ直接プッシュできません。
すべての開発成果は以下のブランチに蓄積されています:

**`claude/add-trading-backtest-ePJat`**

このブランチには `main` にない以下が含まれます:
- `strategies/` (Union戦略, Yagami, Maedai, Market Filters)
- `monitors/` (フォワードテスト監視)
- `dashboard.html` (戦略パフォーマンスダッシュボード)
- `results/performance_log.csv` (バックテスト自動ログ)
- `trade_logs/broker_history_*.csv` (実取引履歴, UTF-8変換済)
- `lib/backtest.py` (Sharpe/Calmar/richmanbtc p-mean法 追加)
- `docs/strategy_*.md` (全戦略ドキュメント)
- `README.md` (v4.0対応の最新版)

**Manus AIの作業前に、このブランチを `main` にマージしてください:**
```
git checkout main
git merge claude/add-trading-backtest-ePJat
git push origin main
```

---

## [2026-03-01] v18: メイン戦略確定 + フォワードモニター刷新 + ダッシュボード更新

**変更者:** Claude Code
**変更種別:** 最終戦略定義・監視体制整備・実験スクリプト削除
**指示書:** `prompt_for_claude_code_v18.md`

### 新設ファイル
- **`strategies/main_strategy.py`** — 最終確定戦略の定義 (XAUUSD+Kelly(f=0.25)+ADX(14)>25)
  - `make_signal_func()`: ADXフィルター付きシグナル関数を返す
  - `run_backtest()`: バックテスト実行 (CLIおよびモジュール利用可)
  - `get_latest_signal()`: 最新バーのシグナル取得
  - `print_results()`: 整形結果出力
  - CLIから直接実行可能: `python strategies/main_strategy.py`
- **`monitors/forward_main_strategy.py`** — フォワードテスト・シグナルモニター (v18刷新)
  - `monitors/monitor_union_kelly.py` から置き換え
  - `trade_logs/forward_signals.csv` にシグナルを自動記録 (datetime_utc, symbol, direction, price, adx)
  - ADX値をログに含める (フィルター判定の透明化)

### 削除ファイル (実験スクリプト)
- ~~`scripts/grid_search_pyramid.py`~~ — Livermore GS (v17完了、不採用確定)
- ~~`scripts/backtest_xau_final.py`~~ — Kelly+Pyramid統合テスト (v16完了)
- ~~`scripts/backtest_xau_adx_filter.py`~~ — ADXフィルター検証 (v17完了、本番昇格済)

### 修正ファイル
- **`dashboard.html`** — メイン戦略セクション追加
  - `XAUUSD+Kelly+ADX(>25)` を最上部に**最優秀戦略**として明示表示
  - KPIカード: Sharpe=2.250, Calmar=11.534, MDD=16.1%, WR=61.0%, 最終資産¥27,175,560
  - 比較チャート `chart-main-strategy`: Union_Base / +Kelly / +Pyramid / +ADX の4戦略比較

### メイン戦略 確定パラメータ

| 項目 | 値 |
|---|---|
| 銘柄 | XAUUSD |
| 時間足 | 4H |
| シグナル | Union_4H_Base (sig_maedai_yagami_union) |
| フィルター | ADX(14) > 25 |
| サイジング | KellyCriterionSizer(f=0.25) |
| Kelly乗数 | 1.13x |
| SL/TP | ATR x2.0 / ATR x4.0 |

### 確定バックテスト結果 (2023-10-06 〜 2026-02-27)

| 指標 | 値 | 目標 | 判定 |
|---|---|---|---|
| Sharpe Ratio | 2.250 | > 1.5 | ✅ |
| Calmar Ratio | 11.534 | > 5.0 | ✅ |
| Max Drawdown | 16.1% | ≤ 30% | ✅ |
| Win Rate | 61.0% | ≥ 30% | ✅ |
| Total Trades | 41 | ≥ 5 | ✅ |
| 最終資産 | ¥27,175,560 | — | +443.5% |

---

## [2026-03-01] v17: ピラミッドGS + ADXフィルター最終判定 + 承認リスト厳格化

**変更者:** Claude Code
**変更種別:** パラメータ最適化 + 重要発見 (XAUUSDでADXフィルター有効)
**指示書:** `prompt_for_claude_code_v17.md`

### 新設ファイル
- **`scripts/grid_search_pyramid.py`** — LivermorePyramidingSizer グリッドサーチ (6パターン)
- **`scripts/backtest_xau_adx_filter.py`** — XAUUSD+Kelly ADXフィルター検証
- **`results/pyramid_grid_search.csv`** — ピラミッドGS全結果

### 修正ファイル
- **`lib/approved_list.py`** v2.0 — 承認条件を `Sharpe>1.0 AND Trades>20` から **`Sharpe>1.5 AND Calmar>5.0`** に厳格化
  - XAUUSD のみ承認 (Sharpe=1.718, Calmar=5.971 ✅)
  - XAGUSD は非承認 (Calmar=2.111 < 5.0 ❌)
  - `--v1` オプションで旧条件での実行も可能 (後方互換)

### バックテスト結果: Task 1 ピラミッドGS (XAUUSD+Kelly base)

ベースライン: Union+Kelly(f=0.25): Sharpe=1.717, MDD=26.3%, ¥25,701,780

| step_pct | max_pyramids | Sharpe | Calmar | MDD% | Trades | 最終資産 | pyramid% |
|---|---|---|---|---|---|---|---|
| 2% | 1 | 1.418 | 4.138 | 34.2 | 88 | ¥21,841,531 | 42.0% |
| 2% | 2 | 1.425 | 3.566 | 39.6 | 88 | ¥21,804,829 | 40.9% |
| 3% | 1 | 1.520 | 4.350 | 35.6 | 84 | ¥23,450,220 | 28.6% |
| 3% | 2 | 1.526 | 4.399 | 35.6 | 84 | ¥23,658,682 | 28.6% |
| **4%** | **1** | **1.636** | **6.191** | **26.0** | **74** | **¥24,190,041** | **24.3%** |
| 4% | 2 | 1.635 | 6.173 | 26.0 | 74 | ¥24,133,580 | 24.3% |

**結論: 全パターンがベースライン (Sharpe=1.717) を下回る → Livermore Pyramidingは不採用**
- step_pct が大きくなるほど良化 (2%→3%→4%) — 頻繁なピラミッドがノイズを生む
- max_pyramids=1 vs 2 の差が小さい → 追加回数より発動条件が重要

### バックテスト結果: Task 2 ADXフィルター — **🔑 重要発見**

ADX(14) 統計 (XAUUSD 4H): mean=27.7, median=26.1, ADX>25 = 53.6%

| 戦略 | Sharpe | Calmar | MDD% | WR% | Trades | 最終資産 |
|---|---|---|---|---|---|---|
| XAUUSD+Kelly(f=0.25) | 1.717 | 6.574 | 26.3 | 50.0 | 70 | ¥25,701,780 |
| **XAUUSD+Kelly+ADX(>25)** | **2.250** | **11.534** | **16.1** | **61.0** | **41** | **¥27,175,560** |

**ADXフィルター効果 (XAUUSD):**
- Sharpe: 1.717 → **2.250 (+0.533 / +31%)**
- MDD: 26.3% → **16.1% (-10.2pt)**
- Calmar: 6.574 → **11.534 (+75%)**
- WR: 50.0% → **61.0% (+11pt)** — 高品質シグナルのみ抽出に成功

**XAGUSD (v15) vs XAUUSD (v17) のADXフィルター対比:**

| 商品 | フィルターなし Sharpe | +ADX(>25) Sharpe | 効果 |
|---|---|---|---|
| XAGUSD (v15) | 1.189 | 1.008 | **-0.181 ❌ 非推奨** |
| **XAUUSD (v17)** | **1.717** | **2.250** | **+0.533 ✅ 有効** |

商品特性による差異: XAUUSDはADX平均27.7でトレンド性が高く、ADXフィルターとUnionの相性が抜群。

### 考察・知見
- **ADXフィルター+XAUUSD = ✅ 次期最優先戦略候補**
  - Sharpe 2.250 > 実トレーダー目標 / Calmar 11.534 (圧倒的安定性)
  - MDD 16.1% — 30%目標を大幅クリア
  - 次期 v18 では `XAUUSD+Kelly+ADX(>25)` をメイン戦略に昇格させることを推奨
- **Livermore Pyramiding は不採用確定**
  - すべてのGSパターンがKelly単独を下回る
  - 将来的に改善する場合は step_pct=5%以上 + max_pyramids=1 を検討
- **承認リスト v2.0**: XAUUSD のみ承認 — 単品集中戦略が最効率

---

## [2026-03-01] v16: XAUUSDメイン化 + リバモア式ピラミッティング

**変更者:** Claude Code
**変更種別:** Kelly サイジング適用 + LivermorePyramidingSizer 実装・検証
**指示書:** `prompt_for_claude_code_v16.md`

### 新設ファイル
- **`scripts/backtest_xau_kelly.py`** — XAUUSD + KellyCriterionSizer(f=0.25) バックテスト
- **`scripts/backtest_xau_final.py`** — Union + Kelly + LivermorePyramiding 統合バックテスト

### 修正ファイル
- **`lib/risk_manager.py`** — `LivermorePyramidingSizer` 追加 (v3.0)
  - `base_sizer`: 初期エントリーを担当する Sizer (KellyCriterionSizer 等)
  - `step_pct`: ピラミッドトリガー価格変動率 (デフォルト 1%)
  - `pyramid_ratios`: 追加ロット比率リスト `[0.5, 0.3, 0.2]` (減少型)
  - `max_pyramids`: 最大追加回数 (デフォルト 3)
  - `reset(entry_price, initial_size)`: 新規ポジション時に BacktestEngine から呼び出し
  - `on_bar(direction, current_price)`: 毎バーチェック → 追加ロット数を返す
- **`lib/backtest.py`** — LivermorePyramidingSizer 連携追加 (v4.2)
  - `_use_livermore = sizer is not None and hasattr(sizer, 'on_bar')` 検出
  - 新規エントリー時: `sizer.reset(entry_price, pos_size)` 呼び出し
  - 毎バー: `sizer.on_bar(dir, close)` でピラミッド追加判定 (ATRベースと排他)

### バックテスト結果 (期間: 2023-10〜2026-02, XAUUSD 3,714バー)

#### Task 1: XAUUSD + Kelly(f=0.25)

| 戦略 | Sharpe | Calmar | MDD% | Trades | 最終資産 | 備考 |
|---|---|---|---|---|---|---|
| Union_XAUUSD_Base | 1.718 | 5.971 | 23.5 | 70 | ¥21,795,420 | ベースライン |
| **XAUUSD+Kelly(f=0.25)** | **1.717** | **6.574** | 26.3 | 70 | **¥25,701,780** | **+18%↑** |

- Kelly乗数: WR=50%, PF=1.828 → f*=0.0566 → 乗数 **1.13x**
- 最終資産: ¥21,795,420 → ¥25,701,780 (+18%改善) — Calmar 5.971→6.574 (+10%)
- Sharpe はほぼ同値 (1.718→1.717)。Kelly で複利効果が出ている

#### Task 3: XAUUSD + Kelly + Livermore ピラミッティング

| 戦略 | Sharpe | Calmar | MDD% | Trades | 最終資産 | 備考 |
|---|---|---|---|---|---|---|
| Union_XAUUSD_Base | 1.718 | 5.971 | 23.5 | 70 | ¥21,795,420 | ベースライン |
| XAUUSD+Kelly(f=0.25) | 1.717 | 6.574 | 26.3 | 70 | ¥25,701,780 | Task1 |
| **XAUUSD+Kelly+Pyramid(LV)** | **0.030** | **-0.178** | **42.9** | **105** | **¥4,091,874** | **❌ 非推奨** |

- ピラミッド発動率: 105トレード中56件 (53.3%)、平均レイヤー数: 2.5
- MDD: 23.5% → 42.9% (+19.4pt)、PF: 1.828 → 0.932 (1.0を下回る)
- WR: 50.0% → 35.2% (ピラミッド後ポジションがSLに当たる頻度が高い)

### 考察・知見
- **Kelly(f=0.25) は有効**: 複利効果で最終資産+18%、Calmar改善。XAUUSD でのメイン戦略として推奨
- **Livermore式ピラミッティング (step_pct=1%) は非推奨**:
  - 4H金相場では1%の動きが頻繁に起こるため、追加ポジションが高値圏で積まれる
  - SL（前レイヤー建値）が遠くなり、逆行時の損失が増大
  - **改善案**: step_pct を 2-3% に引上げ、max_pyramids=1〜2 に制限することで効果を出せる可能性あり
- **次期課題**: step_pct / max_pyramids のグリッドサーチで最適パラメータ探索

---

## [2026-03-01] v15: ユニバース拡張 + ADXフィルター検証

**変更者:** Claude Code
**変更種別:** Union戦略横展開 + シンプルレジームフィルター検証
**指示書:** `prompt_for_claude_code_v15.md`

### 新設ファイル
- **`scripts/backtest_universe.py`** — Union_4H_Base を8商品に一括バックテスト（データなし商品は自動スキップ）
- **`lib/approved_list.py`** — バックテスト結果から承認済み商品リストを生成（Sharpe>1.0 AND Trades>20）
- **`scripts/backtest_adx_filter.py`** — XAGUSD に ADX(14)>25 フィルターを追加検証
- **`results/universe_performance.csv`** — ユニバース全商品バックテスト結果
- **`results/approved_universe.json`** — 承認済み商品リスト（JSON形式）

### バックテスト結果: Task 1 ユニバース拡張

利用可能データ: XAUUSD / XAGUSD (他6商品はデータなし → スキップ)

| 商品 | Sharpe | Calmar | MDD% | PF | WR% | Trades | 期間 |
|---|---|---|---|---|---|---|---|
| XAUUSD | **1.718** | **5.971** | 23.5 | 1.828 | 50.0 | 70 | 2023-10〜2026-02 |
| XAGUSD | 1.189 | 2.111 | 25.3 | 1.611 | 57.6 | 33 | 2025-01〜2026-02 |

### Task 2: 承認リスト (`lib/approved_list.py`)
- 承認条件: `Sharpe > 1.0` AND `Trades > 20`
- 承認商品: **XAUUSD, XAGUSD** (2商品)
- `python lib/approved_list.py` で標準出力、`--save` でJSON保存
- 将来の実運用botが起動時に呼び出して取引対象を動的に決定

### バックテスト結果: Task 3 ADXフィルター検証

対象: XAGUSD | ADX(14)統計: mean=26.2, median=24.4, max=64.8

| 戦略 | Sharpe | Calmar | MDD% | Trades | 判定 |
|---|---|---|---|---|---|
| Union_XAGUSD_Base | 1.189 | 2.111 | 25.3 | 33 | ベースライン |
| Union+ADX(>25) | 1.008 | 1.535 | **21.8** | 21 | **非推奨** |

- MDD: -3.5pt (改善) だがトレード数 33→21 (-36%)、Sharpe 1.189→1.008 (-0.181)
- XAGUSD は ADX>25 が47.3%のバー — Union自体がトレンドフォロー設計のためフィルター効果が限定的
- **結論: ADXフィルター非推奨。Union素のまま or HMM v3.0 が優位**

### 考察
- **XAUUSD > XAGUSD**: Union戦略はXAUUSDで Sharpe=1.718 > XAGUSD=1.189
- **ADXフィルター**: シンプルさの反面、有効シグナルも排除。フィルターなしUnionが優位
- **次期課題**: 他6商品のOHLCデータ取得後に本格的なユニバース評価を実施

---

## [2026-03-01] v14: 5D HMM特徴量強化 + Union+Kellyモニター

**変更者:** Claude Code
**変更種別:** HMM特徴量拡張 + リアルタイム監視インフラ構築
**指示書:** `prompt_for_claude_code_v14.md`

### 新設ファイル
- **`monitors/monitor_union_kelly.py`** — Union+Kelly シグナルモニター (OANDA v20 API, 発注なし)
- **`deploy/union_kelly_monitor.service`** — systemd サービス定義テンプレート
- **`scripts/backtest_v14.py`** — v14 MetaStrategy v2 vs v3 比較バックテスト
- **`results/v14_equity_curves.csv`** — 3戦略の資産曲線データ
- **`results/v14_hmm_regimes.csv`** — 5D HMM日次レジームデータ (ATR/ADX/RSI含む)

### 修正ファイル
- **`lib/regime.py`** — 5D HMM対応に全面改修 (v3.0)
  - `fit(close, ohlc_df=None)` / `predict(close, ohlc_df=None)` — ohlc_df指定で5D特徴量
  - 5D特徴量: `[log_return, abs_return, ATR(14), ADX(14), RSI(14)]` — z-score正規化
  - 後方互換: ohlc_df=None で旧2D動作
  - `feature_stats_by_regime()` 追加 — ダッシュボード表示用特徴量統計
- **`lib/indicators.py`** — `Ind.adx()` 追加 (ADX計算、Wilder平滑化)
- **`strategies/meta_strategy.py`** — v3.0追加
  - `make_meta_signal_v3(daily_close, daily_ohlc, ...)` 5D HMMシグナル生成
  - `grid_search_meta_v3()` グリッドサーチ
- **`dashboard.html`** — v14セクション追加
  - v2.0(2D) vs v3.0(5D) HMM円グラフ対比
  - MetaStrategy v2 vs v3 Sharpe/Calmar/MDD比較棒グラフ
  - 5D HMM特徴量統計テーブル (レジーム別 avg ATR/ADX/RSI)

### バックテスト結果サマリー (v14)
| 戦略 | Sharpe | Calmar | MDD% | PF | WR% | 備考 |
|---|---|---|---|---|---|---|
| Union_4H_Base | **2.817** | 13.7 | 9.8 | 3.624 | 66.7 | ベースライン |
| MetaStrategy v2.0 (2D HMM) | 1.366 | 2.425 | 21.9 | 1.821 | 53.9 | v13実績 |
| **MetaStrategy v3.0 (5D HMM)** | 1.359 | **3.158** | **14.4** | 1.938 | 55.0 | **MDD・Calmar改善** |

### 5D HMM レジーム分布
| レジーム | v2.0(2D) | v3.0(5D) | avg ATR | avg ADX | avg RSI |
|---|---|---|---|---|---|
| range | 49.3% | 45.2% | 38.7 | 26.6 | 64.5 |
| low_trend | 49.0% | 48.3% | 68.1 | 28.1 | 60.8 |
| **high_trend** | **1.7%** | **6.6%** | **172.3** | 29.9 | 55.5 |

high_trend検出: 1.7% → 6.6% (+4.9pt) — ATR特徴量がボラ急騰期間を正確に分離

### 考察・知見
- **5D HMMの効果**: MDD 21.9%→14.4%, Calmar 2.425→3.158 — 目標(Calmar>3.0)達成
- **high_trend分離**: ATR(14)がhigh_trend期間(ATR=172)を効果的に識別。その期間は日次リターン-0.14%と負のため、Union戦略(強トレンドフォロー)の選択が適切
- **Sharpe横ばい**: 1.366→1.359 — Sharpeは変わらないがリスク指標が大幅改善
- **Union+Kellyモニター**: OANDA v20 REST APIで4H足を取得、Union+Kelly(×2.4)シグナルをログ出力。`--once` オプションで単発実行も可能
- **systemd**: `deploy/union_kelly_monitor.service` でデーモン化対応。30分ポーリング

### 次期課題
- MetaStrategy v3.0 のグリッドサーチ範囲拡大 (MAEDAI_GRID, UNION_GRID)
- OANDA API実連携テスト (本番アカウント接続前にpractice環境で検証)
- high_trend期間向けの短期戦略追加検討

---

## [2026-03-01] v13: Kellyの最適化 + MetaStrategy再構築 (3状態HMM)

**変更者:** Claude Code
**変更種別:** サイジング最適化 + HMMレジーム精度向上
**指示書:** `prompt_for_claude_code_v13.md`

### 変更内容

#### Task 1: HybridKellySizer (lib/risk_manager.py)
- `KellyCriterionSizer(f=0.5)` × `VolatilityAdjustedSizer` の積算
- kelly_fraction を v12の0.25 → 0.5 に引上げ
- 結果: Sharpe 2.630, MDD 33.5% (MDD 30%超でリスク過大 — Kelly f=0.25が最適と確認)

#### Task 2: MetaStrategy v2.0 (3状態HMM)
- `lib/regime.py` を3状態対応に全面改修
  - 観測変数: [log_return, abs_log_return] の2次元でリターン方向+ボラを同時学習
  - ラベル: range(低ボラ) / low_trend(中ボラ+正リターン) / high_trend(高ボラ)
- `strategies/meta_strategy.py` を3状態対応に全面改修
  - range → YagamiA_4H / low_trend → Maedai_DC / high_trend → Union_4H
  - グリッドサーチ (3組合せ): Maedai lookback 10/15/20 を探索
- レジーム分布: range=49.3%, low_trend=49.0%, high_trend=1.7% (v1.0のtrend=1.4%から大幅改善)
- **MetaStrategy Sharpe: 0.581 → 1.366 (+135%改善)**

#### Task 3: VolSizer再設計 (disable_atr_sl)
- `lib/backtest.py run()` に `disable_atr_sl=True` オプション追加
- ATR-SLとVolSizerの重複解消 → 固定SL(default_sl_atr×ATR)に切替
- 結果: Sharpe 1.946, Trades 21→40 (シグナル感度変化), MDD 19.3%

#### Task 4: dashboard.html v13セクション更新
- 全6戦略の資産曲線比較 (Union/Kelly0.25/HybridKelly/MetaV1/MetaV2)
- v12/v13 Sharpe×Calmarグラフに v13結果を追記
- 3状態HMMレジーム円グラフ追加 (range49%/low_trend49%/high_trend2%)

### バックテスト結果サマリー (v13)
| 戦略 | Sharpe | Calmar | MDD% | 結論 |
|---|---|---|---|---|
| Union_4H_Base | **2.817** | 13.7 | 9.8 | ベース |
| Union+Kelly(f=0.25) | 2.798 | **24.5** | 22.8 | **Calmar最高** |
| Union+HybridKelly(f=0.5) | 2.630 | 19.6 | 33.5 | MDD過大 |
| Union+VolSizer(noSL) | 1.946 | 6.3 | 19.3 | 感度変化 |
| MetaStrategy v2 | 1.366 | 2.4 | 21.9 | **+135%改善** |

---

## [2026-03-01] v12: 1000万→1億エンジン — VolSizer / Kelly / HMM MetaStrategy

**変更者:** Claude Code
**変更種別:** 新サイジングエンジン実装 + レジーム転換モデル
**指示書:** `prompt_for_claude_code_v12.md`

### 新設ファイル
- **`lib/risk_manager.py`** — `VolatilityAdjustedSizer` + `KellyCriterionSizer`
- **`lib/regime.py`** — `HiddenMarkovRegimeDetector` (hmmlearn GaussianHMM 2状態)
- **`strategies/meta_strategy.py`** — `MetaStrategy` (HMM連動シグナル切替)
- **`scripts/backtest_v12.py`** — v12統合バックテスト実行スクリプト
- **`requirements.txt`** — 依存パッケージ一覧 (numpy, pandas, hmmlearn>=0.3.0)

### 修正ファイル
- **`lib/backtest.py`** — `run()` に `sizer=` パラメータ追加（VolSizer/Kellyを差し込み可能）
- **`results/v12_equity_curves.csv`** — 4戦略の資産曲線データ
- **`dashboard.html`** — v12セクション追加（資産曲線比較 + Sharpe/Calmar棒グラフ）

### バックテスト結果 (期間: 2025-01〜2026-02, 初期資金500万)
| 戦略 | Sharpe | Calmar | MDD% | PF | WR% |
|---|---|---|---|---|---|
| Union_4H_Base (ベースライン) | **2.817** | 13.7 | 9.8 | 3.624 | 66.7 |
| Union+VolSizer | 2.656 | 9.6 | 11.9 | 3.547 | 66.7 |
| Union+Kelly(×2.4) | 2.798 | **24.5** | 22.8 | 3.566 | 66.7 |
| MetaStrategy(HMM) | 0.581 | 0.6 | 26.0 | 1.264 | 43.5 |

### 考察・知見
- **VolSizer**: ATR正規化でDD抑制を期待したが、既存のATRベースSL設計が既にボラ調整済みのため効果限定的
- **Kelly(×2.4)**: 同じ21トレード・同方向だが複利効果で最大3,518万に到達。ただしMDD 22.8% — 高リスク高リターン
- **MetaStrategy**: HMMがトレンド判定1.4%と保守的すぎ、ほぼYagamiAに切替→パフォーマンス低下。レジームしきい値の調整が必要
- **次期課題**: HMMのn_states=3（レンジ/緩やかトレンド/強トレンド）や特徴量の見直しが有効か検討

---

## [2026-03-01] v11: Reality Check — 実績データとバックテストの照合

**変更者:** Claude Code
**変更種別:** 現実データとの照合（Reality Check）
**指示書:** `prompt_for_claude_code_v11.md`
**ブランチ運用:** Claude Code → `claude/add-trading-backtest-ePJat` / Manus AI → `main` へマージ

### 変更内容

#### Task 1: 2026年1月 ドローダウン検証 [Teammate C]
- **実績:** 2026-01 は -1,128,362円（金スポット -701,906円 / 銀スポット -550,074円）
- **XAUUSD価格:** 実際は上昇トレンド（4,399 → 5,068、+668pt）。実トレーダーは逆張りショートで大損失
- **Union戦略の判断:** 1月シグナルは **long 2件のみ（1/2・1/4）**、バックテスト完結トレード = **0件（FLAT）**
- **考察:** Union戦略は過熱相場（急騰局面）でシグナルを出さない設計のため、実トレーダーが犯したショートポジションリスクを自動回避。**ドローダウン耐性の観点で有意な差**を確認

#### Task 2: 2025年12月 勝ちトレード照合 [Teammate B]
- **実績:** 2025-12 は +7,288,126円（金+銀、決済292件、勝率 178/292 = 61%）
- **Union戦略シグナル:** 12/01 01:00 long（4,280円）・12/09 05:00 long（4,220円）
- **実績大勝ち日:** 12/22（+282万）・12/23（+265万）・12/24（+153万）
- **照合結果:** Unionシグナル（12/1, 12/9）は実績大勝ちの約2〜3週間前にlong発令 → **実トレーダーの12月初のロングエントリーと方向性一致**。12/9の4,220は月中最安値付近（最安値 4,213/12/4）で押し目買いシグナル = 実際に最も有効なエントリーポイント
- **結論:** Union戦略の方向性は「正解」。時間軸のズレ（バーベース決済 vs 実際の保有継続）が乖離の主因

#### Task 3: ダッシュボード Real World Performance 追加 [全員]
- **`dashboard.html`** に「🌍 Real World Performance」セクションを新規追加
  - KPI カード: 実現損益合計 +1,759万円 / 決済件数 1,369件・勝率 70.6% / 最大損失月 -112万（2026-01） / Union1月行動 FLAT
  - 月次損益 棒グラフ + 実績累計 vs バックテスト累計 折れ線グラフ（Plotly.js）
  - 銘柄別実現損益内訳（金スポット +1,196万 / 銀スポット +518万 / NQ100 +45万）

### 実績データ月次サマリー (Ground Truth)
| 月 | 実現損益 | 金スポット | 銀スポット | Union戦略 |
|---|---|---|---|---|
| 2025-09 | +60万 | +60万 | — | — |
| 2025-10 | +179万 | +179万 | — | — |
| 2025-11 | +159万 | +159万 | — | — |
| 2025-12 | **+729万** | +大勝ち | +大勝ち | long 2件 (方向一致) |
| 2026-01 | **-113万** | -70万 | -55万 | **FLAT** (ドローダウン回避) |
| 2026-02 | +745万 | +745万 | — | 継続保有 |
| **合計** | **+1,759万** | | | |

---

## [2026-03-01] v9成果の統合・フォワードテスト準備・戦略ダッシュボード構築

**変更者:** Claude Code
**変更種別:** v9成果の統合、フォワードテスト準備、戦略ダッシュボード構築
**指示書:** prompt_for_claude_code_v10_updated.md

### 変更内容

- **mainブランチへのマージ**: `claude/add-trading-backtest-ePJat` → `main` のローカルマージ完了。v8-v9の全成果を main に統合 (125ファイル)。リモートへのプッシュは `claude/` ブランチ制約のため dev ブランチ経由で管理。
- **Union戦略フォワードテスト監視**: `monitors/forward_union.py` を新規作成。最新OHLCを読み込みUnion戦略シグナルをログ出力（1時間cronまたはwatchモード対応）。シグナルは `trade_logs/forward_union_signals.csv` に記録。
- **USD強弱フィルター適用ガイドライン**: `docs/filters_risk_manager.md` の「USD強弱フィルター」セクションに「逆張り系に有効、トレンドフォロー系に効果限定的」という知見と適用ガイドラインを追記。v9検証比較表を根拠として引用。
- **バックテスト自動ログ**: `lib/backtest.py` の `_report()` を修正し、バックテスト完了時に `results/performance_log.csv` へ自動追記する `_append_performance_log()` を追加。
- **戦略ダッシュボード**: `dashboard.html` を新規作成。Plotly.js により Sharpe Ratio 棒グラフ・MDD 棒グラフ・リスク/リターン散布図・全戦略テーブルを表示。インライン CSV と動的フェッチの両対応。

### 主要バックテスト結果 (v10 シード)

| 戦略 | TF | Sharpe | PF | MDD% | WR% | Trades | 判定 |
|------|:--:|:------:|:--:|:----:|:---:|:------:|------|
| Union_4H | 4H | **2.817** | 3.6245 | 9.82% | 66.7% | 21 | PASS |
| DC30_EMA200 | 4H | 1.414 | 2.4948 | 10.52% | 57.9% | 19 | PASS |
| DC30_EMA200+USD | 4H | 1.414 | 2.4948 | 10.52% | 57.9% | 19 | PASS |
| YagamiFull_1H | 1H | 0.748 | 1.1958 | 30.8% | 35.7% | 129 | CHECK |
| YagamiFull_1H_S | 1H | 0.666 | 1.1627 | 30.52% | 35.5% | 121 | CHECK |
| YagamiA_4H | 4H | 0.668 | 1.1076 | 49.9% | 40.2% | 164 | CHECK |

### 追加・変更ファイル

| ファイル | 説明 |
|:---|:---|
| `monitors/__init__.py` | monitors パッケージ初期化 |
| `monitors/forward_union.py` | Union戦略フォワードテスト監視スクリプト |
| `docs/filters_risk_manager.md` | USD強弱フィルター適用ガイドライン追記 |
| `lib/backtest.py` | `_append_performance_log()` 追加 (auto CSV log) |
| `results/performance_log.csv` | バックテスト自動ログ (v10シード実行済み) |
| `dashboard.html` | 戦略パフォーマンス ダッシュボード (Plotly.js) |

---

---

## [2026-03-01] v8成果の検証と戦略確立

**変更者:** Claude Code
**変更種別:** v8成果の検証と戦略確立
**指示書:** prompt_for_claude_code_v9.md

### 変更内容

- **v8成果物のプッシュ確認**: `strategies/`・`docs/` は `claude/add-trading-backtest-ePJat` ブランチ上に既にプッシュ済みを確認
- **Union戦略の確立**: `strategies/union_strategy.py` として単独実行可能スクリプトを作成。`docs/strategy_union.md` にロジックと実績を記録。v9再現バックテストで Sharpe 2.817 (目標1.5超え) を確認
- **USD強弱フィルター横展開**: DC30_EMA200 (Maedai) と YagamiFull_1H (Yagami) にフィルター適用。DC30はDonchianブレイクがUSD強時と重ならないためフィルター効果なし。YagamiFull_1H ではMDD 30.8%→29.0% (-1.8%) に微改善
- **季節フィルターの戦略別最適化**: YagamiFull_1H は 7月+9月除外 (`SEASON_SKIP_JUL_SEP`) をデフォルト採用。YagamiA_4H は 9月がプラス月のため全月対象 (`SEASON_ALL`) を採用。`docs/filters_risk_manager.md` に記録

### v9 主要バックテスト結果

**Union戦略 (XAUUSD 2025 4H):**

| 戦略 | PF | WR% | MDD% | Sharpe | Calmar |
|------|:--:|:---:|:----:|:------:|:------:|
| Union_4H (素) | 3.624 | 66.7% | 9.8% | **2.817** | 13.709 |
| Union_4H+USD | 4.025 | 66.7% | 10.5% | 2.686 | 10.681 |

**USD強弱フィルター横展開 (XAUUSD 2023-2026 4H/1H):**

| 戦略 | PF | MDD% | Sharpe | Calmar | 変化 |
|------|:--:|:----:|:------:|:------:|------|
| DC30_EMA200 | 2.495 | 10.5% | 1.414 | 3.877 | — |
| DC30_EMA200+USD | 2.495 | 10.5% | 1.414 | 3.877 | 変化なし |
| YagamiFull_1H | 1.196 | 30.8% | 0.748 | 1.089 | — |
| YagamiFull_1H+USD | 1.200 | 29.0% | 0.749 | 1.163 | MDD -1.8% |

### 追加・変更ファイル

| ファイル | 説明 |
|:---|:---|
| `strategies/union_strategy.py` | Union戦略 単独実行スクリプト |
| `docs/strategy_union.md` | Union戦略ドキュメント (ロジック・パラメータ・実績) |
| `docs/filters_risk_manager.md` | 戦略別季節フィルター決定事項 + USD横展開結果を追記 |

---

## [2026-03-01] 開発フレームワークの刷新

**変更者:** Claude Code
**変更種別:** 開発フレームワークの刷新
**指示書:** prompt_for_claude_code_v8.md

### 変更内容

- **戦略ポートフォリオアプローチ導入**: 成功バイアス回避のため、3チーム体制による戦略ポートフォリオ (Yagami, Maedai, Risk Manager) アプローチを導入
- **USD強弱フィルター実装**: `strategies/market_filters.py` に `calc_usd_strength()` を実装。XAUUSDの逆モメンタムからUSD強弱プロキシを算出し、threshold=75 (上位25%) でロングシグナルを除去
- **評価指標追加**: Sharpe Ratio, Calmar Ratio を `lib/backtest.py` の `_report()` メソッドに追加
- **ドキュメントとファイル構成の分離**: `strategies/` ディレクトリに戦略別ファイル (`yagami_rules.py`, `maedai_breakout.py`, `market_filters.py`) を配置、`docs/` にチーム別ドキュメントを整備
- **Maedai戦略のDonchianパラメータ探索スクリプト追加**: `strategies/maedai_breakout.py` に `DC_PARAM_GRID` (DC期間: 10/15/20/30/40, EMA: 100/200) と `maedai_dc_variants()` を追加

### 追加・変更ファイル

| ファイル | 説明 |
|:---|:---|
| `strategies/__init__.py` | 戦略ポートフォリオ ハブ |
| `strategies/market_filters.py` | Teammate C: USD強弱フィルター + 季節フィルター |
| `strategies/yagami_rules.py` | Teammate A: Yagami戦略バリアント |
| `strategies/maedai_breakout.py` | Teammate B: Maedai戦略 + Donchianパラメータグリッド |
| `lib/backtest.py` | `_report()` に Sharpe Ratio / Calmar Ratio 追加 |
| `docs/strategy_yagami.md` | Teammate A 戦略ドキュメント |
| `docs/strategy_maedai.md` | Teammate B 戦略ドキュメント |
| `docs/filters_risk_manager.md` | Teammate C フィルター・リスク管理ドキュメント |
| `overheat_monitor.py` | XAUT/XAUUSD 過熱度モニター |
| `price_zone_analyzer.py` | 価格帯滞在時間ヒストグラム (薄いゾーン検出) |

---

## [2026-02-27] 本物のマーケットデータ追加

**変更者:** Manus AI
**変更種別:** データ追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `data/ohlc/XAUUSD_1d.csv` | 日足データ (2019-2026, 1,801バー) |
| `data/ohlc/XAUUSD_1h.csv` | 1時間足データ (2023-2026, 13,693バー) |
| `data/ohlc/XAUUSD_4h.csv` | 4時間足データ (2023-2026, 3,714バー) |
| `data/ohlc/XAUUSD_8h.csv` | 8時間足データ (2023-2026, 1,937バー) |
| `data/ohlc/README.md` | データの説明書 |
| `MANUS_CHANGELOG.md` | この変更履歴ファイル |

### 変更理由

Claude Codeが外部API（Dukascopy等）にアクセスできず、GBM（幾何ブラウン運動）による合成データを使用してバックテストを行っていた。合成データでのバックテスト結果は本番環境での再現性が保証されないため、Manus側でYahoo Finance（GC=F: 金先物）から本物のマーケットデータを取得し、リポジトリに追加した。

### Claude Codeへの指示

1. `data/ohlc/` 配下のCSVファイルをバックテストのデータソースとして使用すること
2. 合成データ（GBM生成）は破棄し、本物のデータに完全に置き換えること
3. `scripts/fetch_data.py` のデータ読み込みパスを `data/ohlc/` に変更すること

### データ品質

- ソース: Yahoo Finance (GC=F)
- NULL値: なし（全ファイル検証済み）
- 価格範囲: $1,266 〜 $5,586（日足基準）
- 注意: GC=F（金先物）とXAUUSD（スポット金）には微小な価格差がある

---

## [2026-03-02] Claude Code指示書v11追加：「バックテスト vs 現実」— 実績データとの照合

**変更者:** Manus AI
**変更種別:** 指示書追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `prompt_for_claude_code_v11.md` | Claude Code指示書v11（実績データとの照合） |

### 変更理由

Claude Codeがリポジトリに統合した2,642件の実際のCFD取引履歴を分析した結果、**+1,758万円**という驚異的な利益と**勝率70.6%**という高いパフォーマンスが確認された。この「実績」をGround Truth（正解データ）とし、バックテスト戦略がどの程度現実を再現できるかを検証するフェーズに移行する。

### 主なタスク指示

1.  **2026年1月のドローダウン分析:** 実績で-112万円の損失が出た1月を、Union戦略のバックテストで再現できるか検証する。
2.  **実績トレードと戦略シグナルの照合:** 利益が最大だった2025年12月の実績勝ちトレードと、Union戦略の買いシグナルがどの程度一致するかを分析する。
3.  **戦略ダッシュボードへの「現実」追加:** `dashboard.html`に現実の月次損益グラフを追加し、バックテストと現実のパフォーマンスを比較できるようにする。

### Claude Codeへの指示

`prompt_for_claude_code_v11.md`を読み、記載されたタスクを順次実行すること。

---

## [2026-03-02] Claude Code指示書v13追加：Kellyの最適化とMetaStrategyの再構築

**変更者:** Manus AI
**変更種別:** 指示書追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `prompt_for_claude_code_v13.md` | Claude Code指示書v13（Kelly最適化とMetaStrategy再構築） |

### 変更理由

v12の結果、Kellyサイザーの有効性とMetaStrategyの課題が明確になった。v13では、Kellyの爆発力とDD抑制を両立させる`HybridKellySizer`を開発し、HMMを3状態に拡張してMetaStrategyを再構築することで、プロジェクトを次のレベルへ引き上げる。

### 主なタスク指示

1.  **Kelly-Volハイブリッドサイザー:** KellyとVolatility Sizerを掛け合わせる`HybridKellySizer`を実装。
2.  **MetaStrategy再構築:** HMMを3状態（高ボラトレンド、低ボラトレンド、レンジ）に拡張し、レジームごとに戦略とパラメータを動的に切り替える。
3.  **VolSizer再設計:** バックテストエンジンを改造し、ATR-SLとの重複を解消。
4.  **ダッシュボード更新:** v13の成果（ハイブリッドサイザー、新MetaStrategy）を可視化。

### Claude Codeへの指示

`prompt_for_claude_code_v13.md`を読み、記載されたタスクを順次実行すること。

---

## [2026-03-02] Claude Code指示書v14追加：MetaStrategy最終進化とUnion+Kelly実運用準備

**変更者:** Manus AI
**変更種別:** 指示書追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `prompt_for_claude_code_v14.md` | Claude Code指示書v14（MetaStrategy最終進化とUnion+Kelly実運用準備） |

### 変更理由

v13の結果、Union+Kelly(f=0.25)が最高効率戦略として確立された一方、MetaStrategyはHMMのトレンド検出率の低さが課題として残った。v14では、HMMにATR/ADX/RSIを追加する特徴量エンジニアリングでMetaStrategyの最終進化を目指し、同時にUnion+Kellyを実運用可能な状態に準備する。

### 主なタスク指示

1.  **MetaStrategy最終進化:** HMMの観測値にATR/ADX/RSIを追加し、5次元の特徴量でレジーム判定精度を向上させる。
2.  **Union+Kelly実運用準備:** リアルタイムでシグナルを監視し、ログを出力する`monitor_union_kelly.py`と、それをデーモン化するための`systemd`サービスファイルを作成。
3.  **ダッシュボード最終化:** v14の成果（5次元HMM、MetaStrategy v3.0）を可視化。

### Claude Codeへの指示

`prompt_for_claude_code_v14.md`を読み、記載されたタスクを順次実行すること。

---

## [2026-03-02] Claude Code指示書v15追加：Union戦略ユニバース拡張とADXフィルター検証

**変更者:** Manus AI
**変更種別:** 指示書追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `prompt_for_claude_code_v15.md` | Claude Code指示書v15（Union戦略ユニバース拡張とADXフィルター検証） |

### 変更理由

v14でMetaStrategy v3.0がUnion単体を超えられなかったため、複雑なHMMアプローチを一旦保留。v15では、最も強力なUnion戦略を複数商品に横展開し、有効な商品を網羅的に検証する方針に転換する。また、HMMの代替としてシンプルなADXフィルターの有効性をテストする。

### 主なタスク指示

1.  **Union戦略ユニバース拡張:** 8つの主要商品（金、銀、株価指数、FX）でUnion戦略のバックテストを一括実行し、結果を`universe_performance.csv`に保存。
2.  **承認リスト自動生成:** バックテスト結果から`Sharpe > 1.0`かつ`Trades > 20`の商品を自動で抽出する`approved_list.py`を作成。
3.  **ADXフィルター検証:** Union戦略に`ADX(14) > 25`の条件を追加し、フィルターの効果を評価。

### Claude Codeへの指示

`prompt_for_claude_code_v15.md`を読み、記載されたタスクを順次実行すること。

---

## [2026-03-02] Claude Code指示書v16追加：XAUUSDメイン化とリバモア式ピラミッティング

**変更者:** Manus AI
**変更種別:** 指示書追加

### 追加ファイル

| ファイル | 説明 |
|:---|:---|
| `prompt_for_claude_code_v16.md` | Claude Code指示書v16（XAUUSDメイン化とリバモア式ピラミッティング） |

### 変更理由

v15でXAUUSDがSharpe 1.718, Calmar 5.971という圧倒的な結果を出したため、v16ではXAUUSDをメイン戦略に昇格させる。さらに、利益最大化を目指してリバモア式ピラミッティングを導入する。

### 主なタスク指示

1.  **XAUUSD + Kelly適用:** XAUUSDに`KellyCriterionSizer(f=0.25)`を適用したバックテストを実行。
2.  **リバモア式ピラミッティング実装:** 含み益が出ているポジションに機械的に追加投資する`LivermorePyramidingSizer`を実装。
3.  **統合バックテスト:** XAUUSDにKellyとピラミッティングを組み合わせた最終形態のバックテストを実行。

### Claude Codeへの指示

`prompt_for_claude_code_v16.md`を読み、記載されたタスクを順次実行すること。

---

## [2026-03-02] Claude Code指示書v17追加：ピラミッティング最適化とADXフィルター検証

**変更者:** Manus AI
**変更種別:** 指示書追加

### v16バックテスト結果のサマリー

| 戦略 | Sharpe | Calmar | MDD% | Trades | 最終資産 | 判定 |
|:---|:---|:---|:---|:---|:---|:---:|
| XAUUSD+Kelly(f=0.25) | 1.717 | 6.574 | 26.3 | 70 | ¥25,701,780 | ✅ **推奨** |
| XAUUSD+Kelly+Pyramid(LV) | 0.030 | -0.178 | 42.9 | 105 | ¥4,091,780 | ❌ **非推奨** |

- **考察1：Kellyは有効**
  - `KellyCriterionSizer(f=0.25)`の適用により、最終資産が+18%（2179万→2570万）、Calmarが+10%（5.971→6.574）向上。XAUUSDのメイン戦略として採用を決定。
- **考察2：ピラミッティングは失敗**
  - `step_pct=1%`が4時間足のXAUUSDには敏感すぎた。高値掴みからの損切りが多発し、勝率が50%→35.2%に悪化、PFも1.0を割り込んだ。

### v17の主なタスク指示

v16の結果を踏まえ、以下のタスクを指示します。

1.  **ピラミッティング・パラメータ最適化:**
    - `step_pct`を`[0.02, 0.03, 0.04]`、`max_pyramids`を`[1, 2]`の範囲でグリッドサーチを実行し、最適な組み合わせを発見する。
2.  **ADXフィルター検証:**
    - XAGUSDでは逆効果だったADXフィルター(>25)が、XAUUSDで有効か否かを最終検証する。
3.  **承認リスト基準の厳格化:**
    - `approved_list.py`の自動承認基準を`Sharpe > 1.5`かつ`Calmar > 5.0`に引き上げ、XAUUSDレベルの戦略のみを対象とする。

### Claude Codeへの指示

`prompt_for_claude_code_v17.md`を読み、記載されたタスクを順次実行すること。

---

## [2026-03-02] Claude Code指示書v18追加：最終戦略の確立とフォワードテスト準備

**変更者:** Manus AI
**変更種別:** 指示書追加

### v17バックテスト結果のサマリー

**🔑 重要発見：ADXフィルターがXAUUSDで絶大な効果を発揮**

| 戦略 | Sharpe | Calmar | MDD% | WR% | Trades | 判定 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| XAUUSD+Kelly(f=0.25) | 1.717 | 6.574 | 26.3 | 50.0 | 70 | ベースライン |
| **XAUUSD+Kelly+ADX(>25)** | **2.250** | **11.534** | **16.1** | **61.0** | **41** | ✅ **最終戦略** |

- **考察1：ADXフィルターはゲームチェンジャー**
  - XAGUSDでは逆効果だったADXフィルターが、XAUUSDではSharpeを+31%、Calmarを+75%向上させ、MDDを10.2ポイントも低減させました。これは、XAUUSDの相場がADX>25となる高トレンド状態（期間の53.6%）であることが多く、トレンドフォロー戦略との相性が抜群に良いためです。
- **考察2：ピラミッティングは不採用**
  - `step_pct`を2%〜4%に広げても、ベースラインのSharpe(1.717)を超えることはありませんでした。これにより、リバモア式ピラミッティングは本戦略では採用しないことが確定しました。

### v18の主なタスク指示

v17の画期的な結果に基づき、戦略の最終化と本番移行準備に入ります。

1.  **最終戦略の確立:**
    - `XAUUSD+Kelly(f=0.25)+ADX(>25)`のロジックを`strategies/main_strategy.py`として単一ファイルに統合します。
2.  **フォワードテスト準備:**
    - リアルタイム監視スクリプトを更新し、この最終戦略のシグナルをOANDA API経由で監視できるようにします。
3.  **リポジトリ整理:**
    - 役目を終えたピラミッティングやADX検証の実験用スクリプトを削除します。
4.  **ダッシュボード更新:**
    - `dashboard.html`のトップに最終戦略の驚異的なパフォーマンス（Sharpe 2.250, Calmar 11.534）を「Main Strategy」として明記します。

### Claude Codeへの指示

`prompt_for_claude_code_v18.md`を読み、記載されたタスクを順次実行すること。
