"""
マエダイメソッド バックテスト
==============================
「背を近くして何度も挑戦、大きい値動きを取る」

設計方針:
  - 勝率: 30% 以上 (低WR許容)
  - RR  : 1:8 〜 1:15 (大きな利益を一発で)
  - DD  : 最大30% 許容
  - 目標: 1回の大きなトレードで資産 +50〜100%

戦略バリエーション:
  1. Donchian20  — 1H 直近20本ブレイク (標準)
  2. Donchian10  — 1H 直近10本ブレイク (短期、より頻繁)
  3. Donchian4h  — 4H 直近20本ブレイク (大きな足)
  4. HTF_MTF     — 4H方向 + 1H精密エントリー
  5. Donchian_ATR— ATR確認付きブレイク (ダマシ削減)

実行:
  python scripts/backtest_maedai.py
  python scripts/backtest_maedai.py --dukascopy
"""
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
IMAGES_DIR  = os.path.join(BASE_DIR, 'reports', 'images')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR,  exist_ok=True)


# ──────────────────────────────────────────────
# エンジン設定: マエダイメソッド専用
# ──────────────────────────────────────────────

def make_maedai_engine(risk_pct=0.02):
    """
    マエダイ式エンジン:
    - tight SL (ATR×0.8)
    - large TP (ATR×10)
    - trailing stop: 含み益 3ATR で発動、SLを 1.5ATR 追従
    - pyramid: 2ATR ごとに同量追加 (最大3回)
    - DD上限: 30% / WR下限: 30%
    """
    from lib.backtest import BacktestEngine
    return BacktestEngine(
        init_cash       = 5_000_000,
        risk_pct        = risk_pct,      # 1トレード最大 risk_pct % リスク
        default_sl_atr  = 0.8,           # タイトSL (背を近く)
        default_tp_atr  = 10.0,          # 大きなTP
        slippage_pips   = 0.3,
        pip             = 0.1,
        use_dynamic_sl  = True,
        pyramid_entries = 3,             # 最大3回追加
        pyramid_atr     = 2.0,           # 2ATR ごとに追加
        pyramid_size_mult = 1.0,         # 同量追加 (マエダイ: 含み益が乗ったら積み増す)
        trail_start_atr = 3.0,           # 含み益 3ATR で追従開始
        trail_dist_atr  = 1.5,           # SLを価格から 1.5ATR で追従
        target_max_dd   = 0.30,          # DD 30% 許容
        target_min_wr   = 0.30,          # WR 30% でも可
        target_rr_threshold = 5.0,       # RR5 以上なら WR 15% でも合格
        target_min_trades = 20,          # 最低20トレード
    )


# ──────────────────────────────────────────────
# データ
# ──────────────────────────────────────────────

def load_data(use_dukascopy=False, start_warmup='2020-01-01'):
    """データ取得 (Dukascopy → サンプル)"""
    try:
        from scripts.fetch_data import load_ticks, ticks_to_ohlc
        ticks = load_ticks()
        if ticks is not None and len(ticks) > 10000:
            ts_start = pd.Timestamp(start_warmup)
            ticks = ticks[ticks.index >= ts_start]
            if len(ticks) > 1000:
                bars_1h = ticks_to_ohlc(ticks, '1h')
                bars_4h = ticks_to_ohlc(ticks, '4h')
                print(f"[Data] Dukascopy: 1h={len(bars_1h)}, 4h={len(bars_4h)} bars")
                return {'source': 'dukascopy', '1h': bars_1h, '4h': bars_4h}
    except Exception as e:
        print(f"[Data] Dukascopy失敗: {e}")

    print("[Data] サンプルデータ生成 (XAU ~2880 USD, 2020〜2026)")
    return _generate_sample()


def _generate_sample():
    """XAU リアル価格帯のサンプルデータ (2年分)"""
    np.random.seed(42)
    n_1h = 365 * 2 * 24   # 2年分
    base = 1800.0
    # より現実的な価格シム: 緩やかな上昇トレンド + 急落・急騰イベント
    rets = np.random.normal(0.00005, 0.0018, n_1h)
    # イベント: 5%の確率で通常の3倍ボラ
    big_move_mask = np.random.rand(n_1h) < 0.05
    rets[big_move_mask] *= 3.0
    prices = base * np.exp(np.cumsum(rets))

    idx = pd.date_range('2020-01-01', periods=n_1h, freq='1h')
    vol = np.abs(np.random.normal(0.001, 0.0005, n_1h))

    bars_1h = pd.DataFrame({
        'open':  prices,
        'high':  prices * (1 + np.abs(np.random.normal(0, 1, n_1h)) * vol),
        'low':   prices * (1 - np.abs(np.random.normal(0, 1, n_1h)) * vol),
        'close': prices * (1 + np.random.normal(0, 1, n_1h) * vol * 0.5),
    }, index=idx)
    bars_1h['high']  = bars_1h[['open', 'high', 'close']].max(axis=1)
    bars_1h['low']   = bars_1h[['open', 'low',  'close']].min(axis=1)

    bars_4h = bars_1h.resample('4h').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    ).dropna()

    print(f"[Data] サンプル: 1h={len(bars_1h)}, 4h={len(bars_4h)} bars")
    return {'source': 'sample', '1h': bars_1h, '4h': bars_4h}


# ──────────────────────────────────────────────
# 戦略リスト
# ──────────────────────────────────────────────

def build_strategies():
    from lib.yagami import (
        sig_maedai_breakout,
        sig_maedai_htf_breakout,
        sig_maedai_breakout_v2,
        sig_maedai_best,
        sig_maedai_htf_pullback,
    )

    return [
        # ── v1: Donchian (ベースライン) ──
        ('Maedai_DC20_1h',    sig_maedai_breakout('1h', lookback=20),              '1h'),
        ('Maedai_DC10_1h',    sig_maedai_breakout('1h', lookback=10),              '1h'),
        ('Maedai_DC30_1h',    sig_maedai_breakout('1h', lookback=30),              '1h'),
        ('Maedai_DC50_1h',    sig_maedai_breakout('1h', lookback=50),              '1h'),
        ('Maedai_DC20_4h',    sig_maedai_breakout('4h', lookback=20),              '4h'),
        ('Maedai_DC10_4h',    sig_maedai_breakout('4h', lookback=10),              '4h'),
        # ATR確認あり (ダマシ削減)
        ('Maedai_DC20_ATR0.3',sig_maedai_breakout('1h', lookback=20, atr_confirm=0.3), '1h'),
        ('Maedai_DC20_ATR0.5',sig_maedai_breakout('1h', lookback=20, atr_confirm=0.5), '1h'),
        # セッション全時間帯 (アジア時間も取る)
        ('Maedai_DC20_AllSess',sig_maedai_breakout('1h', lookback=20, session_filter_on=False), '1h'),
        # ── MTF: 4H方向 + 1H精密エントリー ──
        ('Maedai_HTF_10x5',   sig_maedai_htf_breakout(lookback_htf=10, lookback_ltf=5), '1h'),
        ('Maedai_HTF_20x3',   sig_maedai_htf_breakout(lookback_htf=20, lookback_ltf=3), '1h'),
        ('Maedai_HTF_5x3',    sig_maedai_htf_breakout(lookback_htf=5,  lookback_ltf=3), '1h'),

        # ── v2: レンジ品質 + エントリーモード最適化 ──
        # immediate (v1と同等、レンジ品質フィルタ追加)
        ('v2_Imm_DC20',       sig_maedai_breakout_v2('1h', lookback=20,
                                entry_mode='immediate', require_compression=True), '1h'),
        # next_bar (次バー確認でダマシ削減)
        ('v2_NextBar_DC20',   sig_maedai_breakout_v2('1h', lookback=20,
                                entry_mode='next_bar', require_compression=True), '1h'),
        # retest (後ノリ: リテストで最タイトSL)
        ('v2_Retest_DC20',    sig_maedai_breakout_v2('1h', lookback=20,
                                entry_mode='retest', require_compression=True,
                                retest_tolerance=0.5, retest_window=10), '1h'),
        ('v2_Retest_DC10',    sig_maedai_breakout_v2('1h', lookback=10,
                                entry_mode='retest', require_compression=True,
                                retest_tolerance=0.4, retest_window=8), '1h'),
        ('v2_Retest_DC30',    sig_maedai_breakout_v2('1h', lookback=30,
                                entry_mode='retest', require_compression=True,
                                retest_tolerance=0.6, retest_window=12), '1h'),
        # pullback (後ノリ: 押し目/戻りで追従エントリー)
        ('v2_Pullback_DC20',  sig_maedai_breakout_v2('1h', lookback=20,
                                entry_mode='pullback', require_compression=True,
                                pullback_window=6), '1h'),
        ('v2_Pullback_DC15',  sig_maedai_breakout_v2('1h', lookback=15,
                                entry_mode='pullback', require_compression=False,
                                pullback_window=5), '1h'),
        # 4H版 retest
        ('v2_Retest_4h',      sig_maedai_breakout_v2('4h', lookback=15,
                                entry_mode='retest', require_compression=True,
                                retest_tolerance=0.5, retest_window=8), '4h'),
        # 推奨設定 (retest + 4H方向 + ATR圧縮 + パターン)
        ('Maedai_Best_1h',    sig_maedai_best('1h'),  '1h'),

        # ── HTF方向 × 後ノリ押し目 (最強コンボ) ──
        ('HTF_Pullback_10x5', sig_maedai_htf_pullback(lookback_htf=10, pullback_bars=5), '1h'),
        ('HTF_Pullback_20x5', sig_maedai_htf_pullback(lookback_htf=20, pullback_bars=5), '1h'),
        ('HTF_Pullback_10x3', sig_maedai_htf_pullback(lookback_htf=10, pullback_bars=3), '1h'),
        ('HTF_Pullback_20x8', sig_maedai_htf_pullback(lookback_htf=20, pullback_bars=8), '1h'),
    ]


# ──────────────────────────────────────────────
# バックテスト実行
# ──────────────────────────────────────────────

def run_backtest(data, strategies, risk_pct=0.02):
    engine = make_maedai_engine(risk_pct=risk_pct)
    bars_4h = data.get('4h')
    results = []
    trade_map = {}

    for name, sig_fn, freq in strategies:
        bars = data.get(freq)
        if bars is None or len(bars) < 100:
            continue

        htf = bars_4h if freq == '1h' else None
        try:
            result = engine.run(bars, sig_fn, freq=freq, name=name, htf_bars=htf)
        except Exception as e:
            print(f"  [{name}] エラー: {e}")
            continue

        if result is None:
            continue

        results.append(result)
        trade_map[name] = result.get('trades', [])

    results.sort(key=lambda x: x.get('profit_factor', 0), reverse=True)
    return results, trade_map


# ──────────────────────────────────────────────
# 出力
# ──────────────────────────────────────────────

def print_ranking(results):
    if not results:
        print("\n[結果] 有効な戦略なし")
        return

    print(f"\n{'='*95}")
    print(f"マエダイメソッド バックテスト結果 (DD≤30% / WR≥30% / RR目標 1:8以上)")
    print(f"{'='*95}")
    hdr = (f"{'Rank':<5}{'Strategy':<25}{'TF':<5}{'Trades':>7}"
           f"{'WR%':>7}{'PF':>8}{'RR':>7}{'MaxDD%':>8}"
           f"{'Return%':>9}{'PnL':>13}{'Pass':<6}")
    print(hdr)
    print('-' * 95)

    for rank, r in enumerate(results, 1):
        pf = r.get('profit_factor', 0)
        wr = r.get('win_rate_pct', 0)
        rr = r.get('rr_ratio', 0)
        dd = r.get('max_drawdown_pct', 0)
        ret = r.get('total_return_pct', 0)
        pnl = r.get('total_pnl', 0)
        n   = r.get('total_trades', 0)
        passed = '★' if r.get('passed') else ''
        line = (f"{rank:<5}{r['strategy']:<25}{r.get('timeframe','?'):<5}{n:>7}"
                f"{wr:>7.1f}%{pf:>8.3f}{rr:>7.1f}{dd:>7.1f}%"
                f"{ret:>8.1f}%{pnl:>13,.0f}  {passed}")
        print(line)

    print('-' * 95)
    passed = [r for r in results if r.get('passed')]
    print(f"合格 (PF≥1.3, DD≤30%, WR≥30%): {len(passed)} / {len(results)}")
    print()

    # 合格戦略のサマリー
    if passed:
        print("【合格戦略 詳細】")
        for r in passed:
            print(f"  {r['strategy']}")
            print(f"    PF={r.get('profit_factor')}  WR={r.get('win_rate_pct')}%  "
                  f"RR={r.get('rr_ratio')}  DD={r.get('max_drawdown_pct')}%")
            print(f"    Return={r.get('total_return_pct')}%  "
                  f"PnL={r.get('total_pnl'):,.0f}  Trades={r.get('total_trades')}")
            pyr = r.get('pyramid_trade_rate_pct', 0)
            print(f"    Pyramid追加率={pyr:.1f}%  AvgDuration={r.get('avg_duration_hours')}h")
        print()


def print_trade_history(name, trades, max_show=50):
    print(f"\n{'='*100}")
    print(f"トレード履歴: {name}  ({len(trades)} trades)")
    print(f"{'='*100}")
    hdr = (f"{'#':>4}  {'Entry Time':<18}{'Dir':<6}"
           f"{'Entry':>9}{'Exit':>9}{'SL':>9}{'TP':>9}"
           f"{'Layers':>7}{'PnL':>12}{'Result':<8}  {'Reason'}")
    print(hdr)
    print('-' * 100)

    running = 0.0
    shown = 0
    for i, t in enumerate(trades, 1):
        entry_t = pd.Timestamp(t['entry_time']).strftime('%m-%d %H:%M')
        result  = 'WIN' if t['pnl'] > 0 else 'LOSS'
        running += t['pnl']
        layers  = t.get('pyramid_layers', 1)

        if shown < max_show or layers > 1 or t['pnl'] > 0:
            line = (f"{i:>4}  {entry_t:<18}{t['direction']:<6}"
                    f"{t['entry_price']:>9.2f}{t['exit_price']:>9.2f}"
                    f"{t['sl']:>9.2f}{t['tp']:>9.2f}"
                    f"{layers:>7}{t['pnl']:>12,.1f}  {result:<8}  "
                    f"[{t['exit_reason']}]")
            print(line)
            shown += 1
        elif shown == max_show:
            print(f"  ... (残り {len(trades) - shown} 件省略)")
            shown += 1

    print('-' * 100)
    wins   = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    avg_win  = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    rr_actual = abs(avg_win / avg_loss) if avg_loss else 0
    print(f"  Win: {len(wins)}  Loss: {len(losses)}  "
          f"WR: {len(wins)/len(trades)*100:.1f}%  "
          f"実RR: {rr_actual:.1f}  累計PnL: {running:,.1f}")


def save_results(results, trade_map, top_n=3):
    """結果をCSVに保存"""
    import json

    # ランキングCSV
    rows = []
    for r in results:
        rows.append({
            'strategy': r['strategy'],
            'timeframe': r.get('timeframe'),
            'trades': r.get('total_trades'),
            'win_rate_pct': r.get('win_rate_pct'),
            'profit_factor': r.get('profit_factor'),
            'rr_ratio': r.get('rr_ratio'),
            'max_dd_pct': r.get('max_drawdown_pct'),
            'total_return_pct': r.get('total_return_pct'),
            'total_pnl': r.get('total_pnl'),
            'pyramid_rate_pct': r.get('pyramid_trade_rate_pct'),
            'passed': r.get('passed'),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'backtest_maedai_ranking.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"[保存] {path}")

    # 上位N戦略のトレード履歴
    for r in results[:top_n]:
        name = r['strategy']
        trades = trade_map.get(name, [])
        if not trades:
            continue
        rows2 = []
        for t in trades:
            rows2.append({
                'entry_time': t['entry_time'],
                'exit_time': t['exit_time'],
                'direction': t['direction'],
                'entry': t['entry_price'],
                'exit': t['exit_price'],
                'sl': t['sl'],
                'tp': t['tp'],
                'layers': t.get('pyramid_layers', 1),
                'pnl': t['pnl'],
                'exit_reason': t['exit_reason'],
            })
        safe = name.replace('/', '-').replace(' ', '_')
        p2 = os.path.join(RESULTS_DIR, f'trade_history_maedai_{safe}.csv')
        pd.DataFrame(rows2).to_csv(p2, index=False, encoding='utf-8-sig')
        print(f"[保存] {p2}")


def _generate_pnl_image(name, trades, data_source):
    """PnLチャート生成"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0d0d1a')
        fig.suptitle(f'マエダイメソッド Best: {name}  (source: {data_source})',
                     color='white', fontsize=13)

        # 累積PnL
        ax1 = axes[0]
        ax1.set_facecolor('#111130')
        running = 0
        cumul = [0]
        times = [pd.Timestamp(trades[0]['entry_time'])]
        for t in trades:
            running += t['pnl']
            cumul.append(running)
            times.append(pd.Timestamp(t['exit_time']))
        color = '#00ff88' if running >= 0 else '#ff4444'
        ax1.plot(times, cumul, color=color, lw=1.8)
        ax1.fill_between(times, 0, cumul, alpha=0.2, color=color)
        ax1.axhline(0, color='gray', lw=0.5, ls='--')
        ax1.set_title('Cumulative PnL', color='white')
        ax1.tick_params(colors='#aaa')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        for sp in ax1.spines.values():
            sp.set_color('#334')

        # 個別トレードPnL (棒グラフ)
        ax2 = axes[1]
        ax2.set_facecolor('#111130')
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        ax2.bar(range(len(trades)), pnls,
                color=['#00ff88' if p > 0 else '#ff4444' for p in pnls], alpha=0.8)
        ax2.axhline(0, color='gray', lw=0.5)
        ax2.set_title(f'Trade PnL  W:{len(wins)} / L:{len(losses)}', color='white')
        ax2.tick_params(colors='#aaa')
        for sp in ax2.spines.values():
            sp.set_color('#334')

        # ピラミッドレイヤー分布
        ax3 = axes[2]
        ax3.set_facecolor('#111130')
        layers = [t.get('pyramid_layers', 1) for t in trades]
        max_l = max(layers) if layers else 1
        layer_counts = [layers.count(k) for k in range(1, max_l + 2)]
        ax3.bar(range(1, len(layer_counts) + 1), layer_counts,
                color='#4488ff', alpha=0.8)
        ax3.set_title('Pyramid Layers Distribution', color='white')
        ax3.set_xlabel('Layers', color='#aaa')
        ax3.tick_params(colors='#aaa')
        for sp in ax3.spines.values():
            sp.set_color('#334')

        plt.tight_layout()
        safe = name.replace('/', '-').replace('(', '').replace(')', '').replace(' ', '_')
        fpath = os.path.join(IMAGES_DIR, f'bt_maedai_{safe}.png')
        plt.savefig(fpath, dpi=130, bbox_inches='tight', facecolor='#0d0d1a')
        plt.close()
        print(f"[画像] {fpath}")
    except Exception as e:
        print(f"[画像] スキップ: {e}")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='マエダイメソッド バックテスト')
    parser.add_argument('--dukascopy', action='store_true', help='Dukascopyデータを使用')
    parser.add_argument('--risk-pct', type=float, default=0.02,
                        help='1トレードリスク比率 (デフォルト: 0.02 = 2%%)')
    parser.add_argument('--top-n', type=int, default=3,
                        help='トレード履歴を表示する上位N戦略 (デフォルト: 3)')
    parser.add_argument('--no-image', action='store_true', help='画像生成スキップ')
    args = parser.parse_args()

    print('=' * 70)
    print('マエダイメソッド バックテスト')
    print('「背を近くして何度も挑戦、大きい値動きを取る」')
    print(f'リスク設定: {args.risk_pct*100:.1f}% / トレード  '
          f'SL=ATR×0.8  TP=ATR×10  Trail@3ATR')
    print('=' * 70)

    # データ
    data = load_data(use_dukascopy=args.dukascopy)
    data_source = data['source']
    print(f"データソース: {data_source}\n")

    # 戦略
    strategies = build_strategies()
    print(f"戦略数: {len(strategies)}")

    # バックテスト
    print("バックテスト実行中...")
    results, trade_map = run_backtest(data, strategies, risk_pct=args.risk_pct)

    # 結果表示
    print_ranking(results)

    # 上位N戦略のトレード履歴
    top_n = min(args.top_n, len(results))
    for r in results[:top_n]:
        name = r['strategy']
        trades = trade_map.get(name, [])
        if trades:
            print_trade_history(name, trades)

    # 保存
    if results:
        save_results(results, trade_map, top_n=top_n)

    # チャート
    if not args.no_image and results:
        best_name = results[0]['strategy']
        best_trades = trade_map.get(best_name, [])
        if best_trades:
            _generate_pnl_image(best_name, best_trades, data_source)

    # サマリー表示
    print()
    print('=' * 70)
    print('【マエダイメソッド 設計思想の確認】')
    print('  - 1トレード損失は小さく (ATR×0.8 で即カット)')
    print('  - 何度もエントリーして大きいブレイクを待つ')
    print('  - 含み益 3ATR → トレーリング発動、以降は利益を伸ばすだけ')
    print('  - 含み益 2ATR ごとに同量ピラミッド (最大4倍ポジ)')
    print('  - 勝率30%でも RR1:10 なら期待値 = 0.3×10 - 0.7 = 2.3 (プラス)')
    print()
    if results:
        best = results[0]
        print(f"  最優秀: {best['strategy']}")
        print(f"    勝率={best.get('win_rate_pct')}%  PF={best.get('profit_factor')}  "
              f"RR={best.get('rr_ratio')}  MaxDD={best.get('max_drawdown_pct')}%")
        print(f"    総リターン={best.get('total_return_pct')}%  "
              f"PnL={best.get('total_pnl'):,.0f}")
    print('=' * 70)


if __name__ == '__main__':
    main()
