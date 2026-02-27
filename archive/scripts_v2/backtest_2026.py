"""
2026年バックテスト (2026-01-01 〜 2026-02-26)
================================================
データソース戦略（優先順）:
  1. OANDA v20 API + Dukascopy spread injection (ハイブリッド・最高精度)
  2. Dukascopy tick → OHLC変換
  3. 2026年期間のサンプルデータ（フォールバック）

実行例:
  # サンプルデータでテスト
  python scripts/backtest_2026.py

  # OANDA APIキーがある場合（推奨）
  OANDA_API_KEY=your-key python scripts/backtest_2026.py --oanda

  # DukascopyデータのみでOHLC変換
  python scripts/backtest_2026.py --dukascopy
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
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

# ===== バックテスト対象期間 =====
START_DT = datetime(2026, 1, 1)
END_DT   = datetime(2026, 2, 26, 23, 59)


# ──────────────────────────────────────────────
# 1. データ取得
# ──────────────────────────────────────────────

def fetch_dukascopy_spread_stats(symbol='XAUUSD', days=28):
    """
    Dukascopy tickから時間帯別スプレッド統計を取得。
    Returns: dict {hour(int): avg_spread_in_price} or None
    """
    from scripts.fetch_data import fetch_ticks, load_ticks, ticks_to_ohlc
    ticks = load_ticks(symbol)
    if ticks is None or len(ticks) < 1000:
        print("[Hybrid] Dukascopyキャッシュなし → スプレッド取得スキップ")
        return None

    if 'askPrice' not in ticks.columns or 'bidPrice' not in ticks.columns:
        return None

    spread_raw = ticks['askPrice'] - ticks['bidPrice']
    spread_raw = spread_raw[spread_raw > 0]
    by_hour = spread_raw.groupby(spread_raw.index.hour).median()
    stats = by_hour.to_dict()
    print(f"[Hybrid] Dukascopyスプレッド統計: {len(stats)} 時間帯 "
          f"(avg {np.mean(list(stats.values())):.4f})")
    return stats


def inject_spread(bars, spread_stats):
    """
    OANDA mid-OHLCにDukascopyスプレッド統計を注入。
    entry: mid + spread/2 (ask相当), exit: mid - spread/2 (bid相当)
    → bars['spread'] 列を追加/上書き
    """
    if spread_stats is None:
        default_spread = 0.30  # 30pips = 0.30 USD for XAUUSD
        bars = bars.copy()
        bars['spread'] = default_spread
        return bars

    spreads = bars.index.hour.map(lambda h: spread_stats.get(h, np.mean(list(spread_stats.values()))))
    bars = bars.copy()
    bars['spread'] = spreads
    return bars


def load_data_hybrid(use_oanda=False, use_dukascopy=False, oanda_key=None, oanda_account='practice'):
    """
    データ取得優先順位:
    1. OANDA (--oanda フラグ時) + Dukascopyスプレッド
    2. Dukascopy tick
    3. 2026年サンプルデータ
    """
    from scripts.fetch_data import (
        load_ticks, ticks_to_ohlc,
        fetch_oanda_long, load_oanda_csv,
        generate_sample_ohlc, OANDA_GRANULARITY_MAP,
    )

    spread_stats = fetch_dukascopy_spread_stats()

    # ── 1. OANDA ──
    if use_oanda:
        key = oanda_key or os.environ.get('OANDA_API_KEY')
        if key:
            print("[Data] OANDA API から2026年データを取得中...")
            # 2025-12-15 から取得（インジケーター計算のウォームアップ期間含む）
            warmup_start = datetime(2025, 12, 15)
            delta_days = (END_DT - warmup_start).days + 2

            for gran, freq in [('H1', '1h'), ('H4', '4h')]:
                df = load_oanda_csv(f'XAUUSD_{gran}', gran)
                if df is not None and df.index.max() >= pd.Timestamp('2026-02-20'):
                    print(f"[OANDA] キャッシュ使用: {gran} ({len(df)} bars)")
                else:
                    df = fetch_oanda_long('XAU_USD', gran, days=delta_days,
                                         api_key=key, account_type=oanda_account)

            bars_1h = load_oanda_csv('XAUUSD_H1', 'H1')
            bars_4h = load_oanda_csv('XAUUSD_H4', 'H4')

            if bars_1h is not None and len(bars_1h) > 200:
                bars_1h = inject_spread(bars_1h, spread_stats)
                bars_4h = inject_spread(bars_4h, spread_stats) if bars_4h is not None else None
                # 期間絞り込み（ウォームアップ用に少し前から）
                warmup = pd.Timestamp('2025-12-15')
                end_ts = pd.Timestamp(END_DT)
                bars_1h = bars_1h[bars_1h.index >= warmup]
                bars_1h = bars_1h[bars_1h.index <= end_ts]
                if bars_4h is not None:
                    bars_4h = bars_4h[bars_4h.index >= warmup]
                    bars_4h = bars_4h[bars_4h.index <= end_ts]
                print(f"[Data] OANDA+Spread ハイブリッド: 1h={len(bars_1h)}, "
                      f"4h={len(bars_4h) if bars_4h is not None else 0} bars")
                return {'source': 'oanda+spread', '1h': bars_1h, '4h': bars_4h}
        else:
            print("[OANDA] APIキーなし → スキップ")

    # ── 2. Dukascopy tick ──
    if use_dukascopy or not use_oanda:
        ticks = load_ticks()
        if ticks is not None and len(ticks) > 10000:
            ts_end = pd.Timestamp(END_DT)
            ts_start = pd.Timestamp('2025-12-15')
            ticks = ticks[(ticks.index >= ts_start) & (ticks.index <= ts_end)]
            if len(ticks) > 1000:
                bars_1h = ticks_to_ohlc(ticks, '1h')
                bars_4h = ticks_to_ohlc(ticks, '4h')
                print(f"[Data] Dukascopy tick: 1h={len(bars_1h)}, 4h={len(bars_4h)} bars")
                return {'source': 'dukascopy', '1h': bars_1h, '4h': bars_4h}

    # ── 3. サンプル（2026年期間） ──
    print("[Data] 実データなし → 2026年期間のサンプルデータを生成")
    # ウォームアップ含め2025-12-15〜2026-02-26 = 73日分
    n_1h = 73 * 24
    n_4h = 73 * 6
    bars_1h = _generate_2026_sample(n_1h, '1h')
    bars_4h = _generate_2026_sample(n_4h, '4h')
    bars_1h = inject_spread(bars_1h, spread_stats)
    bars_4h = inject_spread(bars_4h, spread_stats)
    print(f"[Data] サンプル: 1h={len(bars_1h)}, 4h={len(bars_4h)} bars")
    return {'source': 'sample_2026', '1h': bars_1h, '4h': bars_4h}


def _generate_2026_sample(n_bars, freq, seed=2026):
    """2026年実勢価格帯でサンプル生成（XAU ~2850-2950 USD想定）"""
    np.random.seed(seed)
    base_price = 2880.0  # 2026年初頭の想定金価格
    # トレンドありのリターン（微小上昇トレンド）
    drift = 0.00005
    returns = np.random.normal(drift, 0.0025, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    start = pd.Timestamp('2025-12-15')
    dates = pd.date_range(start, periods=n_bars, freq=freq)
    vol = np.abs(np.random.normal(0.0015, 0.0008, n_bars))

    bars = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 1, n_bars)) * vol),
        'low':  prices * (1 - np.abs(np.random.normal(0, 1, n_bars)) * vol),
        'close': prices * (1 + np.random.normal(0, 1, n_bars) * vol),
    }, index=dates)
    bars['high']  = bars[['open', 'high', 'close']].max(axis=1)
    bars['low']   = bars[['open', 'low', 'close']].min(axis=1)
    return bars


# ──────────────────────────────────────────────
# 2. 戦略一覧
# ──────────────────────────────────────────────

def build_strategies(data=None):
    """
    戦略リストを構築。
    data: load_data_hybrid() の返り値 (dict)。
          MTFカスケード戦略の構築に使用。None の場合はスタンドアロン版を使用。
    """
    from lib.yagami import (
        sig_yagami_A, sig_yagami_B,
        sig_yagami_reversal_only, sig_yagami_double_bottom,
        sig_yagami_pattern_break, sig_yagami_london_ny,
        sig_yagami_vol_regime, sig_yagami_trend_regime,
        sig_yagami_prime_time, sig_yagami_full_filter,
        sig_yagami_A_full_filter,
        sig_yagami_mtf_cascade, sig_yagami_mtf_4h_1h,
        sig_yagami_breakout, sig_yagami_breakout_filtered,
    )
    from lib.indicators import sig_sma, sig_rsi, sig_macd, sig_rsi_sma, sig_macd_rsi, sig_bb_rsi

    strats = [
        # Yagami コア
        ('YagamiA',            sig_yagami_A(),            '1h'),
        ('YagamiB',            sig_yagami_B(),            '1h'),
        ('Yagami_Reversal',    sig_yagami_reversal_only(),'1h'),
        ('Yagami_DblBottom',   sig_yagami_double_bottom(),'1h'),
        ('Yagami_PatternBrk',  sig_yagami_pattern_break(),'1h'),
        ('Yagami_LonNY',       sig_yagami_london_ny(),    '1h'),
        # 4h版
        ('YagamiA_4h',         sig_yagami_A('4h'),        '4h'),
        ('YagamiB_4h',         sig_yagami_B('4h'),        '4h'),
        ('Yagami_Reversal_4h', sig_yagami_reversal_only('4h'), '4h'),
        ('Yagami_LonNY_4h',    sig_yagami_london_ny('4h'),'4h'),
        # botter Advent Calendar フィルター統合
        ('Yagami_VolRegime',   sig_yagami_vol_regime(),   '1h'),
        ('Yagami_TrendRegime', sig_yagami_trend_regime(), '1h'),
        ('Yagami_PrimeTime',   sig_yagami_prime_time(),   '1h'),
        ('Yagami_FullFilter',  sig_yagami_full_filter(),  '1h'),
        ('Yagami_A_FullFilter',sig_yagami_A_full_filter(),'1h'),
        ('Yagami_FullFilter_4h', sig_yagami_full_filter('4h'), '4h'),
        ('Yagami_TrendRegime_4h', sig_yagami_trend_regime('4h'), '4h'),
        # MTFカスケード (4H方向 → 1H確認 → タイミング)
        ('MTF_4H1H',           sig_yagami_mtf_4h_1h('1h'), '1h'),
        # ブレイクアウト + レジサポ転換
        ('Breakout_1h',        sig_yagami_breakout('1h'), '1h'),
        ('Breakout_4h',        sig_yagami_breakout('4h'), '4h'),
        ('Breakout_Filtered',  sig_yagami_breakout_filtered('1h'), '1h'),
        # インジケーター
        ('SMA(5/20)',   sig_sma(5, 20),       '1h'),
        ('SMA(10/50)',  sig_sma(10, 50),      '1h'),
        ('RSI(14)',     sig_rsi(14, 30, 70),  '1h'),
        ('MACD(12/26)', sig_macd(12, 26, 9), '1h'),
        ('RSI14+SMA50', sig_rsi_sma(14,30,70,50), '1h'),
        ('MACD+RSI50',  sig_macd_rsi(12,26,9,14,50), '1h'),
        ('BB20+RSI',    sig_bb_rsi(20, 2.0, 14, 30, 70), '1h'),
    ]

    # MTFカスケード（外部4Hデータ版）: bars_4h があれば追加
    if data is not None:
        bars_4h = data.get('4h')
        bars_1h = data.get('1h')
        if bars_4h is not None and len(bars_4h) >= 50:
            strats.append((
                'MTF_Cascade_4H',
                sig_yagami_mtf_cascade({'4h': bars_4h}, min_grade='B'),
                '1h',
            ))
            strats.append((
                'MTF_Cascade_A',
                sig_yagami_mtf_cascade({'4h': bars_4h}, min_grade='A'),
                '1h',
            ))

    return strats


# ──────────────────────────────────────────────
# 3. バックテスト実行 + 集計
# ──────────────────────────────────────────────

def run_backtest_2026(data, strategies, min_trades=5):
    """全戦略のバックテストを実行。2026年期間に絞り込み。"""
    from lib.backtest import BacktestEngine

    engine = BacktestEngine(
        init_cash=5_000_000,
        risk_pct=0.05,
        use_dynamic_sl=True,
        pyramid_entries=2,
        pyramid_atr=1.0,
    )

    start_ts = pd.Timestamp(START_DT)
    end_ts   = pd.Timestamp(END_DT)

    results = []
    trade_map = {}   # name -> trades list

    bars_4h = data.get('4h')  # SLのスウィングピボット検出に使用

    for name, sig_fn, freq in strategies:
        bars = data.get(freq)
        if bars is None or len(bars) < 50:
            continue

        # htf_bars: 1H戦略は4Hバーで、4H戦略はそのまま（4Hバー自身）
        htf = bars_4h if freq == '1h' else None
        result = engine.run(bars, sig_fn, freq=freq, name=name, htf_bars=htf)
        if result is None:
            continue

        # 2026年以降のトレードのみに絞り込み
        all_trades = result.get('trades', [])
        trades_2026 = [
            t for t in all_trades
            if pd.Timestamp(t['entry_time']) >= start_ts
            and pd.Timestamp(t['exit_time']) <= end_ts
        ]

        if len(trades_2026) < min_trades:
            continue

        # 2026年のみで集計し直し
        wins  = [t for t in trades_2026 if t['pnl'] > 0]
        loss  = [t for t in trades_2026 if t['pnl'] <= 0]
        total_pnl = sum(t['pnl'] for t in trades_2026)
        pf = (sum(t['pnl'] for t in wins) /
              max(abs(sum(t['pnl'] for t in loss)), 1e-9))
        wr = len(wins) / len(trades_2026)

        summary = {
            'strategy':    name,
            'freq':        freq,
            'trades':      len(trades_2026),
            'wins':        len(wins),
            'losses':      len(loss),
            'win_rate':    round(wr * 100, 1),
            'pnl':         round(total_pnl, 0),
            'profit_factor': round(pf, 3),
            'avg_pnl':     round(total_pnl / len(trades_2026), 1),
        }
        results.append(summary)
        trade_map[name] = trades_2026

    # PF降順でソート
    results.sort(key=lambda x: x['profit_factor'], reverse=True)
    return results, trade_map


# ──────────────────────────────────────────────
# 4. 出力
# ──────────────────────────────────────────────

def print_ranking(results):
    if not results:
        print("\n[結果] 有効な戦略なし (トレード数不足)")
        return

    print(f"\n{'='*80}")
    print(f"2026年バックテスト結果 ({START_DT.strftime('%Y-%m-%d')} 〜 "
          f"{END_DT.strftime('%Y-%m-%d')})")
    print(f"{'='*80}")
    hdr = f"{'Rank':<5}{'Strategy':<25}{'TF':<5}{'Trades':>7}{'WinRate':>9}"
    hdr += f"{'PF':>8}{'PnL':>12}{'Avg':>9}"
    print(hdr)
    print('-' * 80)

    for rank, r in enumerate(results, 1):
        mark = ' ★' if r['profit_factor'] >= 1.5 else ''
        line = (f"{rank:<5}{r['strategy']:<25}{r['freq']:<5}{r['trades']:>7}"
                f"{r['win_rate']:>8.1f}%{r['profit_factor']:>8.3f}"
                f"{r['pnl']:>12,.0f}{r['avg_pnl']:>9,.0f}{mark}")
        print(line)
    print('-' * 80)
    passed = [r for r in results if r['profit_factor'] >= 1.5]
    print(f"合格戦略 (PF≥1.5): {len(passed)} / {len(results)}")


def print_trade_history(name, trades):
    print(f"\n{'='*90}")
    print(f"トレード履歴: {name}  ({len(trades)} trades)")
    print(f"{'='*90}")
    hdr = (f"{'#':>4}  {'Entry Time':<18}{'Exit Time':<18}{'Dir':<6}"
           f"{'Entry':>9}{'Exit':>9}{'SL':>9}{'TP':>9}"
           f"{'PnL':>10}{'Result':<12}{'Layers':>7}")
    print(hdr)
    print('-' * 90)

    running_pnl = 0.0
    for i, t in enumerate(trades, 1):
        entry_t = pd.Timestamp(t['entry_time']).strftime('%m-%d %H:%M')
        exit_t  = pd.Timestamp(t['exit_time']).strftime('%m-%d %H:%M')
        result  = 'WIN ' if t['pnl'] > 0 else 'LOSS'
        running_pnl += t['pnl']
        line = (f"{i:>4}  {entry_t:<18}{exit_t:<18}{t['direction']:<6}"
                f"{t['entry_price']:>9.2f}{t['exit_price']:>9.2f}"
                f"{t['sl']:>9.2f}{t['tp']:>9.2f}"
                f"{t['pnl']:>10,.1f}  {result:<10}"
                f"{t.get('pyramid_layers',1):>7}  [{t['exit_reason']}]")
        print(line)

    print('-' * 90)
    print(f"{'累計 PnL':>60}: {running_pnl:>12,.1f}")


def save_trade_history_csv(name, trades):
    rows = []
    for t in trades:
        rows.append({
            'entry_time':    t['entry_time'],
            'exit_time':     t['exit_time'],
            'direction':     t['direction'],
            'entry_price':   t['entry_price'],
            'exit_price':    t['exit_price'],
            'sl':            t['sl'],
            'tp':            t['tp'],
            'size':          t['size'],
            'pyramid_layers':t.get('pyramid_layers', 1),
            'pnl':           round(t['pnl'], 2),
            'pnl_pct':       round(t['pnl_pct'], 4),
            'exit_reason':   t['exit_reason'],
            'duration_h':    round(t['duration_sec'] / 3600, 1),
            'atr_at_entry':  round(t['atr_at_entry'], 4),
        })

    df = pd.DataFrame(rows)
    safe_name = name.replace('/', '-').replace('(', '').replace(')', '')
    fname = f"trade_history_2026_{safe_name}.csv"
    fpath = os.path.join(RESULTS_DIR, fname)
    df.to_csv(fpath, index=False)
    print(f"\n[保存] {fpath}")
    return fpath


def save_ranking_csv(results):
    df = pd.DataFrame(results)
    fpath = os.path.join(RESULTS_DIR, 'backtest_2026_ranking.csv')
    df.to_csv(fpath, index=False)
    print(f"[保存] {fpath}")


# ──────────────────────────────────────────────
# 5. 画像生成（任意）
# ──────────────────────────────────────────────

def _generate_image(best_name, best_trades, data_source):
    """ベスト戦略のPnL曲線を簡易生成"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 facecolor='#1a1a2e')
        fig.suptitle(f'2026 Best Strategy: {best_name}  (source: {data_source})',
                     color='white', fontsize=13, y=1.01)

        # ── 累積PnL ──
        ax1 = axes[0]
        ax1.set_facecolor('#16213e')
        cumulative = [0]
        times = [pd.Timestamp(best_trades[0]['entry_time'])]
        running = 0
        for t in best_trades:
            running += t['pnl']
            cumulative.append(running)
            times.append(pd.Timestamp(t['exit_time']))

        color = '#00ff88' if running >= 0 else '#ff4444'
        ax1.plot(times, cumulative, color=color, lw=2)
        ax1.fill_between(times, 0, cumulative,
                         alpha=0.2, color=color)
        ax1.axhline(0, color='gray', lw=0.5, linestyle='--')
        ax1.set_title('Cumulative PnL (pips)', color='white')
        ax1.tick_params(colors='#aaaaaa')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        for spine in ax1.spines.values():
            spine.set_color('#333355')

        # ── トレード分布 ──
        ax2 = axes[1]
        ax2.set_facecolor('#16213e')
        pnls = [t['pnl'] for t in best_trades]
        wins  = [p for p in pnls if p > 0]
        losses= [p for p in pnls if p <= 0]
        ax2.bar(range(len(best_trades)), pnls,
                color=['#00ff88' if p > 0 else '#ff4444' for p in pnls],
                alpha=0.8)
        ax2.axhline(0, color='gray', lw=0.5)
        ax2.set_title(f'Trade PnL  W:{len(wins)} / L:{len(losses)}', color='white')
        ax2.tick_params(colors='#aaaaaa')
        for spine in ax2.spines.values():
            spine.set_color('#333355')

        plt.tight_layout()
        safe_name = best_name.replace('/', '-').replace('(', '').replace(')', '')
        fpath = os.path.join(IMAGES_DIR, f'bt2026_best_{safe_name}.png')
        plt.savefig(fpath, dpi=130, bbox_inches='tight',
                    facecolor='#1a1a2e')
        plt.close()
        print(f"[画像] {fpath}")
    except Exception as e:
        print(f"[画像] 生成スキップ: {e}")


# ──────────────────────────────────────────────
# 6. メイン
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='2026年バックテスト')
    parser.add_argument('--oanda', action='store_true',
                        help='OANDA API でデータ取得（OANDA_API_KEY 環境変数 or --oanda-key 必須）')
    parser.add_argument('--oanda-key',     default=None)
    parser.add_argument('--oanda-account', default='practice',
                        choices=['practice', 'live'])
    parser.add_argument('--dukascopy', action='store_true',
                        help='Dukascopyキャッシュデータを使用')
    parser.add_argument('--min-trades', type=int, default=5,
                        help='集計に含む最低トレード数 (デフォルト: 5)')
    parser.add_argument('--top-n', type=int, default=3,
                        help='トレード履歴を出力する上位N戦略 (デフォルト: 3)')
    parser.add_argument('--no-image', action='store_true',
                        help='画像生成をスキップ')
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"sena3fx 2026年バックテスト")
    print(f"期間: {START_DT.strftime('%Y-%m-%d')} 〜 {END_DT.strftime('%Y-%m-%d')}")
    print(f"{'='*70}\n")

    # ── データ取得 ──
    data = load_data_hybrid(
        use_oanda=args.oanda,
        use_dukascopy=args.dukascopy,
        oanda_key=args.oanda_key,
        oanda_account=args.oanda_account,
    )
    data_source = data['source']
    print(f"\nデータソース: {data_source}")

    # ── 戦略ロード ──
    print("\n戦略ロード中...")
    strategies = build_strategies(data)  # MTFカスケード戦略のためdataを渡す
    print(f"  {len(strategies)} 戦略")

    # ── バックテスト ──
    print(f"\nバックテスト実行中 (min_trades={args.min_trades})...")
    results, trade_map = run_backtest_2026(data, strategies, min_trades=args.min_trades)

    # ── ランキング表示・保存 ──
    print_ranking(results)
    if results:
        save_ranking_csv(results)

    # ── 上位N戦略のトレード履歴 ──
    top_n = min(args.top_n, len(results))
    if top_n == 0:
        print("\n有効な戦略がありませんでした。")
        return

    print(f"\n{'='*70}")
    print(f"上位 {top_n} 戦略のトレード履歴")
    print(f"{'='*70}")

    best_name   = results[0]['strategy']
    best_trades = trade_map[best_name]

    for i in range(top_n):
        name   = results[i]['strategy']
        trades = trade_map[name]
        print_trade_history(name, trades)
        save_trade_history_csv(name, trades)
        if i == 0 and not args.no_image:
            _generate_image(name, trades, data_source)

    # ── サマリー ──
    print(f"\n{'='*70}")
    print(f"[完了] ベスト戦略: {best_name}")
    r0 = results[0]
    print(f"  PF: {r0['profit_factor']:.3f}  |  WinRate: {r0['win_rate']:.1f}%"
          f"  |  Trades: {r0['trades']}  |  PnL: {r0['pnl']:,.0f}")
    print(f"  データ: {data_source}")
    print(f"  結果CSV: results/backtest_2026_ranking.csv")


if __name__ == '__main__':
    main()
