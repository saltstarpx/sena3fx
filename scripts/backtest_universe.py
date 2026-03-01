"""
Union戦略 ユニバース拡張バックテスト
=====================================
Union_4H_Base 戦略を複数商品に適用し、有効な商品を特定する。

対象商品:
  ['XAGUSD', 'XAUUSD', 'NAS100USD', 'US30USD',
   'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']
  → データ(data/ohlc/<SYMBOL>_*_4h.csv)がない商品はスキップ。

出力:
  results/universe_performance.csv — 全商品バックテスト結果
  results/performance_log.csv     — 追記

実行:
  python scripts/backtest_universe.py
"""
import os
import sys
import glob
import csv
import datetime
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

from lib.backtest import BacktestEngine
from lib.yagami import sig_maedai_yagami_union

# ── 対象ユニバース ──────────────────────────────
UNIVERSE = ['XAGUSD', 'XAUUSD', 'NAS100USD', 'US30USD',
            'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']

# ── バックテストエンジン設定 ─────────────────────
ENGINE_CFG = dict(
    init_cash=5_000_000,
    risk_pct=0.05,
    default_sl_atr=2.0,
    default_tp_atr=4.0,
    pyramid_entries=0,
    target_max_dd=0.30,
    target_min_wr=0.30,
    target_rr_threshold=2.0,
    target_min_trades=5,
)

SIGNAL_PARAMS = dict(
    freq='4h', lookback_days=15, ema_days=200,
    confirm_bars=2, rsi_oversold=45,
)

OUTPUT_CSV = os.path.join(ROOT, 'results', 'universe_performance.csv')


# ── データ読み込み ───────────────────────────────

def find_ohlc_4h(symbol: str) -> str | None:
    """data/ohlc/ から <symbol>_*_4h.csv を探して最初のパスを返す。"""
    pat = os.path.join(ROOT, 'data', 'ohlc', f'{symbol}*4h.csv')
    matches = sorted(glob.glob(pat))
    if matches:
        return matches[-1]   # 最新ファイルを優先
    # アンダースコアなしも試す
    pat2 = os.path.join(ROOT, 'data', 'ohlc', f'{symbol}*4H.csv')
    matches2 = sorted(glob.glob(pat2))
    return matches2[-1] if matches2 else None


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    try:
        dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    df['datetime'] = dt
    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    return df.set_index('datetime').sort_index()[cols].astype(float)


def fmt(v, p=4):
    return '' if v is None or (isinstance(v, float) and v != v) else round(v, p)


# ── メイン ──────────────────────────────────────

def main():
    engine = BacktestEngine(**ENGINE_CFG)
    sig_func = sig_maedai_yagami_union(**SIGNAL_PARAMS)

    results = []
    print(f'{"商品":<12} {"Sharpe":>8} {"Calmar":>8} {"MDD%":>7} {"PF":>7} {"WR%":>6} {"Trades":>7} {"期間"}')
    print('-' * 75)

    for symbol in UNIVERSE:
        path = find_ohlc_4h(symbol)
        if path is None:
            print(f'{symbol:<12} — データなし (スキップ)')
            continue

        try:
            df = load_ohlc(path)
        except Exception as e:
            print(f'{symbol:<12} — 読み込みエラー: {e}')
            continue

        if len(df) < 100:
            print(f'{symbol:<12} — データ不足 ({len(df)}バー, スキップ)')
            continue

        period = f'{df.index[0].date()} ~ {df.index[-1].date()}'
        try:
            r = engine.run(data=df, signal_func=sig_func, freq='4h',
                           name=f'Union_{symbol}')
        except Exception as e:
            print(f'{symbol:<12} — バックテストエラー: {e}')
            continue

        sh = r.get('sharpe_ratio')
        ca = r.get('calmar_ratio')
        mdd = r.get('max_drawdown_pct')
        pf  = r.get('profit_factor')
        wr  = r.get('win_rate_pct')
        tr  = r.get('total_trades', 0)

        print(f'{symbol:<12} '
              f'{fmt(sh,3):>8} {fmt(ca,3):>8} {fmt(mdd,1):>7} '
              f'{fmt(pf,3):>7} {fmt(wr,1):>6} {tr:>7}  {period}')

        results.append({
            'symbol':       symbol,
            'data_path':    os.path.basename(path),
            'bars':         len(df),
            'period_start': str(df.index[0].date()),
            'period_end':   str(df.index[-1].date()),
            'sharpe_ratio': fmt(sh, 4),
            'calmar_ratio': fmt(ca, 4),
            'max_drawdown_pct': fmt(mdd, 2),
            'profit_factor': fmt(pf, 4),
            'win_rate_pct':  fmt(wr, 2),
            'total_trades':  tr,
        })

    print('-' * 75)
    print(f'合計 {len(results)} 商品を評価')

    # ── universe_performance.csv 書き出し ──
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f'\n結果保存: {OUTPUT_CSV}')
    else:
        print('\n有効な結果がありません。')

    # ── performance_log.csv 追記 ──
    log_path = os.path.join(ROOT, 'results', 'performance_log.csv')
    write_header = not os.path.exists(log_path)
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'strategy_name', 'parameters',
                             'timeframe', 'sharpe_ratio', 'profit_factor',
                             'max_drawdown', 'win_rate', 'trades'])
        for row in results:
            writer.writerow([
                ts,
                f'Union_{row["symbol"]}',
                'v15_universe',
                '4H',
                row['sharpe_ratio'],
                row['profit_factor'],
                row['max_drawdown_pct'],
                row['win_rate_pct'],
                row['total_trades'],
            ])
    print(f'performance_log.csv 追記: {log_path}')

    return results


if __name__ == '__main__':
    main()
