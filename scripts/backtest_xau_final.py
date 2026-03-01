"""
XAUUSD + Kelly + リバモア式ピラミッティング 統合バックテスト (v16 Task 3)
==========================================================================
Union_4H_Base に LivermorePyramidingSizer(base=KellyCriterionSizer) を組み合わせる。

構成:
  - シグナル: Union_4H_Base (sig_maedai_yagami_union)
  - 初期サイズ: KellyCriterionSizer(f=0.25)
  - 追加エントリー: LivermorePyramidingSizer
      step_pct=0.01 (1%上昇ごとに追加)
      pyramid_ratios=[0.5, 0.3, 0.2] (初期ロットの50%→30%→20%)
      max_pyramids=3

比較:
  1. Union_XAUUSD_Base         : サイジングなし・ピラミッドなし
  2. XAUUSD+Kelly(f=0.25)      : Kellyのみ
  3. XAUUSD+Kelly+Pyramid(LV)  : Kelly + リバモア式ピラミッティング

実行:
  python scripts/backtest_xau_final.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import glob
import pandas as pd

from lib.backtest import BacktestEngine
from lib.yagami import sig_maedai_yagami_union
from lib.risk_manager import KellyCriterionSizer, LivermorePyramidingSizer

SYMBOL = 'XAUUSD'
KELLY_FRACTION = 0.25
STEP_PCT = 0.01
PYRAMID_RATIOS = [0.5, 0.3, 0.2]
MAX_PYRAMIDS = 3

ENGINE_CFG = dict(
    init_cash=5_000_000,
    risk_pct=0.05,
    default_sl_atr=2.0,
    default_tp_atr=4.0,
    pyramid_entries=0,      # ATRベース built-in ピラミッドは無効 (Livermoreで代替)
    target_max_dd=0.30,
    target_min_wr=0.30,
    target_rr_threshold=2.0,
    target_min_trades=5,
)

SIGNAL_PARAMS = dict(
    freq='4h', lookback_days=15, ema_days=200,
    confirm_bars=2, rsi_oversold=45,
)


def find_ohlc_4h(symbol: str) -> str | None:
    pat = os.path.join(ROOT, 'data', 'ohlc', f'{symbol}*4h.csv')
    matches = sorted(glob.glob(pat))
    return matches[-1] if matches else None


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


def fmt(v, p=3):
    return 'N/A' if v is None or (isinstance(v, float) and v != v) else f'{v:.{p}f}'


def main():
    path = find_ohlc_4h(SYMBOL)
    if path is None:
        print(f'データなし: {SYMBOL}')
        sys.exit(1)

    df = load_ohlc(path)
    print(f'[{SYMBOL}] {len(df)} バー, {df.index[0].date()} ~ {df.index[-1].date()}')
    print(f'設定: Kelly f={KELLY_FRACTION}, ピラミッド step={STEP_PCT:.0%}, '
          f'ratios={PYRAMID_RATIOS}, max={MAX_PYRAMIDS}回')

    engine = BacktestEngine(**ENGINE_CFG)
    sig_func = sig_maedai_yagami_union(**SIGNAL_PARAMS)

    results = []

    # ── 1: ベースライン ──
    print('\n[1/3] Union_XAUUSD_Base...', end='', flush=True)
    r_base = engine.run(data=df, signal_func=sig_func, freq='4h',
                        name='Union_XAUUSD_Base')
    results.append(('Union_XAUUSD_Base', r_base))
    print(f' Sharpe={fmt(r_base.get("sharpe_ratio"))}, '
          f'MDD={fmt(r_base.get("max_drawdown_pct"),1)}%, '
          f'Trades={r_base.get("total_trades",0)}')

    # ── 2: Kelly のみ ──
    kelly_only = KellyCriterionSizer(
        win_rate=r_base.get('win_rate_pct', 50) / 100,
        profit_factor=r_base.get('profit_factor', 2.0),
        kelly_fraction=KELLY_FRACTION,
        base_risk_pct=ENGINE_CFG['risk_pct'],
    )
    name_k = f'XAUUSD+Kelly(f={KELLY_FRACTION})'
    print(f'[2/3] {name_k}...', end='', flush=True)
    r_kelly = engine.run(data=df, signal_func=sig_func, freq='4h',
                         name=name_k, sizer=kelly_only)
    results.append((name_k, r_kelly))
    print(f' Sharpe={fmt(r_kelly.get("sharpe_ratio"))}, '
          f'MDD={fmt(r_kelly.get("max_drawdown_pct"),1)}%, '
          f'Trades={r_kelly.get("total_trades",0)}')

    # ── 3: Kelly + Livermore ピラミッティング ──
    # KellyCriterionSizer を base_sizer として使用
    kelly_base = KellyCriterionSizer(
        win_rate=r_base.get('win_rate_pct', 50) / 100,
        profit_factor=r_base.get('profit_factor', 2.0),
        kelly_fraction=KELLY_FRACTION,
        base_risk_pct=ENGINE_CFG['risk_pct'],
    )
    livermore = LivermorePyramidingSizer(
        base_sizer=kelly_base,
        step_pct=STEP_PCT,
        pyramid_ratios=PYRAMID_RATIOS,
        max_pyramids=MAX_PYRAMIDS,
    )
    print(f'\nLivermoreSizer: {livermore}')

    name_lv = 'XAUUSD+Kelly+Pyramid(LV)'
    print(f'[3/3] {name_lv}...', end='', flush=True)
    r_lv = engine.run(data=df, signal_func=sig_func, freq='4h',
                      name=name_lv, sizer=livermore)
    results.append((name_lv, r_lv))
    print(f' Sharpe={fmt(r_lv.get("sharpe_ratio"))}, '
          f'MDD={fmt(r_lv.get("max_drawdown_pct"),1)}%, '
          f'Trades={r_lv.get("total_trades",0)}')

    # ── ピラミッド統計 ──
    lv_trades = r_lv.get('trades', [])
    pyramid_trades = [t for t in lv_trades if t.get('pyramid_layers', 1) > 1]
    print(f'\nピラミッティング統計 (Livermore):')
    print(f'  全{len(lv_trades)}トレード中 {len(pyramid_trades)}件でピラミッド発動 '
          f'({len(pyramid_trades)/max(len(lv_trades),1)*100:.1f}%)')
    if pyramid_trades:
        avg_layers = sum(t.get('pyramid_layers', 1) for t in pyramid_trades) / len(pyramid_trades)
        print(f'  平均レイヤー数: {avg_layers:.1f}')

    # ── 比較テーブル ──
    print('\n' + '=' * 90)
    print(f'  {SYMBOL} — Union vs Kelly vs Kelly+Livermore 統合比較')
    print('=' * 90)
    header = (f"{'戦略':<34} {'Sharpe':>8} {'Calmar':>8} {'MDD%':>7}"
              f" {'PF':>7} {'WR%':>6} {'Trades':>7} {'最終資産':>12}")
    print(header)
    print('-' * 90)

    for name, r in results:
        pf     = fmt(r.get('profit_factor'))
        wr     = fmt(r.get('win_rate_pct'), 1) if r.get('win_rate_pct') else 'N/A'
        mdd    = fmt(r.get('max_drawdown_pct'), 1) if r.get('max_drawdown_pct') else 'N/A'
        sharpe = fmt(r.get('sharpe_ratio'))
        calmar = fmt(r.get('calmar_ratio'))
        trades = r.get('total_trades', 0)
        end_v  = f"¥{r.get('end_value',0):,.0f}" if r.get('end_value') else 'N/A'
        print(f'{name:<34} {sharpe:>8} {calmar:>8} {mdd:>7} {pf:>7} {wr:>6} {trades:>7} {end_v:>12}')

    print('=' * 90)

    # ── 評価 ──
    base_sh = r_base.get('sharpe_ratio') or 0
    lv_sh   = r_lv.get('sharpe_ratio') or 0
    lv_mdd  = r_lv.get('max_drawdown_pct') or 0
    lv_ca   = r_lv.get('calmar_ratio') or 0

    print(f'\n[ピラミッティング効果評価]')
    print(f'  Sharpe: Base={base_sh:.3f} → Pyramid={lv_sh:.3f} ({lv_sh-base_sh:+.3f})')
    print(f'  MDD:    {r_base.get("max_drawdown_pct",0):.1f}% → {lv_mdd:.1f}% ({lv_mdd - (r_base.get("max_drawdown_pct") or 0):+.1f}pt)')
    print(f'  Calmar: {r_base.get("calmar_ratio",0):.3f} → {lv_ca:.3f}')
    end_val = r_lv.get('end_value', 0)
    base_end = r_base.get('end_value', ENGINE_CFG['init_cash'])
    print(f'  最終資産: ¥{base_end:,.0f} → ¥{end_val:,.0f} ({(end_val/base_end-1)*100:+.1f}%)')

    print(f'\nperformance_log.csv に記録済み (BacktestEngine 自動追記)')

    return results


if __name__ == '__main__':
    main()
