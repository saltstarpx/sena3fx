"""
RUN-20260305-006 (案2): Maedai系シグナル深掘りバックテスト
DC30+EMA200を複数銘柄・複数時間足で検証
ハンドオフ文書の「真のアルファ源泉（WR=75%, MDD=3%）」を定量的に検証する
"""
import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib.backtest import BacktestEngine
from lib.yagami import sig_maedai_d1_dc30, sig_maedai_d1_dc_multi

DATA_DIR = os.path.join(ROOT, 'data', 'ohlc')
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_ohlc(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            return None
    return df.dropna(subset=['open', 'high', 'low', 'close'])

# ── テスト対象の設定 ──────────────────────────────────────────────────
CONFIGS = [
    # (ファイル名, 時間足, 戦略名, シグナル関数)
    ('XAUUSD_1d.csv',   '1d', 'DC30_EMA200_XAUUSD_D1',      sig_maedai_d1_dc30(30, 200)),
    ('XAUUSD_4h.csv',   '4h', 'DC30_EMA200_XAUUSD_4H',      sig_maedai_d1_dc30(30, 200)),
    ('XAUUSD_1d.csv',   '1d', 'DC30_EMA200_confirm_XAUUSD',  sig_maedai_d1_dc_multi(30, 200, confirm_close=True)),
    ('USDJPY_1d.csv',   '1d', 'DC30_EMA200_USDJPY_D1',       sig_maedai_d1_dc30(30, 200)),
    ('EURUSD_1d.csv',   '1d', 'DC30_EMA200_EURUSD_D1',       sig_maedai_d1_dc30(30, 200)),
    ('GBPUSD_1d.csv',   '1d', 'DC30_EMA200_GBPUSD_D1',       sig_maedai_d1_dc30(30, 200)),
    ('XAUUSD_1d.csv',   '1d', 'DC20_EMA200_XAUUSD_D1',       sig_maedai_d1_dc30(20, 200)),
    ('XAUUSD_1d.csv',   '1d', 'DC50_EMA200_XAUUSD_D1',       sig_maedai_d1_dc30(50, 200)),
    ('XAUUSD_1d.csv',   '1d', 'DC30_EMA100_XAUUSD_D1',       sig_maedai_d1_dc30(30, 100)),
]

# BacktestEngine設定（マエダイメソッド準拠: 高RR/低WR）
ENGINE_KWARGS = dict(
    init_cash=5_000_000,
    risk_pct=0.02,
    default_sl_atr=0.8,
    default_tp_atr=10.0,
    trail_start_atr=3.0,
    trail_dist_atr=1.5,
    pyramid_entries=0,
    use_dynamic_sl=True,
    sl_n_confirm=2,
    sl_min_atr=0.5,
    target_max_dd=0.30,
    target_min_wr=0.30,
    target_min_trades=10,
)

print("=" * 70)
print("RUN-20260305-006: Maedai系シグナル深掘りバックテスト")
print("DC30+EMA200 複数銘柄・複数パラメータ検証")
print("=" * 70)

results = []
for fname, freq, name, sig_func in CONFIGS:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [SKIP] {fname} が存在しません")
        continue

    df = load_ohlc(fpath)
    if df is None or len(df) < 250:
        print(f"  [SKIP] {fname} データ不足")
        continue

    engine = BacktestEngine(**ENGINE_KWARGS)
    try:
        res = engine.run(data=df, signal_func=sig_func, freq=freq, name=name)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        continue

    if res is None:
        print(f"  [SKIP] {name}: バックテスト結果なし")
        continue

    n = res.get('total_trades', 0)
    pf = res.get('profit_factor', 0)
    wr = res.get('win_rate_pct', 0)  # already in %
    mdd = res.get('max_drawdown_pct', 0)  # already in %
    total_ret = res.get('total_return_pct', 0)  # already in %
    sharpe = res.get('sharpe_ratio', res.get('rr_ratio', 0))
    avg_win = res.get('avg_win', 0)
    avg_loss = res.get('avg_loss', 0)
    rr = res.get('rr_ratio', abs(avg_win / avg_loss) if avg_loss != 0 else 0)

    passed = (pf >= 1.5 and wr >= 30 and mdd <= 20 and n >= 10)

    print(f"  {name}: PF={pf:.3f}, WR={wr:.1f}%, MDD={mdd:.1f}%, N={n}, RR={rr:.2f}, {'✓ PASS' if passed else '✗'}")
    results.append({
        'strategy': name,
        'symbol': fname.replace('.csv', ''),
        'freq': freq,
        'pf': round(pf, 3),
        'wr_pct': round(wr, 1),
        'mdd_pct': round(mdd, 1),
        'n_trades': n,
        'rr': round(rr, 2),
        'sharpe': round(sharpe, 3),
        'total_return_pct': round(total_ret, 2),
        'avg_win': round(avg_win, 0),
        'avg_loss': round(avg_loss, 0),
        'passed': passed,
    })

df_res = pd.DataFrame(results)
out_csv = os.path.join(RESULTS_DIR, 'run006_maedai_deep.csv')
df_res.to_csv(out_csv, index=False)
print(f"\n結果保存: {out_csv}")

print("\n" + "=" * 70)
print("【サマリー】")
print(f"{'戦略':<40} {'PF':>6} {'WR%':>6} {'MDD%':>6} {'N':>5} {'RR':>5} {'判定'}")
print("-" * 70)
for _, row in df_res.sort_values('pf', ascending=False).iterrows():
    mark = '✓ PASS' if row['passed'] else '✗'
    print(f"  {row['strategy']:<38} {row['pf']:>6.3f} {row['wr_pct']:>6.1f} {row['mdd_pct']:>6.1f} {row['n_trades']:>5} {row['rr']:>5.2f} {mark}")

passed_count = df_res['passed'].sum()
print(f"\n合格: {passed_count}/{len(df_res)} 戦略")
print(f"最良PF: {df_res['pf'].max():.3f} ({df_res.loc[df_res['pf'].idxmax(), 'strategy']})")
print(f"最高WR: {df_res['wr_pct'].max():.1f}% ({df_res.loc[df_res['wr_pct'].idxmax(), 'strategy']})")
print("=" * 70)
