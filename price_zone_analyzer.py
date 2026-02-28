"""
価格帯滞在時間ヒストグラム分析
================================
XAUUSDの4時間足データから「薄いゾーン」（滞在本数が少ない価格帯）を
定量的に可視化し、ボラ拡大が予想される価格帯を事前に把握する。

使い方:
    # スタンドアロン実行 (ヒストグラム画像 + thin_zones.json 生成)
    python price_zone_analyzer.py

    # モジュールとして利用
    from price_zone_analyzer import load_thin_zones, is_thin_zone
    zones = load_thin_zones()
    if is_thin_zone(current_price, zones):
        lot_size *= 0.7
        stop_distance *= 1.3
"""
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
#  設定                                                               #
# ------------------------------------------------------------------ #
BIN_SIZE           = 10.0    # 価格ビン幅 ($10)
THIN_PERCENTILE    = 15      # 下位15%以下のビン = 薄いゾーン
DEFAULT_DATA_PATH  = os.path.join(ROOT, 'data', 'ohlc', 'XAUUSD_2025_4h.csv')
THIN_ZONES_PATH    = os.path.join(ROOT, 'data', 'thin_zones.json')
HISTOGRAM_PATH     = os.path.join(ROOT, 'results', 'price_zone_histogram.png')


# ------------------------------------------------------------------ #
#  データ読み込み                                                     #
# ------------------------------------------------------------------ #
def load_ohlc_4h(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """XAUUSD 4H足データを読み込む"""
    df = pd.read_csv(path)
    try:
        dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df['datetime'])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)
    df['datetime'] = dt
    df = df.set_index('datetime').sort_index()
    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)


# ------------------------------------------------------------------ #
#  価格帯ヒストグラム計算                                             #
# ------------------------------------------------------------------ #
def compute_price_histogram(df: pd.DataFrame, bin_size: float = BIN_SIZE) -> pd.Series:
    """
    各価格帯ビンに何本のローソク足が滞在しているかをカウントする。

    「滞在」の定義: ローソク足の [low, high] レンジが
    ビン [bin_low, bin_high] と重複している場合にカウント。

    Args:
        df:       OHLC DataFrame
        bin_size: ビン幅 ($)

    Returns:
        pd.Series: index=ビン下限価格 (float), value=滞在本数
    """
    price_min = np.floor(df['low'].min()  / bin_size) * bin_size
    price_max = np.ceil(df['high'].max()  / bin_size) * bin_size

    bins = np.arange(price_min, price_max + bin_size, bin_size)
    counts = np.zeros(len(bins) - 1, dtype=int)

    lows  = df['low'].values
    highs = df['high'].values

    for i, (bl, bh) in enumerate(zip(bins[:-1], bins[1:])):
        # bar.low <= bin_high AND bar.high >= bin_low → 重複
        overlap = (lows <= bh) & (highs >= bl)
        counts[i] = overlap.sum()

    return pd.Series(counts, index=bins[:-1], name='bar_count')


# ------------------------------------------------------------------ #
#  薄いゾーン検出                                                     #
# ------------------------------------------------------------------ #
def detect_thin_zones(histogram: pd.Series,
                      bin_size: float = BIN_SIZE,
                      percentile: float = THIN_PERCENTILE) -> list:
    """
    滞在本数が下位 percentile% 以下のビンを「薄いゾーン」として返す。

    Returns:
        list of dict: [{'low': float, 'high': float, 'bar_count': int}, ...]
    """
    threshold = np.percentile(histogram[histogram > 0].values, percentile)
    thin = histogram[(histogram > 0) & (histogram <= threshold)]

    zones = []
    for bin_low, count in thin.items():
        zones.append({
            'low':       float(bin_low),
            'high':      float(bin_low + bin_size),
            'bar_count': int(count),
        })

    # 連続するビンをマージ
    zones.sort(key=lambda x: x['low'])
    merged = []
    for z in zones:
        if merged and z['low'] <= merged[-1]['high'] + 1e-6:
            merged[-1]['high']      = max(merged[-1]['high'], z['high'])
            merged[-1]['bar_count'] += z['bar_count']
        else:
            merged.append(dict(z))

    return merged


# ------------------------------------------------------------------ #
#  JSON保存・読み込み                                                 #
# ------------------------------------------------------------------ #
def save_thin_zones(zones: list, path: str = THIN_ZONES_PATH) -> None:
    """薄いゾーンをJSONに保存する"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'bin_size':     BIN_SIZE,
        'percentile':   THIN_PERCENTILE,
        'zone_count':   len(zones),
        'zones':        zones,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"thin_zones.json 保存: {path}  ({len(zones)} ゾーン)")


def load_thin_zones(path: str = THIN_ZONES_PATH) -> list:
    """
    thin_zones.json を読み込んで zones リストを返す。
    ファイルがない場合は空リストを返す。
    """
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('zones', [])


# ------------------------------------------------------------------ #
#  エントリーフィルター関数                                          #
# ------------------------------------------------------------------ #
def is_thin_zone(current_price: float, thin_zones: list) -> bool:
    """
    現在価格が薄いゾーンにいるか判定する。

    Args:
        current_price: 現在の価格 (USD)
        thin_zones:    load_thin_zones() の出力

    Returns:
        bool: True = 薄いゾーン（ボラ拡大モード）
    """
    for zone in thin_zones:
        if zone['low'] <= current_price <= zone['high']:
            return True
    return False


def get_thin_zone_params(current_price: float, thin_zones: list,
                         lot_scale: float = 0.7,
                         stop_scale: float = 1.3,
                         allow_chase: bool = True) -> dict:
    """
    現在価格が薄いゾーンにいる場合のパラメータ調整値を返す。

    Returns:
        dict: {
            'is_thin':          bool,
            'lot_scale':        float,   # ロットに掛ける倍率
            'stop_scale':       float,   # ストップに掛ける倍率
            'allow_chase_entry': bool,   # 後乗りエントリー許可
        }
    """
    in_thin = is_thin_zone(current_price, thin_zones)
    return {
        'is_thin':           in_thin,
        'lot_scale':         lot_scale if in_thin else 1.0,
        'stop_scale':        stop_scale if in_thin else 1.0,
        'allow_chase_entry': allow_chase if in_thin else False,
    }


# ------------------------------------------------------------------ #
#  ヒストグラムチャート生成                                          #
# ------------------------------------------------------------------ #
def plot_histogram(df: pd.DataFrame,
                   histogram: pd.Series,
                   thin_zones: list,
                   bin_size: float = BIN_SIZE,
                   outpath: str = HISTOGRAM_PATH) -> None:
    """価格帯ヒストグラムと薄いゾーンを可視化する"""

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor('#0d1117')

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.40, wspace=0.35,
        top=0.93, bottom=0.04, left=0.07, right=0.97,
    )

    GOLD   = '#FFD700'
    RED    = '#f85149'
    BLUE   = '#58a6ff'
    GREEN  = '#26a641'
    ORANGE = '#f0a500'

    current_price = df['close'].iloc[-1]
    percentile_threshold = np.percentile(
        histogram[histogram > 0].values, THIN_PERCENTILE
    )

    # タイトル
    fig.text(
        0.5, 0.965,
        f'XAUUSD Price Zone Dwell Histogram\n'
        f'Period: {df.index[0].date()} to {df.index[-1].date()}  '
        f'/ 4H  / Bin: USD{bin_size:.0f}  '
        f'/ Thin zone: bottom {THIN_PERCENTILE}%  '
        f'/ Current: USD{current_price:,.0f}',
        ha='center', va='top', color='white',
        fontsize=12, fontweight='bold', linespacing=1.6,
    )

    # --- [0, :] 横型ヒストグラム (Market Profile) ---
    ax_hp = fig.add_subplot(gs[0, :])
    ax_hp.set_facecolor('#161b22')
    ax_hp.set_title(
        'Market Profile: Price Zone Dwell Count (4H bars)',
        color='white', fontsize=11, pad=6,
    )

    prices  = histogram.index.values
    counts  = histogram.values
    bar_colors = [
        RED if c <= percentile_threshold and c > 0
        else (BLUE if c > 0 else '#2d333b')
        for c in counts
    ]

    ax_hp.barh(prices, counts, height=bin_size * 0.9,
               color=bar_colors, alpha=0.85, align='edge')
    ax_hp.axhline(current_price, color=GOLD, lw=2.0, ls='--',
                  label=f'Current USD{current_price:,.0f}', alpha=0.9)
    ax_hp.axvline(percentile_threshold, color=RED, lw=1.2, ls=':',
                  alpha=0.8, label=f'Thin threshold ({THIN_PERCENTILE}pct = {percentile_threshold:.0f} bars)')

    # 薄いゾーン帯をハイライト
    for z in thin_zones:
        ax_hp.axhspan(z['low'], z['high'], color=RED, alpha=0.08)

    ax_hp.set_xlabel('Dwell count (4H bars)', color='white', fontsize=9)
    ax_hp.set_ylabel('Price (USD)', color='white', fontsize=9)
    ax_hp.tick_params(colors='white', labelsize=8)
    ax_hp.spines[:].set_color('#30363d')
    ax_hp.legend(fontsize=8.5, facecolor='#161b22', edgecolor='#30363d',
                 labelcolor='white', loc='lower right')

    # --- [1, 0] 縦型ヒストグラム (時系列視点) ---
    ax_vt = fig.add_subplot(gs[1, 0])
    ax_vt.set_facecolor('#161b22')
    ax_vt.set_title('Price Zone Density (vertical)', color='white', fontsize=10, pad=5)

    bar_colors_v = [
        RED if c <= percentile_threshold and c > 0
        else (BLUE if c > 0 else '#2d333b')
        for c in counts
    ]
    ax_vt.bar(prices, counts, width=bin_size * 0.9,
              color=bar_colors_v, alpha=0.85, align='edge')
    ax_vt.axhline(percentile_threshold, color=RED, lw=1.2, ls=':', alpha=0.8)

    for z in thin_zones:
        ax_vt.axvspan(z['low'], z['high'], color=RED, alpha=0.1)

    ax_vt.axvline(current_price, color=GOLD, lw=1.8, ls='--', alpha=0.9)
    ax_vt.set_xlabel('Price (USD)', color='white', fontsize=9)
    ax_vt.set_ylabel('Dwell count', color='white', fontsize=9)
    ax_vt.tick_params(colors='white', labelsize=8, axis='x', rotation=45)
    ax_vt.spines[:].set_color('#30363d')

    # --- [1, 1] 薄いゾーン一覧テーブル ---
    ax_tbl = fig.add_subplot(gs[1, 1])
    ax_tbl.set_facecolor('#161b22')
    ax_tbl.axis('off')
    ax_tbl.set_title(
        f'Thin Zones  (bottom {THIN_PERCENTILE}%  /  {len(thin_zones)} zones)',
        color='white', fontsize=10, pad=5,
    )

    # 現在価格に近い順でソート
    sorted_zones = sorted(thin_zones, key=lambda z: abs((z['low']+z['high'])/2 - current_price))
    rows = []
    for z in sorted_zones[:20]:  # 上位20件
        mid  = (z['low'] + z['high']) / 2
        dist = mid - current_price
        in_zone = '** HERE **' if z['low'] <= current_price <= z['high'] else f"{dist:+.0f}"
        rows.append([
            f"USD{z['low']:,.0f}-{z['high']:,.0f}",
            str(z['bar_count']),
            in_zone,
        ])

    tbl = ax_tbl.table(
        cellText=rows[:20],
        colLabels=['Price Range', 'Bars', 'Dist from current'],
        loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#30363d')
        if r == 0:
            cell.set_facecolor('#0d1117')
            cell.set_text_props(color='#58a6ff', fontweight='bold')
        else:
            in_z_flag = '**' in (rows[r-1][2] if r-1 < len(rows) else '')
            cell.set_facecolor('#f8514922' if in_z_flag else '#1c2128')
            cell.set_text_props(color='#f85149' if in_z_flag else 'white')

    # --- [2, :] 価格チャート + 薄いゾーンオーバーレイ ---
    ax_ch = fig.add_subplot(gs[2, :])
    ax_ch.set_facecolor('#161b22')
    ax_ch.set_title('XAUUSD Chart + Thin Zone Overlay', color='white', fontsize=10, pad=5)

    # 簡易ローソク足 (close ラインで代替)
    ax_ch.plot(df.index, df['close'], color=GOLD, lw=1.0, alpha=0.9, label='Close')
    ax_ch.fill_between(df.index, df['low'], df['high'], color=GOLD, alpha=0.08)

    # 薄いゾーンをハイライト
    for z in thin_zones:
        ax_ch.axhspan(z['low'], z['high'], color=RED, alpha=0.15,
                      label='Thin zone' if z == thin_zones[0] else '')

    ax_ch.set_ylabel('Price (USD)', color='white', fontsize=9)
    ax_ch.tick_params(colors='white', labelsize=8)
    ax_ch.spines[:].set_color('#30363d')
    ax_ch.legend(fontsize=8.5, facecolor='#161b22', edgecolor='#30363d',
                 labelcolor='white', loc='upper left')

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"ヒストグラム画像保存: {outpath}")


# ------------------------------------------------------------------ #
#  メイン処理                                                         #
# ------------------------------------------------------------------ #
def run_analysis(data_path: str = DEFAULT_DATA_PATH,
                 bin_size: float = BIN_SIZE,
                 percentile: float = THIN_PERCENTILE) -> list:
    """
    価格帯分析を実行し、thin_zones.json を保存する。

    Returns:
        list: 薄いゾーンのリスト
    """
    print("=" * 60)
    print("  XAUUSD 価格帯滞在時間ヒストグラム分析")
    print("=" * 60)

    # データ読み込み
    df = load_ohlc_4h(data_path)
    print(f"\nデータ: {len(df)} 本  "
          f"({df.index[0].date()} 〜 {df.index[-1].date()})")
    print(f"価格レンジ: ${df['low'].min():,.0f} 〜 ${df['high'].max():,.0f}")

    # ヒストグラム計算
    print("\n価格帯別滞在本数を計算中...")
    hist = compute_price_histogram(df, bin_size)
    total_bins  = (hist > 0).sum()
    thin_thresh = np.percentile(hist[hist > 0].values, percentile)
    print(f"  ビン数 (滞在あり): {total_bins}")
    print(f"  薄いゾーン閾値 (下位{percentile}%): {thin_thresh:.1f} 本/ビン")

    # 薄いゾーン検出
    zones = detect_thin_zones(hist, bin_size, percentile)
    print(f"\n薄いゾーン: {len(zones)} ゾーン検出")
    for z in zones:
        print(f"  ${z['low']:>8,.0f} 〜 ${z['high']:>8,.0f}  "
              f"({z['bar_count']} 本)")

    # 現在価格チェック
    current = df['close'].iloc[-1]
    in_thin = is_thin_zone(current, zones)
    print(f"\n現在価格: ${current:,.2f}")
    print(f"薄いゾーン: {'★ 該当 (ボラ拡大モード推奨)' if in_thin else '非該当 (通常モード)'}")
    if in_thin:
        p = get_thin_zone_params(current, zones)
        print(f"  → ロット倍率: {p['lot_scale']:.1f}x  "
              f"ストップ倍率: {p['stop_scale']:.1f}x  "
              f"後乗り許可: {p['allow_chase_entry']}")

    # JSON保存
    save_thin_zones(zones, THIN_ZONES_PATH)

    # チャート生成
    plot_histogram(df, hist, zones, bin_size, HISTOGRAM_PATH)

    return zones


if __name__ == '__main__':
    run_analysis()
