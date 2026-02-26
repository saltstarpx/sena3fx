"""
バックテスト結果ビジュアライザ
================================
1枚の画像にすべての主要指標をまとめる。

出力レイアウト (1400×900 px):
  ┌─────────────────────────────────────────┐
  │  エクイティカーブ + ドローダウン (60%)  │
  ├────────────┬───────────┬────────────────┤
  │ 月次リターン│ 勝敗分布  │ 主要指標テーブル│
  │  (棒グラフ) │ (円 +bar) │                │
  └────────────┴───────────┴────────────────┘
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless環境対応
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from datetime import datetime

# カラーパレット（ダークテーマ）
BG      = '#0f1117'
BG2     = '#1a1d27'
GREEN   = '#00c896'
RED     = '#ff4f5e'
BLUE    = '#3d8bff'
YELLOW  = '#ffd166'
GRAY    = '#4a4e69'
WHITE   = '#e0e0e0'
LGRAY   = '#9a9dc4'


def _equity_curve(trades: list, init_cash: float) -> tuple:
    """トレードリストからエクイティカーブとドローダウンを生成"""
    if not trades:
        return [], [], []

    cash = init_cash
    peak = init_cash
    equity = [init_cash]
    dd_pct = [0.0]
    times = [trades[0]['entry_time']]

    for t in trades:
        cash += t['pnl']
        equity.append(cash)
        times.append(t['exit_time'])
        if cash > peak:
            peak = cash
        dd_pct.append((peak - cash) / peak * 100 if peak > 0 else 0)

    return times, equity, dd_pct


def _monthly_returns(trades: list, init_cash: float) -> pd.Series:
    """月次リターン（%）を計算"""
    if not trades:
        return pd.Series(dtype=float)

    df = pd.DataFrame(trades)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['month'] = df['exit_time'].dt.to_period('M')

    # 月ごとにPnLを合計して月次リターンに変換
    monthly_pnl = df.groupby('month')['pnl'].sum()

    # 月初資産を推計（累積で計算）
    running_cash = init_cash
    monthly_ret = {}
    for period in sorted(monthly_pnl.index):
        pnl = monthly_pnl[period]
        ret = pnl / running_cash * 100
        monthly_ret[str(period)] = ret
        running_cash += pnl

    return pd.Series(monthly_ret)


def plot_backtest_report(result: dict,
                         output_path: str,
                         title: str = None) -> str:
    """
    バックテスト結果を1枚の画像に出力。

    Args:
        result: BacktestEngine.run() の戻り値
        output_path: 保存先パス (.png)
        title: グラフタイトル（省略時は strategy名）

    Returns:
        str: 保存先パス
    """
    trades = result.get('trades', [])
    init_cash = 5_000_000  # デフォルト
    # tradesから逆算 (total_pnlとend_valueから)
    if result.get('end_value'):
        init_cash = result['end_value'] - result.get('total_pnl', 0)

    times, equity, dd_pct = _equity_curve(trades, init_cash)
    monthly_rets = _monthly_returns(trades, init_cash)

    # ===== レイアウト =====
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.6, 1.0],
        hspace=0.38,
        wspace=0.32,
        left=0.06, right=0.97, top=0.91, bottom=0.07,
    )

    ax_equity = fig.add_subplot(gs[0, :])       # 上段: エクイティ（3列フル）
    ax_monthly = fig.add_subplot(gs[1, 0])       # 月次リターン
    ax_winloss = fig.add_subplot(gs[1, 1])       # 勝敗分布
    ax_metrics = fig.add_subplot(gs[1, 2])       # 指標テーブル

    for ax in [ax_equity, ax_monthly, ax_winloss, ax_metrics]:
        ax.set_facecolor(BG2)
        for spine in ax.spines.values():
            spine.set_color(GRAY)

    strategy = result.get('strategy', 'Strategy')
    passed = result.get('passed', False)
    pass_label = 'PASS' if passed else 'FAIL'
    pass_color = GREEN if passed else RED

    fig.suptitle(
        f"{title or strategy}",
        fontsize=15, color=WHITE, fontweight='bold', x=0.46, y=0.97,
    )
    fig.text(0.97, 0.97, pass_label, fontsize=15, color=pass_color,
             fontweight='bold', va='top', ha='right',
             transform=fig.transFigure)

    # ===== 1. エクイティカーブ =====
    if len(equity) > 1:
        x = range(len(equity))
        eq_arr = np.array(equity)
        dd_arr = np.array(dd_pct)

        # ドローダウン（塗りつぶし）- 第2軸
        ax_dd = ax_equity.twinx()
        ax_dd.fill_between(x, 0, dd_arr, alpha=0.25, color=RED, label='DD%')
        ax_dd.set_ylim(max(dd_arr) * 4, 0)  # 上が0%、下がマイナス
        ax_dd.set_ylabel('Drawdown %', color=RED, fontsize=8)
        ax_dd.tick_params(axis='y', colors=RED, labelsize=7)
        ax_dd.set_facecolor(BG2)
        for spine in ax_dd.spines.values():
            spine.set_color(GRAY)

        # エクイティ線
        color = GREEN if eq_arr[-1] >= eq_arr[0] else RED
        ax_equity.plot(x, eq_arr / 1e6, color=color, linewidth=1.8, zorder=3)
        ax_equity.fill_between(x, init_cash / 1e6, eq_arr / 1e6,
                               alpha=0.12, color=color)

        # 初期資金ライン
        ax_equity.axhline(init_cash / 1e6, color=GRAY, linewidth=0.8,
                          linestyle='--', alpha=0.6)

        ret_pct = result.get('total_return_pct', 0)
        ax_equity.set_title(
            f"Equity Curve    Return: {ret_pct:+.2f}%    "
            f"Max DD: {result.get('max_drawdown_pct', 0):.1f}%",
            fontsize=10, color=LGRAY, pad=4,
        )
        ax_equity.set_ylabel('Balance (M JPY)', color=WHITE, fontsize=9)
        ax_equity.tick_params(colors=WHITE, labelsize=8)
        ax_equity.yaxis.label.set_color(WHITE)

        # トレード数が多い場合は間引いて表示
        if len(times) > 1:
            step = max(1, len(times) // 8)
            tick_pos = list(range(0, len(times), step))
            tick_labels = [str(times[i])[:10] if i < len(times) else '' for i in tick_pos]
            ax_equity.set_xticks(tick_pos)
            ax_equity.set_xticklabels(tick_labels, rotation=20, ha='right',
                                      color=LGRAY, fontsize=7)
    else:
        ax_equity.text(0.5, 0.5, 'No data', ha='center', va='center',
                       color=LGRAY, fontsize=12, transform=ax_equity.transAxes)

    # ===== 2. 月次リターン棒グラフ =====
    ax_monthly.set_title('Monthly Return %', fontsize=10, color=WHITE, pad=4)
    if len(monthly_rets) > 0:
        colors = [GREEN if v >= 0 else RED for v in monthly_rets.values]
        bars = ax_monthly.bar(range(len(monthly_rets)), monthly_rets.values,
                              color=colors, alpha=0.85, width=0.7)
        ax_monthly.axhline(0, color=GRAY, linewidth=0.8)

        # ラベル
        step = max(1, len(monthly_rets) // 6)
        ax_monthly.set_xticks(range(0, len(monthly_rets), step))
        ax_monthly.set_xticklabels(
            [monthly_rets.index[i] for i in range(0, len(monthly_rets), step)],
            rotation=40, ha='right', color=LGRAY, fontsize=7,
        )
        ax_monthly.set_ylabel('%', color=WHITE, fontsize=8)
        ax_monthly.tick_params(colors=WHITE, labelsize=7)
        ax_monthly.yaxis.label.set_color(WHITE)

        # プラス/マイナス月の割合を表示
        pos_months = sum(1 for v in monthly_rets.values if v > 0)
        total_months = len(monthly_rets)
        ax_monthly.set_xlabel(
            f'Positive months: {pos_months}/{total_months} ({pos_months/total_months*100:.0f}%)',
            color=LGRAY, fontsize=8,
        )
    else:
        ax_monthly.text(0.5, 0.5, 'No data', ha='center', va='center',
                        color=LGRAY, fontsize=10, transform=ax_monthly.transAxes)

    # ===== 3. 勝敗分布 =====
    ax_winloss.set_title('Win/Loss by Direction', fontsize=10, color=WHITE, pad=4)
    wins = result.get('wins', 0)
    losses = result.get('losses', 0)
    total = wins + losses

    if total > 0:
        # 勝敗の水平積み上げバー
        ax_winloss.barh(['W/L'], [wins], color=GREEN, alpha=0.85, height=0.35, label=f'Win {wins}')
        ax_winloss.barh(['W/L'], [losses], left=[wins], color=RED, alpha=0.85, height=0.35, label=f'Loss {losses}')

        if trades:
            df_t = pd.DataFrame(trades)
            long_wins = len(df_t[(df_t['direction'] == 'long') & (df_t['pnl'] > 0)])
            long_loss = len(df_t[(df_t['direction'] == 'long') & (df_t['pnl'] <= 0)])
            short_wins = len(df_t[(df_t['direction'] == 'short') & (df_t['pnl'] > 0)])
            short_loss = len(df_t[(df_t['direction'] == 'short') & (df_t['pnl'] <= 0)])

            ax_winloss.barh(['Long'], [long_wins], color=GREEN, alpha=0.75, height=0.35)
            ax_winloss.barh(['Long'], [long_loss], left=[long_wins], color=RED, alpha=0.75, height=0.35)
            ax_winloss.barh(['Short'], [short_wins], color=GREEN, alpha=0.75, height=0.35)
            ax_winloss.barh(['Short'], [short_loss], left=[short_wins], color=RED, alpha=0.75, height=0.35)

        ax_winloss.set_xlim(0, total * 1.05)
        ax_winloss.axvline(wins, color=WHITE, linewidth=0.5, alpha=0.4)

        wr = wins / total * 100
        ax_winloss.set_xlabel(f'Win Rate {wr:.1f}%', color=LGRAY, fontsize=8)
        ax_winloss.tick_params(colors=WHITE, labelsize=8)
        ax_winloss.legend(fontsize=7, facecolor=BG, edgecolor=GRAY,
                          labelcolor=WHITE, loc='lower right')
    else:
        ax_winloss.text(0.5, 0.5, 'No trades', ha='center', va='center',
                        color=LGRAY, fontsize=10, transform=ax_winloss.transAxes)

    # ===== 4. 主要指標テーブル =====
    ax_metrics.set_title('Key Metrics', fontsize=10, color=WHITE, pad=4)
    ax_metrics.axis('off')

    metrics = [
        ('Profit Factor', f"{result.get('profit_factor', 0):.3f}",
         result.get('profit_factor', 0) >= 1.5),
        ('Win Rate', f"{result.get('win_rate_pct', 0):.1f}%",
         result.get('win_rate_pct', 0) >= 50),
        ('RR Ratio', f"{result.get('rr_ratio', 0):.2f}",
         result.get('rr_ratio', 0) >= 2.0),
        ('Max Drawdown', f"{result.get('max_drawdown_pct', 0):.1f}%",
         result.get('max_drawdown_pct', 0) <= 10),
        ('Total Return', f"{result.get('total_return_pct', 0):+.2f}%",
         result.get('total_return_pct', 0) > 0),
        ('Trades', f"{result.get('total_trades', 0)}",
         result.get('total_trades', 0) >= 30),
        ('Avg Hold (h)', f"{result.get('avg_duration_hours', 0):.1f}", None),
        ('Timeframe', result.get('timeframe', '-'), None),
    ]

    y_start = 0.92
    row_h = 0.115
    for i, (label, value, ok) in enumerate(metrics):
        y = y_start - i * row_h
        # 背景
        bg_color = BG2 if i % 2 == 0 else '#1e2132'
        rect = FancyBboxPatch((0.0, y - row_h * 0.65), 1.0, row_h * 0.88,
                              boxstyle='round,pad=0.01',
                              facecolor=bg_color, edgecolor='none',
                              transform=ax_metrics.transAxes, clip_on=False)
        ax_metrics.add_patch(rect)

        ax_metrics.text(0.05, y - row_h * 0.15, label, fontsize=9,
                        color=LGRAY, va='center',
                        transform=ax_metrics.transAxes)

        val_color = (GREEN if ok else RED) if ok is not None else WHITE
        ax_metrics.text(0.95, y - row_h * 0.15, value, fontsize=10,
                        color=val_color, va='center', ha='right',
                        fontweight='bold', transform=ax_metrics.transAxes)

    # 合格基準の注釈
    ax_metrics.text(0.5, 0.02, 'Criteria: PF>=1.5 / WR>=50% / DD<=10% / N>=30',
                    fontsize=6.5, color=GRAY, ha='center',
                    transform=ax_metrics.transAxes)

    # ===== 保存 =====
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    return output_path


def plot_monthly_heatmap(trades: list, init_cash: float,
                         output_path: str, title: str = '') -> str:
    """
    年×月のリターンヒートマップ。データ期間が2年以上の場合に有効。

    Args:
        trades: トレードリスト
        init_cash: 初期資金
        output_path: 保存先
        title: タイトル
    """
    if not trades:
        return None

    df = pd.DataFrame(trades)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['year'] = df['exit_time'].dt.year
    df['month'] = df['exit_time'].dt.month

    pivot = df.pivot_table(values='pnl', index='year', columns='month',
                           aggfunc='sum', fill_value=0)

    # 月次リターン%に変換（近似）
    monthly_pct = pivot / init_cash * 100

    if monthly_pct.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, max(3, len(monthly_pct) * 1.2)),
                           facecolor=BG)
    ax.set_facecolor(BG2)

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    # カラーマップ（緑→赤）
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'grrd', [RED, BG2, GREEN])

    max_abs = max(abs(monthly_pct.values.max()), abs(monthly_pct.values.min()), 1)

    im = ax.imshow(monthly_pct.values, cmap=cmap,
                   vmin=-max_abs, vmax=max_abs, aspect='auto')

    # 軸ラベル
    ax.set_xticks(range(len(monthly_pct.columns)))
    ax.set_xticklabels([month_names[m - 1] for m in monthly_pct.columns],
                       color=WHITE, fontsize=9)
    ax.set_yticks(range(len(monthly_pct.index)))
    ax.set_yticklabels(monthly_pct.index, color=WHITE, fontsize=9)

    # セル値
    for y_idx in range(len(monthly_pct.index)):
        for x_idx in range(len(monthly_pct.columns)):
            val = monthly_pct.values[y_idx, x_idx]
            if val != 0:
                ax.text(x_idx, y_idx, f'{val:+.1f}%',
                        ha='center', va='center',
                        color=WHITE, fontsize=8, fontweight='bold')

    plt.colorbar(im, ax=ax, label='リターン%', shrink=0.8)
    ax.set_title(f'{title}  月次リターン ヒートマップ',
                 color=WHITE, fontsize=12, pad=10)
    for spine in ax.spines.values():
        spine.set_color(GRAY)

    fig.savefig(output_path, dpi=110, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    return output_path


def plot_season_analysis(trades: list, init_cash: float,
                         output_path: str) -> str:
    """
    月別勝率・PnL分析。どの月にトレードすべきかを視覚化。
    """
    if not trades:
        return None

    df = pd.DataFrame(trades)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['month'] = df['exit_time'].dt.month

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    stats = []
    for m in range(1, 13):
        sub = df[df['month'] == m]
        if len(sub) == 0:
            stats.append({'month': m, 'trades': 0, 'wr': 0, 'pnl': 0, 'pf': 0})
            continue
        wins = len(sub[sub['pnl'] > 0])
        total_win = sub[sub['pnl'] > 0]['pnl'].sum()
        total_loss = abs(sub[sub['pnl'] <= 0]['pnl'].sum())
        stats.append({
            'month': m,
            'trades': len(sub),
            'wr': wins / len(sub) * 100,
            'pnl': sub['pnl'].sum(),
            'pf': total_win / total_loss if total_loss > 0 else 0,
        })

    sdf = pd.DataFrame(stats)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), facecolor=BG)
    fig.suptitle('Monthly Seasonality Analysis', color=WHITE, fontsize=14, fontweight='bold')

    for ax in axes:
        ax.set_facecolor(BG2)
        for spine in ax.spines.values():
            spine.set_color(GRAY)
        ax.tick_params(colors=WHITE, labelsize=8)

    months = sdf['month'].values

    # 月次PnL
    colors = [GREEN if v >= 0 else RED for v in sdf['pnl'].values]
    axes[0].bar(months, sdf['pnl'].values / 1000, color=colors, alpha=0.85)
    axes[0].axhline(0, color=GRAY, linewidth=0.8)
    axes[0].set_ylabel('PnL (K JPY)', color=WHITE, fontsize=9)
    axes[0].set_title('PnL by Month', color=LGRAY, fontsize=10)
    axes[0].set_xticks(months)
    axes[0].set_xticklabels(month_names, color=WHITE, fontsize=8)

    # 月別勝率
    wr_colors = [GREEN if v >= 50 else RED for v in sdf['wr'].values]
    axes[1].bar(months, sdf['wr'].values, color=wr_colors, alpha=0.85)
    axes[1].axhline(50, color=YELLOW, linewidth=1.0, linestyle='--', alpha=0.7, label='50%')
    axes[1].set_ylabel('Win Rate %', color=WHITE, fontsize=9)
    axes[1].set_title('Win Rate by Month', color=LGRAY, fontsize=10)
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(months)
    axes[1].set_xticklabels(month_names, color=WHITE, fontsize=8)
    axes[1].legend(fontsize=8, facecolor=BG, edgecolor=GRAY, labelcolor=WHITE)

    # 月別PF
    pf_colors = [GREEN if v >= 1.5 else (YELLOW if v >= 1.0 else RED)
                 for v in sdf['pf'].values]
    axes[2].bar(months, sdf['pf'].values, color=pf_colors, alpha=0.85)
    axes[2].axhline(1.0, color=GRAY, linewidth=0.8, linestyle='--', alpha=0.6)
    axes[2].axhline(1.5, color=GREEN, linewidth=1.0, linestyle='--', alpha=0.7, label='PF 1.5')
    axes[2].set_ylabel('Profit Factor', color=WHITE, fontsize=9)
    axes[2].set_title('Profit Factor by Month  (green>=1.5, yellow>=1.0)', color=LGRAY, fontsize=10)
    axes[2].set_xticks(months)
    axes[2].set_xticklabels(month_names, color=WHITE, fontsize=8)
    axes[2].legend(fontsize=8, facecolor=BG, edgecolor=GRAY, labelcolor=WHITE)

    # おすすめトレード月
    recommended = sdf[(sdf['pf'] >= 1.5) & (sdf['wr'] >= 50) & (sdf['trades'] >= 2)]['month'].tolist()
    avoid = sdf[(sdf['pf'] < 1.0) & (sdf['trades'] >= 2)]['month'].tolist()
    rec_names = [month_names[m - 1] for m in recommended]
    avoid_names = [month_names[m - 1] for m in avoid]
    note = f"Recommended: {', '.join(rec_names) or 'None'}    Avoid: {', '.join(avoid_names) or 'None'}"
    fig.text(0.5, 0.01, note, ha='center', color=LGRAY, fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, dpi=110, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    return output_path
