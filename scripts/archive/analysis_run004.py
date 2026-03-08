"""
RUN-20260305-004 定量分析・可視化スクリプト
P0修正効果（C1フィルター切替）の可視化
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.family'] = 'Noto Sans CJK JP'
rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIGS, exist_ok=True)

# ──────────────────────────────────────────────────────
# データ
# ──────────────────────────────────────────────────────
c1_data = {
    'Before\n(extract_levels)': 88.5,
    'After\n(extract_levels_binned)': 25.6,
    'ハンドオフ文書\n予測値': 30.0,
}

bt_data = {
    'Yagami_A_Before':    {'pf': 1.501, 'wr': 44.4, 'trades': 18,  'mdd': 5.9},
    'Yagami_B_Before':    {'pf': 1.122, 'wr': 39.1, 'trades': 192, 'mdd': 18.7},
    'Yagami_LonNY_Before':{'pf': 1.067, 'wr': 41.3, 'trades': 109, 'mdd': 20.4},
    'Yagami_A_After':     {'pf': 2.012, 'wr': 42.9, 'trades': 7,   'mdd': 2.5},
    'Yagami_B_After':     {'pf': 0.818, 'wr': 30.0, 'trades': 80,  'mdd': 16.7},
    'Yagami_LonNY_After': {'pf': 1.056, 'wr': 38.0, 'trades': 50,  'mdd': 14.9},
}

p0_fixes = [
    ('P0-1', 'C1フィルター切替\n(extract_levels_binned)', '完了', 'C1充足率 88.5%→25.6%\nYagami_A PF: 1.501→2.012'),
    ('P0-2', 'sig_maedai_yagami_union\n未定義疑惑', '確認済み', 'lib/yagami.py:1727に定義済み\n問題なし'),
    ('P0-3', 'ファイル名年度\nハードコード', '完了', 'glob()による動的検出に変更\n2026年以降も自動対応'),
    ('P1-4', 'USDフィルター非対称\n(long側のみ)', '完了', 'short側フィルターも追加\n対称設計に修正'),
    ('P2-10', 'PDCAインサイト重複\n登録バグ', '完了', 'set()による重複排除を追加\nknowledge.json汚染防止'),
    ('P1-1', 'Sharpe=2.8統計的\n脆弱性(N=21)', '未着手', 'Walk-Forward検証が必要\n次サイクルで実施予定'),
]

# ──────────────────────────────────────────────────────
# 図1: C1充足率の変化
# ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('RUN-20260305-004: P0修正効果定量分析\nClaude Codeハンドオフ文書 P0→P1→P2 実装結果',
             fontsize=14, fontweight='bold', y=1.02)

# 左: C1充足率比較
ax1 = axes[0]
labels = list(c1_data.keys())
values = list(c1_data.values())
colors = ['#e74c3c', '#27ae60', '#3498db']
bars = ax1.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
ax1.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='50%ライン')
ax1.axhline(y=30, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='目標値(30%)')
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax1.set_title('C1充足率の変化\n(P0-1修正効果)', fontsize=12, fontweight='bold')
ax1.set_ylabel('C1充足率 (%)', fontsize=10)
ax1.set_ylim(0, 105)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.annotate('フィルター機能\n回復', xy=(1, 25.6), xytext=(1.5, 55),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=10, color='green', fontweight='bold')

# 中: PF比較（Before vs After）
ax2 = axes[1]
strategies = ['Yagami_A', 'Yagami_B', 'Yagami_LonNY']
pf_before = [bt_data[f'{s}_Before']['pf'] for s in strategies]
pf_after = [bt_data[f'{s}_After']['pf'] for s in strategies]
x = np.arange(len(strategies))
width = 0.35
b1 = ax2.bar(x - width/2, pf_before, width, label='Before (旧C1)', color='#e74c3c', alpha=0.8)
b2 = ax2.bar(x + width/2, pf_after, width, label='After (新C1)', color='#27ae60', alpha=0.8)
ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='損益分岐点(PF=1.0)')
ax2.axhline(y=1.5, color='blue', linestyle='--', linewidth=1.5, alpha=0.5, label='目標(PF=1.5)')
for bar in b1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in b2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
ax2.set_title('プロフィットファクター比較\n(Before vs After)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Profit Factor', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(strategies, fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 2.5)

# 右: 修正項目サマリー
ax3 = axes[2]
ax3.axis('off')
table_data = []
for item in p0_fixes:
    table_data.append([item[0], item[1], item[2]])
table = ax3.table(
    cellText=table_data,
    colLabels=['ID', '修正内容', '状態'],
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)
# 色付け
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif table_data[row-1][2] == '完了':
        cell.set_facecolor('#d5f5e3')
    elif table_data[row-1][2] == '確認済み':
        cell.set_facecolor('#d6eaf8')
    elif table_data[row-1][2] == '未着手':
        cell.set_facecolor('#fdebd0')
ax3.set_title('修正項目一覧\n(P0→P1→P2)', fontsize=12, fontweight='bold')

plt.tight_layout()
out = os.path.join(FIGS, 'fig11_run004_p0_fix.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"図11保存: {out}")

# ──────────────────────────────────────────────────────
# 図2: トレード数とMDD比較
# ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# トレード数
ax1 = axes[0]
strategies = ['Yagami_A', 'Yagami_B', 'Yagami_LonNY']
n_before = [bt_data[f'{s}_Before']['trades'] for s in strategies]
n_after = [bt_data[f'{s}_After']['trades'] for s in strategies]
x = np.arange(len(strategies))
width = 0.35
b1 = ax1.bar(x - width/2, n_before, width, label='Before', color='#e74c3c', alpha=0.8)
b2 = ax1.bar(x + width/2, n_after, width, label='After', color='#27ae60', alpha=0.8)
ax1.axhline(y=30, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='統計的有意水準(N=30)')
for bar in b1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_title('トレード数の変化\n(C1フィルター強化による減少)', fontsize=12, fontweight='bold')
ax1.set_ylabel('トレード数', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(strategies, fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# MDD比較
ax2 = axes[1]
mdd_before = [bt_data[f'{s}_Before']['mdd'] for s in strategies]
mdd_after = [bt_data[f'{s}_After']['mdd'] for s in strategies]
b1 = ax2.bar(x - width/2, mdd_before, width, label='Before', color='#e74c3c', alpha=0.8)
b2 = ax2.bar(x + width/2, mdd_after, width, label='After', color='#27ae60', alpha=0.8)
ax2.axhline(y=15, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='目標上限(15%)')
for bar in b1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
for bar in b2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
ax2.set_title('最大ドローダウンの変化\n(C1フィルター強化による改善)', fontsize=12, fontweight='bold')
ax2.set_ylabel('最大ドローダウン (%)', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(strategies, fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('RUN-004: C1フィルター切替によるリスク特性の変化', fontsize=13, fontweight='bold')
plt.tight_layout()
out2 = os.path.join(FIGS, 'fig12_run004_risk_profile.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"図12保存: {out2}")

print("\n可視化完了")
