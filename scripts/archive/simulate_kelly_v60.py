import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_kelly():
    # トレード履歴の読み込み
    trades_path = "/home/ubuntu/sena3fx/results/trades_v60_kelly.csv"
    if not os.path.exists(trades_path):
        print(f"Error: {trades_path} not found.")
        return
    
    trades_df = pd.read_csv(trades_path)
    pnl_pips = trades_df['pnl_pips'].values
    
    # 統計量
    win_rate = np.sum(pnl_pips > 0) / len(pnl_pips)
    avg_win = np.mean(pnl_pips[pnl_pips > 0])
    avg_loss = abs(np.mean(pnl_pips[pnl_pips <= 0]))
    odds = avg_win / avg_loss
    
    # ケリー係数
    kelly_f = (odds * win_rate - (1 - win_rate)) / odds if odds > 0 else 0
    
    # シミュレーション設定
    initial_balance = 1000000 # 100万円
    n_simulations = 100
    n_trades = len(pnl_pips)
    
    # 1. ケリー係数（フルケリー）を使用した場合（負数の場合は0とする）
    f_full = max(0, kelly_f)
    # 2. ハーフケリー（保守的）
    f_half = f_full * 0.5
    # 3. 固定比率（例: 2%リスク）
    f_fixed = 0.02
    
    def run_sim(f):
        results = []
        for _ in range(n_simulations):
            balance = initial_balance
            history = [balance]
            # 実際のトレード順序をシャッフルしてシミュレーション
            shuffled_pnl = np.random.permutation(pnl_pips)
            for pnl in shuffled_pnl:
                # リスク額 = 残高 * f
                # 1pipsの価値を100円（1ロット=10万通貨想定）とする
                # 実際には損切り幅(ATR)に合わせてロットを調整するが、ここでは簡易化
                risk_amount = balance * f
                # 簡易的な損益計算: (pnl / avg_loss) * risk_amount
                # avg_lossで割ることで、1単位のリスクに対する倍率とする
                trade_result = (pnl / avg_loss) * risk_amount
                balance += trade_result
                if balance <= 0:
                    balance = 0
                history.append(balance)
            results.append(history)
        return np.array(results)

    sim_full = run_sim(f_full)
    sim_half = run_sim(f_half)
    sim_fixed = run_sim(f_fixed)
    
    # グラフ作成
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(sim_fixed.T, color='gray', alpha=0.1)
    plt.plot(np.median(sim_fixed, axis=0), color='blue', label='Fixed 2% Risk (Median)')
    plt.title(f'Equity Simulation: Fixed 2% Risk (n={n_simulations})')
    plt.ylabel('Balance (JPY)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(sim_half.T, color='gray', alpha=0.1)
    plt.plot(np.median(sim_half, axis=0), color='green', label='Half Kelly (Median)')
    plt.title(f'Equity Simulation: Half Kelly (f={f_half:.4f})')
    plt.xlabel('Trade Number')
    plt.ylabel('Balance (JPY)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("/home/ubuntu/sena3fx/results/v60_kelly_simulation.png")
    
    print("=== Kelly Simulation Results ===")
    print(f"Full Kelly Fraction: {kelly_f:.4f}")
    print(f"Half Kelly Fraction: {f_half:.4f}")
    print(f"Fixed 2% Median Final Balance: {np.median(sim_fixed, axis=0)[-1]:.0f}")
    print(f"Half Kelly Median Final Balance: {np.median(sim_half, axis=0)[-1]:.0f}")
    print(f"Simulation graph saved to: /home/ubuntu/sena3fx/results/v60_kelly_simulation.png")

if __name__ == "__main__":
    simulate_kelly()
