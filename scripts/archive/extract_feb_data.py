import pandas as pd

def extract_feb_data():
    input_path = "/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Q1.csv"
    output_path = "/home/ubuntu/sena3fx/data/ohlc/USDJPY_1m_2026_Feb.csv"
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2月のデータをフィルタリング
    feb_df = df[(df['timestamp'] >= '2026-02-01') & (df['timestamp'] < '2026-03-01')]
    
    if feb_df.empty:
        print("No February data found.")
        return
    
    print(f"Extracted {len(feb_df)} rows for February.")
    feb_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    extract_feb_data()
