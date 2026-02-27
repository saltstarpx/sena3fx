import yfinance as yf
import pandas as pd
import os

outdir = '/home/ubuntu/sena3fx/data/ohlc'
os.makedirs(outdir, exist_ok=True)

def fetch_and_save(ticker_symbol, label, start, end):
    """指定ティッカーの全時間足を取得・保存"""
    t = yf.Ticker(ticker_symbol)
    results = {}
    
    # 日足
    print(f"  日足取得中...")
    df = t.history(start=start, end=end, interval="1d")
    df = df[['Open','High','Low','Close','Volume']].copy()
    df.index.name = 'datetime'
    df.columns = ['open','high','low','close','volume']
    path = f'{outdir}/{label}_2025_1d.csv'
    df.to_csv(path)
    print(f"  日足: {len(df)}行 → {path}")
    results['1d'] = df
    
    # 1時間足
    print(f"  1時間足取得中...")
    df_1h = t.history(start=start, end=end, interval="1h")
    df_1h = df_1h[['Open','High','Low','Close','Volume']].copy()
    df_1h.index.name = 'datetime'
    df_1h.columns = ['open','high','low','close','volume']
    path = f'{outdir}/{label}_2025_1h.csv'
    df_1h.to_csv(path)
    print(f"  1時間足: {len(df_1h)}行 → {path}")
    results['1h'] = df_1h
    
    # 15分足
    print(f"  15分足取得中...")
    df_15m = t.history(start=start, end=end, interval="15m")
    df_15m = df_15m[['Open','High','Low','Close','Volume']].copy()
    df_15m.index.name = 'datetime'
    df_15m.columns = ['open','high','low','close','volume']
    path = f'{outdir}/{label}_2025_15m.csv'
    df_15m.to_csv(path)
    print(f"  15分足: {len(df_15m)}行 → {path}")
    results['15m'] = df_15m
    
    # 4時間足（1hから生成）
    print(f"  4時間足生成中...")
    df_4h = df_1h.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    path = f'{outdir}/{label}_2025_4h.csv'
    df_4h.to_csv(path)
    print(f"  4時間足: {len(df_4h)}行 → {path}")
    
    # 8時間足（1hから生成）
    print(f"  8時間足生成中...")
    df_8h = df_1h.resample('8h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    path = f'{outdir}/{label}_2025_8h.csv'
    df_8h.to_csv(path)
    print(f"  8時間足: {len(df_8h)}行 → {path}")
    
    return results

start = "2025-01-01"
end = "2026-02-27"

# XAUUSD (Gold Futures)
print("=" * 50)
print("XAUUSD (GC=F) 取得開始")
print("=" * 50)
fetch_and_save("GC=F", "XAUUSD", start, end)

# XAGUSD (Silver Futures)
print("\n" + "=" * 50)
print("XAGUSD (SI=F) 取得開始")
print("=" * 50)
fetch_and_save("SI=F", "XAGUSD", start, end)

print("\n全データ取得完了!")
print("\n生成ファイル一覧:")
for f in sorted(os.listdir(outdir)):
    if f.endswith('.csv'):
        size = os.path.getsize(f'{outdir}/{f}')
        print(f"  {f} ({size:,} bytes)")
