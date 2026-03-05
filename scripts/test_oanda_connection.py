"""
OANDA API 接続・発注テストスクリプト
USD_JPY を100通貨で成行買い → 5秒待機 → 決済 という一連の動作を検証する。
"""

import sys
import time
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.oanda_client import OandaClient


def main():
    print("=" * 50)
    print("OANDA API 接続・発注テスト開始")
    print("=" * 50)

    # クライアント初期化
    client = OandaClient()
    print(f"[OK] OandaClient 初期化完了")
    print(f"     環境: {client.environment}")
    print(f"     口座ID: {client.account_id}")

    # Step 1: USD_JPY 100通貨 成行買い注文
    print("\n[Step 1] USD_JPY 100通貨 成行買い注文を発注...")
    response = client.create_order("USD_JPY", 100)

    # トレードIDを取得
    trade_id = None
    if "orderFillTransaction" in response:
        trade_id = response["orderFillTransaction"].get("tradeOpened", {}).get("tradeID")
    if not trade_id and "relatedTransactionIDs" in response:
        trade_id = response["relatedTransactionIDs"][-1] if response["relatedTransactionIDs"] else None

    # フォールバック: オープントレードから取得
    if not trade_id:
        open_trades = client.get_open_trades("USD_JPY")
        if open_trades:
            trade_id = open_trades[0]["id"]

    if not trade_id:
        print("[ERROR] トレードIDを取得できませんでした。")
        print(f"レスポンス: {response}")
        sys.exit(1)

    print(f"[OK] 発注成功！ トレードID: {trade_id}")

    # Step 2: 5秒待機
    print("\n[Step 2] 5秒待機中...")
    for i in range(5, 0, -1):
        print(f"  {i}秒後に決済...", end="\r")
        time.sleep(1)
    print()

    # Step 3: ポジション決済
    print("[Step 3] ポジションを決済...")
    close_response = client.close_trade(trade_id)

    if "orderFillTransaction" in close_response:
        pl = close_response["orderFillTransaction"].get("pl", "N/A")
        print(f"[OK] 決済成功！ 損益: {pl} JPY")
    else:
        print(f"[OK] 決済レスポンス受信: {close_response}")

    print("\n" + "=" * 50)
    print("OANDA API connection and order test successful.")
    print("=" * 50)


if __name__ == "__main__":
    main()
