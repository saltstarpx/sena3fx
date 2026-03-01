"""
承認済み商品リスト生成 (Approved Universe)
==========================================
results/universe_performance.csv を読み込み、
以下の条件を満たす商品名のリストを生成する。

  承認条件: Sharpe > 1.0 AND Trades > 20

実行方法:
  python lib/approved_list.py                    # 標準出力のみ
  python lib/approved_list.py --json             # JSON形式で出力
  python lib/approved_list.py --save             # results/approved_universe.json に保存

使用方法 (モジュールとして):
  from lib.approved_list import load_approved
  symbols = load_approved()   # ['XAUUSD', 'XAGUSD', ...]
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 承認条件
DEFAULT_MIN_SHARPE = 1.0
DEFAULT_MIN_TRADES = 20

UNIVERSE_CSV   = os.path.join(ROOT, 'results', 'universe_performance.csv')
APPROVED_JSON  = os.path.join(ROOT, 'results', 'approved_universe.json')


def load_approved(
    csv_path: str = UNIVERSE_CSV,
    min_sharpe: float = DEFAULT_MIN_SHARPE,
    min_trades: int   = DEFAULT_MIN_TRADES,
) -> list[str]:
    """
    バックテスト結果CSVから承認済み商品リストを返す。

    Args:
        csv_path  : universe_performance.csv のパス
        min_sharpe: Sharpe比の最低閾値 (デフォルト 1.0)
        min_trades: 最低トレード数 (デフォルト 20)

    Returns:
        list[str]: 承認済み商品シンボルのリスト（Sharpe降順）

    Raises:
        FileNotFoundError: CSVファイルが存在しない場合
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'universe_performance.csv が見つかりません: {csv_path}\n'
            f'先に scripts/backtest_universe.py を実行してください。'
        )

    import pandas as pd
    df = pd.read_csv(csv_path)

    # 数値型変換（CSVに空文字列がある場合に対応）
    df['sharpe_ratio'] = pd.to_numeric(df['sharpe_ratio'], errors='coerce')
    df['total_trades'] = pd.to_numeric(df['total_trades'], errors='coerce').fillna(0).astype(int)

    approved = df[
        (df['sharpe_ratio'] > min_sharpe) &
        (df['total_trades'] > min_trades)
    ].sort_values('sharpe_ratio', ascending=False)

    return approved['symbol'].tolist()


def load_approved_detail(
    csv_path: str = UNIVERSE_CSV,
    min_sharpe: float = DEFAULT_MIN_SHARPE,
    min_trades: int   = DEFAULT_MIN_TRADES,
) -> list[dict]:
    """
    承認済み商品の詳細情報（辞書リスト）を返す。
    モニターや実運用botが取引パラメータを参照するために使用。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'{csv_path} が存在しません。')

    import pandas as pd
    df = pd.read_csv(csv_path)
    df['sharpe_ratio'] = pd.to_numeric(df['sharpe_ratio'], errors='coerce')
    df['calmar_ratio'] = pd.to_numeric(df['calmar_ratio'], errors='coerce')
    df['max_drawdown_pct'] = pd.to_numeric(df['max_drawdown_pct'], errors='coerce')
    df['profit_factor']    = pd.to_numeric(df['profit_factor'],    errors='coerce')
    df['win_rate_pct']     = pd.to_numeric(df['win_rate_pct'],     errors='coerce')
    df['total_trades']     = pd.to_numeric(df['total_trades'], errors='coerce').fillna(0).astype(int)

    approved = df[
        (df['sharpe_ratio'] > min_sharpe) &
        (df['total_trades'] > min_trades)
    ].sort_values('sharpe_ratio', ascending=False)

    return approved.to_dict(orient='records')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='承認済み商品リスト生成')
    parser.add_argument('--min-sharpe', type=float, default=DEFAULT_MIN_SHARPE,
                        help=f'Sharpe比最低値 (デフォルト: {DEFAULT_MIN_SHARPE})')
    parser.add_argument('--min-trades', type=int, default=DEFAULT_MIN_TRADES,
                        help=f'最低トレード数 (デフォルト: {DEFAULT_MIN_TRADES})')
    parser.add_argument('--json', action='store_true', help='JSON形式で出力')
    parser.add_argument('--save', action='store_true',
                        help=f'results/approved_universe.json に保存')
    args = parser.parse_args()

    try:
        details = load_approved_detail(
            min_sharpe=args.min_sharpe,
            min_trades=args.min_trades,
        )
        symbols = [d['symbol'] for d in details]
    except FileNotFoundError as e:
        print(f'エラー: {e}', file=sys.stderr)
        sys.exit(1)

    if not symbols:
        print('承認条件を満たす商品がありません。')
        print(f'条件: Sharpe > {args.min_sharpe}, Trades > {args.min_trades}')
        sys.exit(0)

    print('=' * 60)
    print(f'承認済み商品リスト (Sharpe>{args.min_sharpe}, Trades>{args.min_trades})')
    print('=' * 60)

    if args.json:
        print(json.dumps(symbols, ensure_ascii=False, indent=2))
    else:
        print(f'{"商品":<12} {"Sharpe":>8} {"Calmar":>8} {"MDD%":>7} {"Trades":>7}')
        print('-' * 50)
        for d in details:
            sh  = d.get('sharpe_ratio', '')
            ca  = d.get('calmar_ratio', '')
            mdd = d.get('max_drawdown_pct', '')
            tr  = d.get('total_trades', 0)
            sh_s  = f'{sh:.3f}'  if isinstance(sh, float)  else str(sh)
            ca_s  = f'{ca:.3f}'  if isinstance(ca, float)  else str(ca)
            mdd_s = f'{mdd:.1f}' if isinstance(mdd, float) else str(mdd)
            print(f'{d["symbol"]:<12} {sh_s:>8} {ca_s:>8} {mdd_s:>7} {tr:>7}')
        print('-' * 50)
        print(f'合計: {len(symbols)} 商品承認')

    if args.save:
        payload = {
            'generated_at': str(__import__('datetime').datetime.now()),
            'criteria': {'min_sharpe': args.min_sharpe, 'min_trades': args.min_trades},
            'approved_symbols': symbols,
            'details': details,
        }
        with open(APPROVED_JSON, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f'\n保存: {APPROVED_JSON}')

    return symbols


if __name__ == '__main__':
    main()
