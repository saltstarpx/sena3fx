"""
承認済み商品リスト生成 (Approved Universe) v2.0
=================================================
results/universe_performance.csv を読み込み、
以下の条件を満たす商品名のリストを生成する。

  承認条件 v2.0 (v16結果に基づく厳格化):
    Sharpe > 1.5 AND Calmar > 5.0

  v1.0 旧条件 (参考): Sharpe > 1.0 AND Trades > 20

  根拠: XAUUSD+Kelly(f=0.25) が Sharpe=1.717, Calmar=6.574 という
  卓越したパフォーマンスを記録。このレベルを基準として
  「真に取引すべき商品」のみを自動承認する。

実行方法:
  python lib/approved_list.py                    # 標準出力のみ
  python lib/approved_list.py --json             # JSON形式で出力
  python lib/approved_list.py --save             # results/approved_universe.json に保存
  python lib/approved_list.py --v1               # v1.0旧条件で実行 (比較用)

使用方法 (モジュールとして):
  from lib.approved_list import load_approved
  symbols = load_approved()   # ['XAUUSD', ...]  (Sharpe>1.5 AND Calmar>5.0)
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 承認条件 v2.0 (厳格化) ──────────────────────────
DEFAULT_MIN_SHARPE = 1.5    # v1.0: 1.0 → v2.0: 1.5
DEFAULT_MIN_CALMAR = 5.0    # v2.0新設: Calmar > 5.0

# v1.0 後方互換 (--v1 オプション用)
_V1_MIN_SHARPE = 1.0
_V1_MIN_TRADES = 20

UNIVERSE_CSV   = os.path.join(ROOT, 'results', 'universe_performance.csv')
APPROVED_JSON  = os.path.join(ROOT, 'results', 'approved_universe.json')


def load_approved(
    csv_path: str = UNIVERSE_CSV,
    min_sharpe: float = DEFAULT_MIN_SHARPE,
    min_calmar: float = DEFAULT_MIN_CALMAR,
) -> list[str]:
    """
    バックテスト結果CSVから承認済み商品リストを返す (v2.0 厳格基準)。

    Args:
        csv_path  : universe_performance.csv のパス
        min_sharpe: Sharpe比の最低閾値 (デフォルト 1.5)
        min_calmar: Calmar比の最低閾値 (デフォルト 5.0)

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

    df['sharpe_ratio'] = pd.to_numeric(df['sharpe_ratio'], errors='coerce')
    df['calmar_ratio'] = pd.to_numeric(df['calmar_ratio'], errors='coerce')

    approved = df[
        (df['sharpe_ratio'] > min_sharpe) &
        (df['calmar_ratio'] > min_calmar)
    ].sort_values('sharpe_ratio', ascending=False)

    return approved['symbol'].tolist()


def load_approved_detail(
    csv_path: str = UNIVERSE_CSV,
    min_sharpe: float = DEFAULT_MIN_SHARPE,
    min_calmar: float = DEFAULT_MIN_CALMAR,
) -> list[dict]:
    """
    承認済み商品の詳細情報（辞書リスト）を返す。
    モニターや実運用botが取引パラメータを参照するために使用。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'{csv_path} が存在しません。')

    import pandas as pd
    df = pd.read_csv(csv_path)
    for col in ['sharpe_ratio', 'calmar_ratio', 'max_drawdown_pct',
                'profit_factor', 'win_rate_pct']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['total_trades'] = pd.to_numeric(df['total_trades'], errors='coerce').fillna(0).astype(int)

    approved = df[
        (df['sharpe_ratio'] > min_sharpe) &
        (df['calmar_ratio'] > min_calmar)
    ].sort_values('sharpe_ratio', ascending=False)

    return approved.to_dict(orient='records')


def _load_approved_v1(csv_path: str = UNIVERSE_CSV) -> list[dict]:
    """v1.0旧条件 (Sharpe>1.0 AND Trades>20) で返す (後方互換・比較用)。"""
    if not os.path.exists(csv_path):
        return []
    import pandas as pd
    df = pd.read_csv(csv_path)
    df['sharpe_ratio'] = pd.to_numeric(df['sharpe_ratio'], errors='coerce')
    df['calmar_ratio'] = pd.to_numeric(df['calmar_ratio'], errors='coerce')
    df['total_trades'] = pd.to_numeric(df['total_trades'], errors='coerce').fillna(0).astype(int)
    approved = df[
        (df['sharpe_ratio'] > _V1_MIN_SHARPE) &
        (df['total_trades'] > _V1_MIN_TRADES)
    ].sort_values('sharpe_ratio', ascending=False)
    return approved.to_dict(orient='records')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='承認済み商品リスト生成 v2.0')
    parser.add_argument('--min-sharpe', type=float, default=DEFAULT_MIN_SHARPE,
                        help=f'Sharpe比最低値 (デフォルト: {DEFAULT_MIN_SHARPE})')
    parser.add_argument('--min-calmar', type=float, default=DEFAULT_MIN_CALMAR,
                        help=f'Calmar比最低値 (デフォルト: {DEFAULT_MIN_CALMAR})')
    parser.add_argument('--json', action='store_true', help='JSON形式で出力')
    parser.add_argument('--save', action='store_true',
                        help='results/approved_universe.json に保存')
    parser.add_argument('--v1', action='store_true',
                        help='v1.0旧条件で実行 (Sharpe>1.0 AND Trades>20, 比較用)')
    args = parser.parse_args()

    if args.v1:
        details = _load_approved_v1()
        criteria_str = f'v1.0: Sharpe>{_V1_MIN_SHARPE}, Trades>{_V1_MIN_TRADES}'
    else:
        try:
            details = load_approved_detail(
                min_sharpe=args.min_sharpe,
                min_calmar=args.min_calmar,
            )
        except FileNotFoundError as e:
            print(f'エラー: {e}', file=sys.stderr)
            sys.exit(1)
        criteria_str = f'v2.0: Sharpe>{args.min_sharpe}, Calmar>{args.min_calmar}'

    symbols = [d['symbol'] for d in details]

    if not symbols:
        print('承認条件を満たす商品がありません。')
        print(f'条件: {criteria_str}')
        sys.exit(0)

    print('=' * 62)
    print(f'承認済み商品リスト ({criteria_str})')
    print('=' * 62)

    if args.json:
        print(json.dumps(symbols, ensure_ascii=False, indent=2))
    else:
        print(f'{"商品":<12} {"Sharpe":>8} {"Calmar":>8} {"MDD%":>7} {"Trades":>7}')
        print('-' * 52)
        for d in details:
            sh  = d.get('sharpe_ratio', '')
            ca  = d.get('calmar_ratio', '')
            mdd = d.get('max_drawdown_pct', '')
            tr  = d.get('total_trades', 0)
            sh_s  = f'{sh:.3f}'  if isinstance(sh, float) else str(sh)
            ca_s  = f'{ca:.3f}'  if isinstance(ca, float) else str(ca)
            mdd_s = f'{mdd:.1f}' if isinstance(mdd, float) else str(mdd)
            print(f'{d["symbol"]:<12} {sh_s:>8} {ca_s:>8} {mdd_s:>7} {tr:>7}')
        print('-' * 52)
        print(f'合計: {len(symbols)} 商品承認')

    if args.save:
        payload = {
            'generated_at': str(__import__('datetime').datetime.now()),
            'criteria_version': 'v2.0',
            'criteria': {
                'min_sharpe': args.min_sharpe,
                'min_calmar': args.min_calmar,
                'note': 'v1.0では Sharpe>1.0 AND Trades>20 だったが、XAUUSD+Kelly実績に基づき厳格化',
            },
            'approved_symbols': symbols,
            'details': details,
        }
        with open(APPROVED_JSON, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f'\n保存: {APPROVED_JSON}')

    return symbols


if __name__ == '__main__':
    main()
