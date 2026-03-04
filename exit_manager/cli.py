"""
Exit Manager — コマンドラインインターフェース
=============================================
水原がOANDAでトレードを建てた後、このCLIで Exit Manager に登録する。
botはその後の出口（損切・利確・ポジション管理）のみを強制する。

コマンド:
  entry   - 新規トレード登録（SL確認プロンプト付き）
  status  - 現在のポジション状況表示
  override - 緊急手動介入（パスワード保護）
  report  - 集計レポート生成

Usage:
  python -m exit_manager.cli entry --trade-id 12345 \\
      --reason "H4二番底+包み陽線+レジサポ反転"

  python -m exit_manager.cli status

  python -m exit_manager.cli override \\
      --trade-id 12345 --action close_all

  python -m exit_manager.cli report --days 14
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from exit_manager.lot_calculator import calculate_position_size
from exit_manager.logger import ExitManagerLogger
from exit_manager.notifier import DiscordNotifier
from exit_manager.oanda_client import ExitManagerClient
from exit_manager.position_manager import TradePhase, TradeRegistry, TradeState
from exit_manager.exit_rules import check_initial_sl


def load_config() -> dict:
    cfg_path = Path(__file__).parent / 'config.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  entry コマンド                                                     #
# ------------------------------------------------------------------ #

def cmd_entry(args, config, client, registry, logger):
    """
    新規トレードを Exit Manager に登録する。

    フロー:
      1. OANDAからトレード詳細を取得（SL確認）
      2. ロット計算 & 検証
      3. Kill Switchチェック（エントリー可否）
      4. SL整合性確認プロンプト
      5. TradeStateを作成しレジストリに登録
      6. JSONL ログ出力
    """
    trade_id = args.trade_id
    reason = getattr(args, 'reason', '')

    print(f'\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print(f'  Exit Manager — トレード登録')
    print(f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')

    # 1. OANDAからトレード詳細を取得
    try:
        trade_details = client.get_trade_details(trade_id)
    except Exception as e:
        print(f'\n❌ OANDAからトレード取得失敗: {e}')
        sys.exit(1)

    if not trade_details:
        print(f'\n❌ トレード {trade_id} が見つかりません（OANDAでオープン確認してください）')
        sys.exit(1)

    # OANDAレスポンスからフィールドを抽出
    instrument = trade_details.get('instrument', '')
    raw_units = int(trade_details.get('currentUnits', 0))
    side = 'long' if raw_units > 0 else 'short'
    entry_price = float(trade_details.get('price', 0))
    current_units = abs(raw_units)

    # SLをOANDAから取得
    sl_order = trade_details.get('stopLossOrder', {})
    sl_price = float(sl_order.get('price', 0)) if sl_order else None

    if not sl_price:
        print('\n❌ OANDAにSL（ストップロス）が設定されていません。')
        print('   先にOANDA上でSLを設定してから登録してください。')
        sys.exit(1)

    sl_distance = abs(entry_price - sl_price)

    # 2. ロット計算
    lot_result = calculate_position_size(
        symbol=instrument,
        entry_price=entry_price,
        invalidation_price=sl_price,
        config=config,
    )

    if instrument not in config.get('instruments', {}):
        print(f'\n⚠️ 銘柄 {instrument} はconfigに設定されていません。')

    # 3. 情報表示
    print(f'\n  銘柄:         {instrument} {side.upper()}')
    print(f'  エントリー:   ${entry_price:,.3f}')
    print(f'  現在SL:       ${sl_price:,.3f}  (幅: ${sl_distance:.3f})')
    print(f'  ユニット数:   {current_units}')
    print(f'  1R (基準):    ¥{config["account"]["max_loss_jpy"]:,}')
    if not lot_result.get('rejected'):
        print(f'  計算ロット:   {lot_result["units"]}u (リスク ¥{lot_result["risk_jpy"]:,})')
    else:
        print(f'  ⚠️ ロット計算警告: {lot_result.get("reason", "")}')

    # 4. SL整合性確認
    print(f'\n  ━━ 重要確認 ━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print(f'  ⚠️ SL ${sl_price:,.3f} は「最後の押し安値/戻り高値」に置いていますか？')
    print(f'  ⚠️ 「中途半端な位置に置くならエントリーしなくて良い」（ポジり方の本）')
    if reason:
        print(f'\n  エントリー理由: {reason}')

    confirm = input('\n  確認して Enter を押してください（SLが適切: y, キャンセル: n）: ')
    if confirm.strip().lower() != 'y':
        print('\n登録をキャンセルしました。')
        sys.exit(0)

    # 5. TradeState を作成
    lockout_until = datetime.utcnow() + timedelta(minutes=60)
    trade = TradeState(
        trade_id=trade_id,
        instrument=instrument,
        side=side,
        entry_price=entry_price,
        entry_time=datetime.utcnow(),
        sl_price=sl_price,
        original_units=current_units,
        current_units=current_units,
        sl_distance_usd=sl_distance,
        one_r_jpy=float(config['account']['max_loss_jpy']),
        phase=TradePhase.OPEN,
        notes=reason,
    )

    # SL検証
    valid, warning = check_initial_sl(trade, config)
    if not valid:
        print(f'\n❌ SL検証失敗: {warning}')
        sys.exit(1)
    if warning:
        print(f'\n⚠️ SL警告: {warning}')

    registry.register(trade)

    # 6. JSONL ログ出力
    logger.log_trade_registered(
        trade_id=trade_id,
        symbol=instrument,
        direction=side.upper(),
        entry_price=entry_price,
        initial_sl=sl_price,
        sl_distance_usd=sl_distance,
        units=current_units,
        risk_jpy=lot_result.get('risk_jpy', 0) if not lot_result.get('rejected') else 0,
        tp1_price=entry_price + sl_distance if side == 'long' else entry_price - sl_distance,
        lockout_until=lockout_until,
        reason=reason,
    )

    print(f'\n✅ トレード {trade_id} を登録しました')
    print(f'   フェーズ: OPEN')
    print(f'   ロックアウト: 60分（〜{lockout_until.strftime("%H:%M")} UTC）')
    print(f'   main.py が出口管理を開始します。\n')


# ------------------------------------------------------------------ #
#  status コマンド                                                    #
# ------------------------------------------------------------------ #

def cmd_status(args, config, client, registry, logger):
    """現在のポジション状況を表示する。"""
    active = registry.get_active_trades()

    print(f'\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print(f'  Exit Manager — ポジション状況')
    print(f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')

    if not active:
        print('  管理中のトレードはありません。')
        print()
        return

    for t in active:
        try:
            bid, ask = client.get_current_price(t.instrument)
            current_price = ask if t.side == 'long' else bid
        except Exception:
            current_price = t.entry_price
            bid = ask = 0.0

        current_r = t.unrealized_r(current_price)
        jpy_rate = config.get('instruments', {}).get(
            t.instrument, {}
        ).get('jpy_per_dollar_per_unit', 151.8)
        pnl_jpy = t.unrealized_pnl_jpy(current_price, jpy_rate)

        hold_h = (datetime.utcnow() - t.entry_time).total_seconds() / 3600.0

        print(f'\n  [{t.trade_id}] {t.instrument} {t.side.upper()}')
        print(f'    フェーズ:     {t.phase.value}')
        print(f'    エントリー:   ${t.entry_price:.3f}')
        print(f'    現在価格:     ${current_price:.3f}')
        print(f'    SL:           ${t.sl_price:.3f}')
        print(f'    含み損益:     ¥{pnl_jpy:,.0f}  ({current_r:+.2f}R)')
        print(f'    保有時間:     {hold_h:.1f}h')
        print(f'    ユニット:     {t.current_units}')
        if t.non_textbook:
            print(f'    ⚠️ non_textbook フラグあり')

    print()


# ------------------------------------------------------------------ #
#  override コマンド                                                  #
# ------------------------------------------------------------------ #

def cmd_override(args, config, client, registry, logger):
    """
    緊急手動介入（パスワード保護）。

    環境変数 EXIT_MANAGER_OVERRIDE_PASSWORD にパスワードを設定。
    """
    pw_env = 'EXIT_MANAGER_OVERRIDE_PASSWORD'
    expected_pw = os.environ.get(pw_env, '')

    if not expected_pw:
        print(f'\n❌ 環境変数 {pw_env} が設定されていません。')
        print(f'   export {pw_env}=your_password')
        logger.log_validation_warning(
            trade_id=args.trade_id,
            warning='override_pw_env_not_set',
        )
        sys.exit(1)

    entered = input('\nオーバーライドパスワード: ')
    if entered.strip() != expected_pw:
        print('\n❌ パスワードが違います。キャンセルしました。')
        logger.log_validation_warning(
            trade_id=args.trade_id,
            warning='override_auth_failed',
        )
        sys.exit(1)

    trade = registry.get(args.trade_id)
    if not trade:
        print(f'\n⚠️ トレード {args.trade_id} はレジストリに見つかりません（OANDAを直接確認してください）')

    print(f'\n⚠️ 緊急オーバーライド: trade_id={args.trade_id} action={args.action}')
    confirm = input('本当に実行しますか？ [yes/N]: ')
    if confirm.strip().lower() != 'yes':
        print('キャンセルしました。')
        sys.exit(0)

    logger.log(
        'MANUAL_OVERRIDE',
        trade_id=args.trade_id,
        action=args.action,
    )

    if args.action == 'close_all':
        if trade:
            try:
                resp = client.close_position(trade.instrument, trade.side)
                registry.mark_closed(args.trade_id)
                print(f'\n✅ 決済完了: {resp}')
            except Exception as e:
                print(f'\n❌ 決済失敗: {e}')
                sys.exit(1)
        else:
            print('\n⚠️ レジストリにトレードが見つかりません。OANDAダッシュボードで直接決済してください。')


# ------------------------------------------------------------------ #
#  report コマンド                                                    #
# ------------------------------------------------------------------ #

def cmd_report(args, config, client, registry, logger):
    """集計レポートを生成する。"""
    from exit_manager.evaluator import load_events, compute_metrics, generate_report

    log_dir = config['logging']['output_dir']
    jsonl_path = str(Path(log_dir) / 'exit_manager.jsonl')

    events = load_events(jsonl_path, days=args.days)
    if not events:
        print(f'\n⚠️ {args.days}日以内のログが見つかりません: {jsonl_path}')
        return

    metrics = compute_metrics(events)
    report = generate_report(metrics, output_path=args.output)
    print(report)


# ------------------------------------------------------------------ #
#  エントリーポイント                                                  #
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m exit_manager.cli',
        description='Exit Manager CLI — 出口管理ボット',
    )
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')

    # entry コマンド
    p_entry = sub.add_parser('entry', help='新規トレード登録')
    p_entry.add_argument('--trade-id', required=True, help='OANDA トレードID')
    p_entry.add_argument('--reason', default='', help='エントリー理由（ログ用）')

    # status コマンド
    sub.add_parser('status', help='現在のポジション状況を表示')

    # override コマンド
    p_override = sub.add_parser('override', help='緊急手動介入（パスワード保護）')
    p_override.add_argument('--trade-id', required=True, help='OANDA トレードID')
    p_override.add_argument(
        '--action', choices=['close_all'], required=True,
        help='実行するアクション'
    )

    # report コマンド
    p_report = sub.add_parser('report', help='集計レポート生成')
    p_report.add_argument('--days', type=int, default=14, help='集計期間（日数）')
    p_report.add_argument('--output', default=None, help='出力ファイルパス（.md）')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = load_config()

    # レジストリはCLI実行ごとに空（main.pyが永続ループを持つ）
    # CLIは主に entry / status / override / report に使う
    registry = TradeRegistry()
    logger = ExitManagerLogger(log_dir=config['logging']['output_dir'])

    # entry と override は OANDA API を使う
    if args.command in ('entry', 'override', 'status'):
        try:
            client = ExitManagerClient.from_env()
        except EnvironmentError as e:
            print(f'\n❌ {e}')
            sys.exit(1)
    else:
        client = None

    dispatch = {
        'entry':    cmd_entry,
        'status':   cmd_status,
        'override': cmd_override,
        'report':   cmd_report,
    }
    dispatch[args.command](args, config, client, registry, logger)


if __name__ == '__main__':
    main()
