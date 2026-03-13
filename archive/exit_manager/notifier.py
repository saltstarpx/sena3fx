"""
Exit Manager — Discord 通知（オプション）
==========================================
config.yaml の notification.enabled=true の場合にのみ送信。
Webhook URLは環境変数 DISCORD_WEBHOOK_URL から取得。

送信対象イベント:
  - TP1_HIT
  - MAX_LOSS_GUARD
  - TRADE_CLOSED
  - LOCKOUT_BLOCKED

stdlib の urllib のみ使用（追加依存なし）。
"""

import json
import logging
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

log = logging.getLogger('sena3fx.exit_manager.notifier')

# Discord 通知を送信するイベント種別
NOTIFY_EVENTS = {'TP1_HIT', 'MAX_LOSS_GUARD', 'TRADE_CLOSED', 'LOCKOUT_BLOCKED'}


class DiscordNotifier:
    """
    Discord Webhook 通知クラス。

    config.yaml の notification セクション:
        enabled: false          # true で有効化
        discord_webhook_url:    # コードには書かない

    Webhook URL は環境変数 DISCORD_WEBHOOK_URL を優先。
    未設定の場合は config の discord_webhook_url を使用。

    Usage:
        notifier = DiscordNotifier(config)
        notifier.notify('TP1_HIT', trade_id='12345', price=2905.5, pnl_jpy=75000)
    """

    def __init__(self, config: dict):
        notif_cfg = config.get('notification', {})
        self.enabled = False # ユーザーの指示によりDiscord通知を無効化
        self.webhook_url = (
            os.environ.get('DISCORD_WEBHOOK_URL')
            or notif_cfg.get('discord_webhook_url', '')
        )
        self.notify_on = set(notif_cfg.get('notify_on', list(NOTIFY_EVENTS)))

    def notify(self, event: str, trade_id: str = '', **kwargs) -> bool:
        """
        イベントを Discord に通知する。

        Args:
            event:    イベント種別（NOTIFY_EVENTS のいずれか）
            trade_id: OANDA トレードID（表示用）
            **kwargs: イベント固有のフィールド

        Returns:
            bool: True = 送信成功（または無効化中）, False = 送信失敗
        """
        if not self.enabled:
            return True
        if event not in self.notify_on:
            return True
        if not self.webhook_url:
            log.warning('Discord 通知が有効ですが DISCORD_WEBHOOK_URL が未設定です')
            return False

        content = self._format_message(event, trade_id, **kwargs)
        return self._send(content)

    def _format_message(self, event: str, trade_id: str, **kwargs) -> str:
        """Discord メッセージ本文を整形する。"""
        emoji_map = {
            'TP1_HIT':       '🎯',
            'MAX_LOSS_GUARD': '🚨',
            'TRADE_CLOSED':  '✅',
            'LOCKOUT_BLOCKED': '🔒',
        }
        emoji = emoji_map.get(event, '📊')
        lines = [f'{emoji} **[ExitManager] {event}**']
        if trade_id:
            lines.append(f'trade_id: `{trade_id}`')
        for k, v in kwargs.items():
            if isinstance(v, float):
                lines.append(f'{k}: {v:,.2f}')
            else:
                lines.append(f'{k}: {v}')
        return '\n'.join(lines)

    def _send(self, content: str) -> bool:
        """Discord Webhook に POST リクエストを送信する。"""
        payload = json.dumps({'content': content}).encode('utf-8')
        req = Request(
            self.webhook_url,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urlopen(req, timeout=5) as resp:
                if resp.status not in (200, 204):
                    log.warning(f'Discord Webhook がHTTP {resp.status} を返しました')
                    return False
            return True
        except URLError as e:
            log.warning(f'Discord 通知失敗: {e}')
            return False
        except Exception as e:
            log.warning(f'Discord 通知例外: {e}')
            return False
