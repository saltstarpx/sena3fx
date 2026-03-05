"""
OANDA認証環境変数の優先順位ユニットテスト
=========================================
テストケース:
  Case1: OANDA_API_KEY のみ設定 → api_key に OANDA_API_KEY が使われる
  Case2: OANDA_API_TOKEN のみ設定（後方互換） → api_key に OANDA_API_TOKEN が使われる
  Case3: 両方設定 → OANDA_API_KEY が優先される

  Case4: OANDA_ENVIRONMENT のみ設定 → environment に OANDA_ENVIRONMENT が使われる
  Case5: OANDA_ENV のみ設定（後方互換） → environment に OANDA_ENV が使われる
  Case6: 両方設定 → OANDA_ENVIRONMENT が優先される

  Case7: api_key 未設定 → ValueError が発生する
  Case8: account_id 未設定 → ValueError が発生する
"""
import os
import sys
import unittest

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.oanda_client import resolve_oanda_credentials


def _clean_env(*keys):
    """指定したキーを環境変数から削除するコンテキストマネージャ用ヘルパー"""
    saved = {k: os.environ.pop(k, None) for k in keys}
    return saved


def _restore_env(saved):
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


class TestOandaEnvCompat(unittest.TestCase):

    ENV_KEYS = [
        'OANDA_API_KEY', 'OANDA_API_TOKEN',
        'OANDA_ENVIRONMENT', 'OANDA_ENV',
        'OANDA_ACCOUNT_ID',
    ]

    def setUp(self):
        """テスト前に全OANDA環境変数を退避・削除"""
        self._saved = _clean_env(*self.ENV_KEYS)

    def tearDown(self):
        """テスト後に環境変数を復元"""
        _restore_env(self._saved)

    # ─── Case 1: OANDA_API_KEY のみ ───────────────────────────────────
    def test_case1_api_key_only(self):
        """OANDA_API_KEY のみ設定 → api_key に OANDA_API_KEY が使われる"""
        os.environ['OANDA_API_KEY'] = 'key_value'
        os.environ['OANDA_ACCOUNT_ID'] = 'acct_001'
        creds = resolve_oanda_credentials()
        self.assertEqual(creds['api_key'], 'key_value',
                         "OANDA_API_KEY が api_key に反映されていない")

    # ─── Case 2: OANDA_API_TOKEN のみ（後方互換） ─────────────────────
    def test_case2_api_token_fallback(self):
        """OANDA_API_TOKEN のみ設定 → api_key に OANDA_API_TOKEN が使われる（後方互換）"""
        os.environ['OANDA_API_TOKEN'] = 'token_value'
        os.environ['OANDA_ACCOUNT_ID'] = 'acct_001'
        creds = resolve_oanda_credentials()
        self.assertEqual(creds['api_key'], 'token_value',
                         "OANDA_API_TOKEN フォールバックが機能していない")

    # ─── Case 3: 両方設定 → OANDA_API_KEY が優先 ─────────────────────
    def test_case3_api_key_takes_priority(self):
        """OANDA_API_KEY と OANDA_API_TOKEN が両方設定 → OANDA_API_KEY が優先される"""
        os.environ['OANDA_API_KEY'] = 'key_wins'
        os.environ['OANDA_API_TOKEN'] = 'token_loses'
        os.environ['OANDA_ACCOUNT_ID'] = 'acct_001'
        creds = resolve_oanda_credentials()
        self.assertEqual(creds['api_key'], 'key_wins',
                         "OANDA_API_KEY が OANDA_API_TOKEN より優先されていない")

    # ─── Case 4: OANDA_ENVIRONMENT のみ ──────────────────────────────
    def test_case4_environment_only(self):
        """OANDA_ENVIRONMENT のみ設定 → environment に反映される"""
        os.environ['OANDA_API_KEY'] = 'key_value'
        os.environ['OANDA_ACCOUNT_ID'] = 'acct_001'
        os.environ['OANDA_ENVIRONMENT'] = 'live'
        creds = resolve_oanda_credentials()
        self.assertEqual(creds['environment'], 'live',
                         "OANDA_ENVIRONMENT が environment に反映されていない")

    # ─── Case 5: OANDA_ENV のみ（後方互換） ──────────────────────────
    def test_case5_env_fallback(self):
        """OANDA_ENV のみ設定 → environment に OANDA_ENV が使われる（後方互換）"""
        os.environ['OANDA_API_KEY'] = 'key_value'
        os.environ['OANDA_ACCOUNT_ID'] = 'acct_001'
        os.environ['OANDA_ENV'] = 'practice'
        creds = resolve_oanda_credentials()
        self.assertEqual(creds['environment'], 'practice',
                         "OANDA_ENV フォールバックが機能していない")

    # ─── Case 6: 両方設定 → OANDA_ENVIRONMENT が優先 ─────────────────
    def test_case6_environment_takes_priority(self):
        """OANDA_ENVIRONMENT と OANDA_ENV が両方設定 → OANDA_ENVIRONMENT が優先される"""
        os.environ['OANDA_API_KEY'] = 'key_value'
        os.environ['OANDA_ACCOUNT_ID'] = 'acct_001'
        os.environ['OANDA_ENVIRONMENT'] = 'live'
        os.environ['OANDA_ENV'] = 'practice'
        creds = resolve_oanda_credentials()
        self.assertEqual(creds['environment'], 'live',
                         "OANDA_ENVIRONMENT が OANDA_ENV より優先されていない")

    # ─── Case 7: api_key 未設定 → ValueError ─────────────────────────
    def test_case7_missing_api_key_raises(self):
        """api_key が未設定の場合 ValueError が発生する"""
        os.environ['OANDA_ACCOUNT_ID'] = 'acct_001'
        with self.assertRaises(ValueError) as ctx:
            resolve_oanda_credentials()
        self.assertIn('OANDA API', str(ctx.exception),
                      "ValueError のメッセージに 'OANDA API' が含まれていない")

    # ─── Case 8: account_id 未設定 → ValueError ──────────────────────
    def test_case8_missing_account_id_raises(self):
        """account_id が未設定の場合 ValueError が発生する"""
        os.environ['OANDA_API_KEY'] = 'key_value'
        with self.assertRaises(ValueError) as ctx:
            resolve_oanda_credentials()
        self.assertIn('OANDA_ACCOUNT_ID', str(ctx.exception),
                      "ValueError のメッセージに 'OANDA_ACCOUNT_ID' が含まれていない")

    # ─── Case 9: デフォルト environment は 'practice' ─────────────────
    def test_case9_default_environment_is_practice(self):
        """OANDA_ENVIRONMENT も OANDA_ENV も未設定 → デフォルトは 'practice'"""
        os.environ['OANDA_API_KEY'] = 'key_value'
        os.environ['OANDA_ACCOUNT_ID'] = 'acct_001'
        creds = resolve_oanda_credentials()
        self.assertEqual(creds['environment'], 'practice',
                         "デフォルト environment が 'practice' でない")


if __name__ == '__main__':
    unittest.main(verbosity=2)
