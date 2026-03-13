# コードレビュー報告: OANDA認証環境変数の堅牢化

**EntryID**: 20260305-005  
**ReviewID**: REVIEW-20260305-001  
**対象ブランチ**: `claude/organize-repo-structure-fUJir`  
**レビュー実施**: Manus AI  
**日時**: 2026-03-05  

---

## 1. 調査結果サマリー

Claude Codeが報告した「OANDA認証の環境変数解決を堅牢化」（コミット `228b425`）について、リポジトリ全体を精査した結果、以下が判明した。

| 確認項目 | 結果 |
|---------|------|
| コミット `228b425` の存在 | **存在しない**（GitHub API: `No commit found for SHA: 228b425`） |
| `resolve_oanda_credentials()` 関数 | **存在しない**（全ブランチ・全ファイルで未検出） |
| `OANDA_API_TOKEN` フォールバック（`lib/oanda_client.py`） | **存在しない** |
| `OANDA_ENVIRONMENT` / `OANDA_ENV` 両系統解決 | **存在しない** |
| `tests/test_oanda_env_compat.py` | **存在しない** |
| `scripts/fetch_oanda_15m_universe.py` | **存在しない** |
| `scripts/record_oanda_ticks.py` | **存在しない** |

**結論: Claude Codeが報告した変更は、現時点でリポジトリに存在しない。**

---

## 2. 現状のOANDA認証実装（実際のコード）

### 2.1 `lib/oanda_client.py`（88行）

```python
# 現状の実装（OANDA_API_KEY のみ対応）
self.api_key = os.environ.get("OANDA_API_KEY")
self.environment = os.environ.get("OANDA_ENVIRONMENT", "practice")
if not self.api_key:
    raise ValueError("OANDA_API_KEY が設定されていません。")
```

**問題点**: `OANDA_API_TOKEN`（後方互換）と `OANDA_ENV`（短縮形）に対応していない。

### 2.2 `monitors/forward_main_strategy.py` / `monitors/monitor_union_kelly.py`

```python
# 現状の実装（OANDA_API_TOKEN のみ対応）
OANDA_ENV       = os.environ.get('OANDA_ENV', 'practice')
OANDA_API_TOKEN = os.environ.get('OANDA_API_TOKEN', '')
if not OANDA_API_TOKEN:
    logger.warning('OANDA_API_TOKEN が未設定。ローカルCSVにフォールバック。')
```

**問題点**: `lib/oanda_client.py` は `OANDA_API_KEY` を要求するが、モニタースクリプトは `OANDA_API_TOKEN` を使用しており、**変数名が不統一**。

### 2.3 `exit_manager/oanda_client.py`（別実装）

```python
# exit_manager独自実装（OANDA_API_KEY のみ）
api_key = os.environ.get('OANDA_API_KEY')
environment = os.environ.get('OANDA_ENVIRONMENT', 'practice')
```

**問題点**: `lib/oanda_client.py` と `exit_manager/oanda_client.py` の2つの独立した実装が存在し、保守コストが高い。

---

## 3. 定量的リスク評価

| リスク項目 | 深刻度 | 影響範囲 |
|-----------|--------|---------|
| 変数名不統一（`OANDA_API_KEY` vs `OANDA_API_TOKEN`） | **高** | 既存環境での起動失敗 |
| `resolve_oanda_credentials()` 未実装 | **高** | 両系統解決ができない |
| `OANDA_ENV` / `OANDA_ENVIRONMENT` 不統一 | **中** | 環境切り替え時のバグ |
| 2つの独立した `oanda_client.py` | **中** | 保守コスト増大 |
| `tests/test_oanda_env_compat.py` 未実装 | **中** | 回帰防止なし |

---

## 4. 推奨対応（Manusによる実装）

Claude Codeが報告した変更が存在しないため、Manusが以下を実装することを提案する。

### 4.1 `lib/oanda_client.py` への `resolve_oanda_credentials()` 追加

```python
def resolve_oanda_credentials():
    """
    OANDA認証情報を環境変数から解決する。
    優先順位:
      API KEY: OANDA_API_KEY > OANDA_API_TOKEN（後方互換）
      ENV:     OANDA_ENVIRONMENT > OANDA_ENV（後方互換）
    """
    api_key = os.environ.get("OANDA_API_KEY") or os.environ.get("OANDA_API_TOKEN")
    environment = os.environ.get("OANDA_ENVIRONMENT") or os.environ.get("OANDA_ENV", "practice")
    if not api_key:
        raise ValueError(
            "OANDA APIキーが設定されていません。\n"
            "  推奨: export OANDA_API_KEY='your-api-key'\n"
            "  後方互換: export OANDA_API_TOKEN='your-api-token'（非推奨）"
        )
    return api_key, environment
```

### 4.2 `tests/test_oanda_env_compat.py` の作成

3ケースのユニットテスト:
1. `OANDA_API_KEY` のみ設定 → 正常解決
2. `OANDA_API_TOKEN` のみ設定 → 後方互換で解決
3. 両方未設定 → `ValueError` を発生

---

## 5. 次アクション

水原様のご判断を仰ぎたい。

**選択肢A**: Manusが上記の `resolve_oanda_credentials()` とテストを実装し、`claude/organize-repo-structure-fUJir` ブランチに追加してmainへマージする。

**選択肢B**: Claude Codeに対して「コミット228b425が見つからない。再度プッシュしてほしい」と伝え、Claude Codeの実装を待ってからレビューする。

---

*本レポートはManus AIが実際のリポジトリコードを精査した結果に基づく。*
