#!/bin/bash
# deploy_gcp.sh - GCP Cloud Run 全自動デプロイスクリプト
# =====================================================
# 使い方: bash deploy/deploy_gcp.sh
#
# 前提:
#   1. gcloud CLI がインストール・認証済み
#   2. deploy/.env に環境変数を設定済み
#   3. GCPプロジェクトが作成済み
#
# 実行内容:
#   1. Cloud Run サービスをビルド＆デプロイ
#   2. Cloud Scheduler ジョブを作成（1分毎にトリガー）
#   3. GCS バケットを作成（状態管理用）
set -euo pipefail

# ── 設定読み込み ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: ${ENV_FILE} が見つかりません"
    echo "  cp deploy/.env.example deploy/.env して設定してください"
    exit 1
fi

source "$ENV_FILE"

PROJECT_ID="${GCP_PROJECT:-aiyagami}"
REGION="${GCP_REGION:-asia-northeast1}"
SERVICE_NAME="${SERVICE_NAME:-sena3fx-trader}"
GCS_BUCKET="${GCS_BUCKET:-sena3fx-paper-trading}"
SCHEDULER_NAME="${SCHEDULER_NAME:-sena3fx-run-cycle}"

echo "========================================"
echo "YAGAMI改 Cloud Run デプロイ"
echo "========================================"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Service:  ${SERVICE_NAME}"
echo "  Broker:   ${BROKER:-oanda}"
echo "  Bucket:   ${GCS_BUCKET}"
echo "========================================"

# ── 1. GCSバケット作成（存在しなければ） ──────────────
echo "[1/4] GCSバケット確認..."
if ! gsutil ls -b "gs://${GCS_BUCKET}" >/dev/null 2>&1; then
    gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${GCS_BUCKET}"
    echo "  バケット作成完了: ${GCS_BUCKET}"
else
    echo "  バケット既存: ${GCS_BUCKET}"
fi

# ── 2. Cloud Run ビルド＆デプロイ ─────────────────────
echo "[2/4] Cloud Run ビルド＆デプロイ..."
cd "${SCRIPT_DIR}/../cloud_run"

# 環境変数を構築（ブローカー種別に応じて）
ENV_VARS="DISCORD_WEBHOOK=${DISCORD_WEBHOOK},GCS_BUCKET=${GCS_BUCKET},GCP_PROJECT=${PROJECT_ID},BROKER=${BROKER:-oanda}"

if [ "${BROKER:-oanda}" = "exness" ]; then
    ENV_VARS="${ENV_VARS},METAAPI_TOKEN=${METAAPI_TOKEN:-},METAAPI_ACCOUNT_ID=${METAAPI_ACCOUNT_ID:-},EQUITY_JPY=${EQUITY_JPY:-1000000}"
else
    ENV_VARS="${ENV_VARS},OANDA_TOKEN=${OANDA_TOKEN:-},OANDA_ACCOUNT=${OANDA_ACCOUNT:-}"
fi

gcloud run deploy "${SERVICE_NAME}" \
    --source . \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars "${ENV_VARS}" \
    --memory 512Mi \
    --timeout 60 \
    --min-instances 1 \
    --max-instances 1

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --format "value(status.url)")
echo "  デプロイ完了: ${SERVICE_URL}"

# ── 3. Cloud Scheduler ジョブ作成 ─────────────────────
echo "[3/4] Cloud Scheduler 設定..."

# メインサイクル（毎分実行）
if gcloud scheduler jobs describe "${SCHEDULER_NAME}" \
    --project "${PROJECT_ID}" --location "${REGION}" >/dev/null 2>&1; then
    gcloud scheduler jobs update http "${SCHEDULER_NAME}" \
        --project "${PROJECT_ID}" \
        --location "${REGION}" \
        --schedule "* * * * *" \
        --uri "${SERVICE_URL}/run" \
        --http-method POST \
        --attempt-deadline 60s
    echo "  Scheduler更新: ${SCHEDULER_NAME}"
else
    gcloud scheduler jobs create http "${SCHEDULER_NAME}" \
        --project "${PROJECT_ID}" \
        --location "${REGION}" \
        --schedule "* * * * *" \
        --uri "${SERVICE_URL}/run" \
        --http-method POST \
        --attempt-deadline 60s \
        --time-zone "UTC"
    echo "  Scheduler作成: ${SCHEDULER_NAME}"
fi

# 朝9時レポートは /run サイクル内で自動送信（JST 9:00検知）。
# Cloud Scheduler からの /report 呼び出しは重複の原因になるため廃止。
# 旧ジョブが残っている場合は削除する。
REPORT_JOB="${SERVICE_NAME}-report"
if gcloud scheduler jobs describe "${REPORT_JOB}" \
    --project "${PROJECT_ID}" --location "${REGION}" >/dev/null 2>&1; then
    gcloud scheduler jobs delete "${REPORT_JOB}" \
        --project "${PROJECT_ID}" \
        --location "${REGION}" \
        --quiet
    echo "  旧レポートScheduler削除: ${REPORT_JOB}（/run内で自動送信に統一）"
else
    echo "  朝レポート: /run サイクル内で JST 9:00 に自動送信"
fi

# 週次フィードバック（月曜 JST 0:00 = 日曜 UTC 15:00）
WEEKLY_JOB="${SERVICE_NAME}-weekly"
if gcloud scheduler jobs describe "${WEEKLY_JOB}" \
    --project "${PROJECT_ID}" --location "${REGION}" >/dev/null 2>&1; then
    gcloud scheduler jobs update http "${WEEKLY_JOB}" \
        --project "${PROJECT_ID}" \
        --location "${REGION}" \
        --schedule "0 15 * * 0" \
        --uri "${SERVICE_URL}/weekly_feedback" \
        --http-method POST \
        --attempt-deadline 30s
else
    gcloud scheduler jobs create http "${WEEKLY_JOB}" \
        --project "${PROJECT_ID}" \
        --location "${REGION}" \
        --schedule "0 15 * * 0" \
        --uri "${SERVICE_URL}/weekly_feedback" \
        --http-method POST \
        --attempt-deadline 30s \
        --time-zone "UTC"
fi
echo "  週次FB: 毎週月曜 JST 0:00"

# ── 4. ヘルスチェック ─────────────────────────────────
echo "[4/4] ヘルスチェック..."
HEALTH=$(curl -s "${SERVICE_URL}/health" 2>/dev/null || echo '{"status":"error"}')
echo "  ${HEALTH}"

echo ""
echo "========================================"
echo "デプロイ完了!"
echo "========================================"
echo "  URL:       ${SERVICE_URL}"
echo "  ヘルス:    ${SERVICE_URL}/health"
echo "  ステータス: ${SERVICE_URL}/status"
echo "  Scheduler: 毎分 POST ${SERVICE_URL}/run"
echo ""
echo "  ※ PC電源オフでも24時間自動稼働します"
echo "========================================"
