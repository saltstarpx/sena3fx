FROM python:3.11-slim

WORKDIR /app

# CA証明書を確実にインストール
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 依存パッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリコード + ブローカーモジュール
COPY main.py .
COPY broker_base.py .
COPY broker_oanda.py .
COPY broker_metaapi.py .
COPY strategies/ ./strategies/

# SSL証明書パスを明示（requests/certifi用）
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Cloud Runのポート
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
