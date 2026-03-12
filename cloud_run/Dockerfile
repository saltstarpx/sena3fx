FROM python:3.11-slim

WORKDIR /app

# 依存パッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリコード + ブローカーモジュール
COPY main.py .
COPY broker_base.py .
COPY broker_oanda.py .
COPY broker_metaapi.py .
COPY strategies/ ./strategies/

# Cloud Runのポート
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
