#!/bin/sh

set -e

echo "Starting FastAPI backend..."

python -m uvicorn api.main:app \
    --host 127.0.0.1 \
    --port 8000 &

BACKEND_PID=$!

echo "Waiting for FastAPI to become healthy..."

for i in $(seq 1 60); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" >/dev/null 2>&1; then
        echo "FastAPI is healthy."
        break
    fi

    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "FastAPI process stopped unexpectedly."
        exit 1
    fi

    sleep 1
done

if ! python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" >/dev/null 2>&1; then
    echo "FastAPI failed health check."
    exit 1
fi

echo "Starting Streamlit frontend..."

exec python -m streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-8501}" \
    --server.headless true