FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System packages needed for Python packages and FastEmbed/ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src
COPY app.py .
COPY start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Download/cache the FastEmbed model during image build
RUN python -c "from src.rag_pipeline import warm_up; warm_up()"

EXPOSE 8501

CMD ["./start.sh"]