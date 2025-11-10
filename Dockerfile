FROM python:3.10-slim

# 1) Env de base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# 2) Paquets système nécessaires (compilation éventuelle + git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# 3) D’abord upgrade pip/setuptools/wheel (critique pour wheels manylinux)
RUN python -m pip install --upgrade pip setuptools wheel

# 4) Dépendances Python
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# 5) Version & code
ARG GIT_SHA=dev
RUN echo $GIT_SHA > /app/version.txt
COPY . /app

EXPOSE 8501
CMD ["bash", "-lc", "streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT}"]
