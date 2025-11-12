FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8051 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# SHAP utilise OpenMP -> libgomp1 suffit (pas besoin d’outils de build)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installe uniquement à partir des requirements pour profiter du cache Docker
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install --prefer-binary -r requirements.txt

# Puis seulement le code
COPY . .

EXPOSE 8051
CMD ["sh","-lc","python -m streamlit run ${TARGET:-app.py} --server.address=0.0.0.0 --server.port=${PORT:-8051} --server.headless=true"]
