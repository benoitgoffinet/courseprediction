FROM python:3.10-slim

# --- Runtime et perfs ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    # Port par défaut (Azure fournit PORT, on garde 8051 en fallback)
    PORT=8051 \
    # Streamlit: mode headless et un poil moins verbeux
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Outils + OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl ca-certificates wget libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Wheels lourdes en binaire uniquement
RUN pip install --only-binary=:all: --prefer-binary \
    "numpy==1.26.4" \
    "scikit-learn>=1.3,<1.6" \
    "shap>=0.45,<0.46"

# Dépendances applis
COPY requirements.txt /app/requirements.txt
RUN pip install --prefer-binary -r /app/requirements.txt

# Version
ARG GIT_SHA=dev
RUN echo $GIT_SHA > /app/version.txt

# Code
COPY . /app

# Expose (informative)
EXPOSE 8051

# --- HEALTHCHECK Docker (ping la racine locale) ---
# NB: Azure Web App n'utilise PAS ce HEALTHCHECK pour son "Health Check" portail,
# mais c'est utile pour que le moteur de conteneur sache si l'app est up.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD wget -q -O- "http://127.0.0.1:${PORT:-8051}/" > /dev/null || exit 1

# --- Lancement Streamlit ---
# Essaie app.py, sinon app/app.py ; échoue proprement si rien n'existe.
CMD sh -lc '\
  set -e; \
  TARGET=""; \
  if [ -f app.py ]; then TARGET=app.py; \
  elif [ -f app/app.py ]; then TARGET=app/app.py; \
  else echo "❌ app.py introuvable (ni ./app.py ni ./app/app.py). Contenu /app :" && ls -la && exit 2; fi; \
  exec streamlit run "$TARGET" --server.address=0.0.0.0 --server.port="${PORT:-8051}" --server.headless=true \
'
