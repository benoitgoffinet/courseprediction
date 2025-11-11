FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Outils + runtime OpenMP (scikit-learn / shap) + git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pip/Setuptools/Wheel à jour = wheels manylinux utilisées au lieu de compiler
RUN python -m pip install --upgrade pip setuptools wheel

# (1) Pré-installer les lourds en wheels uniquement (évite la compilation)
RUN pip install --only-binary=:all: --prefer-binary \
    "numpy==1.26.4" \
    "scikit-learn>=1.3,<1.6" \
    "shap>=0.45,<0.46"

# (2) Installer le reste
COPY requirements.txt /app/requirements.txt
RUN pip install --prefer-binary -r /app/requirements.txt

ARG GIT_SHA=dev
RUN echo $GIT_SHA > /app/version.txt
COPY . /app

EXPOSE 8501
CMD sh -c 'streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true'
