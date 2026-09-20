# ---------------------------------------------------------------------------
# CPAK Automatic — imagen de despliegue (Hugging Face Spaces, SDK docker)
# ---------------------------------------------------------------------------
FROM python:3.10-slim

# --- Librerías del sistema que necesita OpenCV (dependencia de ultralytics) ---
# Debian 13 (trixie) renombró libglib2.0-0 -> libglib2.0-0t64 (transición time_t),
# por eso se intenta el nombre nuevo y se cae al antiguo para otras versiones.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && ( apt-get install -y --no-install-recommends libglib2.0-0t64 \
         || apt-get install -y --no-install-recommends libglib2.0-0 ) \
    && rm -rf /var/lib/apt/lists/*

# --- Usuario no-root (recomendado por Spaces para evitar problemas de permisos) ---
RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR $HOME/app

# --- Dependencias Python (se copian primero para aprovechar la caché de capas) ---
COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Código y pesos de los modelos ---
COPY --chown=user:user . .

# Puerto por defecto de Spaces para SDK docker (declarado como app_port en README.md)
EXPOSE 7860

CMD ["streamlit", "run", "scripts/30_streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
