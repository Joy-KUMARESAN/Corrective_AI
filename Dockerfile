# ---------- Base ----------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (curl for healthchecks; build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# --- Security: upgrade pip/setuptools/wheel to safe versions (fixes CVE) ---
RUN python -m pip install --no-cache-dir --upgrade \
    "pip>=25.3" "setuptools>=75.8" "wheel>=0.45"

# ---------- Workdir ----------
WORKDIR /app

# ---------- Python deps first (cache-friendly) ----------
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# ---------- App code ----------
COPY app.py /app/
COPY src /app/src
# (Best practice: do NOT bake secrets in the image; set env at deploy time)
# COPY .env /app/.env   # <-- removed on purpose

# --- Run as non-root (create dedicated user) ---
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ---------- Streamlit config ----------
EXPOSE 8501

# Healthcheck (Streamlit)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ---------- Run ----------
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0","--browser.gatherUsageStats=false"]
