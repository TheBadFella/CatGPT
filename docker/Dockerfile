# ── Stage 1: Build CatGPT with jlesage/baseimage-gui ────────────
FROM jlesage/baseimage-gui:debian-12-v4

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Application settings for jlesage GUI
ENV APP_NAME="CatGPT"
ENV KEEP_APP_RUNNING=1

# ── System dependencies ─────────────────────────────────────────
# Install Python 3, venv, development tools, fonts, and Chromium runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    # Chrome runtime dependencies
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    libwayland-client0 \
    # Fonts for page rendering
    fonts-liberation \
    fonts-noto-color-emoji \
    fonts-dejavu-core \
    # Utilities
    curl \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python Virtual Environment ──────────────────────────────────
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ── Application setup ───────────────────────────────────────────
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install patchright's Chromium browser and runtime dependencies
RUN patchright install chromium && patchright install-deps chromium

# Copy application source code and scripts
COPY src/ src/
COPY scripts/ scripts/

# ── Directory setup ─────────────────────────────────────────────
RUN mkdir -p /app/browser_data /app/logs /app/downloads/images /app/downloads/audio

# ── GUI initialization & startup scripts ────────────────────────
COPY docker/10-catgpt-init.sh /etc/cont-init.d/10-catgpt-init.sh
COPY docker/startapp.sh /startapp.sh
RUN chmod +x /etc/cont-init.d/10-catgpt-init.sh /startapp.sh

# ── App Environment Variables ───────────────────────────────────
ENV HEADLESS=false
ENV BROWSER_DATA_DIR=/app/browser_data
ENV LOG_DIR=/app/logs
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV LOG_LEVEL=DEBUG
ENV VERBOSE=true
ENV PYTHONPATH=/app
ENV DISPLAY_WIDTH=1366
ENV DISPLAY_HEIGHT=768

# ── Ports ───────────────────────────────────────────────────────
# 8000: FastAPI server
# 5800: Web GUI (HTTP)
# 5801: Web GUI (HTTPS)
# 5900: VNC
EXPOSE 8000 5800 5801 5900

# ── Health check ────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || curl -f http://localhost:8000/v1/models || exit 1
