# ── Stage 1: R base + system deps (cached unless R version changes) ──
FROM rocker/r-ver:4.5.2 AS r-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libfreetype6-dev libpng-dev \
    libtiff5-dev libjpeg-dev pkg-config \
    # rpy2 builds from source and links against R's deps:
    libpcre2-dev libdeflate-dev libzstd-dev liblzma-dev \
    libbz2-dev zlib1g-dev libicu-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: R packages via renv (cached unless renv.lock changes) ──
FROM r-base AS r-deps

COPY docker/renv.lock /tmp/renv.lock
RUN R -e "install.packages('renv', repos='https://cloud.r-project.org')" \
    && mkdir -p /renv && cp /tmp/renv.lock /renv/renv.lock \
    && R -e "renv::consent(provided=TRUE); renv::restore(lockfile='/renv/renv.lock')"

# ── Stage 3: Python deps via uv (cached unless pyproject.toml/uv.lock change) ──
FROM r-deps AS py-deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_PREFERENCE=only-managed

# rocker/r-ver installs R at /usr/local/lib/R (not /usr/lib/R).
# rpy2 needs R_HOME and R on PATH to build from source.
ENV R_HOME=/usr/local/lib/R
ENV LD_LIBRARY_PATH=/usr/local/lib/R/lib:${LD_LIBRARY_PATH}

RUN uv python install 3.13

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --extra dev --extra r --frozen --no-install-project

# ── Stage 4: Source code + project install (rebuilt on code changes) ──
FROM py-deps AS test

COPY jaxgam/ jaxgam/
COPY tests/ tests/
COPY scripts/ scripts/
COPY pyproject.toml uv.lock ./
RUN uv sync --extra dev --extra r --frozen

CMD ["uv", "run", "pytest", "-x", "--tb=short"]
