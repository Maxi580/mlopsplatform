# Base images come from deploy/values.yaml via install.sh.
ARG UV_IMAGE
ARG PYTHON_IMAGE
FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE}
# Set to a mlp-stages extra: hf for a trainer image, evaluate for the stages image.
ARG STAGES_EXTRA=""
# Set for the stages image only: the bfcl-eval release `evaluate` runs.
ARG BFCL_VERSION=""
COPY --from=uv /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --frozen --no-dev --package mlp-stages ${STAGES_EXTRA:+--extra "$STAGES_EXTRA"}
# BFCL pins libraries the other harnesses can't share, so it gets a virtualenv of its own, with
# the platform's code to drive it; its qwen-agent imports soundfile without depending on it.
RUN if [ -n "$BFCL_VERSION" ]; then \
      uv venv /opt/bfcl \
      && uv pip install --python /opt/bfcl/bin/python "bfcl-eval==$BFCL_VERSION" soundfile \
      && uv pip install --python /opt/bfcl/bin/python --no-deps ./packages/core ./packages/stages; \
    fi
ENV PATH="/app/.venv/bin:$PATH"
# Lets fetch download with Xet at full speed.
ENV HF_XET_HIGH_PERFORMANCE=1
