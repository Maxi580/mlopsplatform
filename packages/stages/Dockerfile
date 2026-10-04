# Base images come from deploy/values.yaml via install.sh.
ARG UV_IMAGE
ARG PYTHON_IMAGE
FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE}
# Set to a mlp-stages extra: hf for a trainer image, evaluate for the stages image.
ARG STAGES_EXTRA=""
COPY --from=uv /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --frozen --no-dev --package mlp-stages ${STAGES_EXTRA:+--extra "$STAGES_EXTRA"}
ENV PATH="/app/.venv/bin:$PATH"
# Lets fetch download with Xet at full speed.
ENV HF_XET_HIGH_PERFORMANCE=1
