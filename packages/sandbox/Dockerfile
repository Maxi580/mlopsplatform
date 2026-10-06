# Base images come from deploy/values.yaml via install.sh.
ARG UV_IMAGE
ARG PYTHON_IMAGE
FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE}
COPY --from=uv /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Dependencies before the code, so a code change rebuilds only the layers below.
RUN uv sync --frozen --no-dev --package mlp-sandbox --no-install-workspace
COPY packages ./packages
RUN uv sync --frozen --no-dev --package mlp-sandbox
ENV PATH="/app/.venv/bin:$PATH"
# Snippets run as nobody, so they can't change the image.
USER 65534
# The chart sets UVICORN_PORT from deploy/values.yaml.
CMD ["uvicorn", "mlp_sandbox.server:app", "--host", "0.0.0.0"]
