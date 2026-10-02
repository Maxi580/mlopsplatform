# Base images come from deploy/values.yaml via install.sh.
ARG UV_IMAGE
ARG PYTHON_IMAGE
FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE}
COPY --from=uv /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --frozen --no-dev --package mlp-api
ENV PATH="/app/.venv/bin:$PATH"
# The chart sets UVICORN_PORT from deploy/values.yaml.
CMD ["uvicorn", "mlp_api.app:app", "--host", "0.0.0.0"]
