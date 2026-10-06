# Base images come from deploy/values.yaml via install.sh.
ARG UV_IMAGE
ARG PYTHON_IMAGE
FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE}
# Set to a mlp-stages extra: hf or unsloth for a trainer image, evaluate for the stages image.
ARG STAGES_EXTRA=""
# Set for the stages image only: the bfcl-eval release `evaluate` runs.
ARG BFCL_VERSION=""
COPY --from=uv /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Dependencies before the code, so a code change rebuilds only the last layers.
RUN uv sync --frozen --no-dev --package mlp-stages ${STAGES_EXTRA:+--extra "$STAGES_EXTRA"} --no-install-workspace
# speculators derives the draft layers' head_dim from the verifier when its config leaves it unset
# (Qwen2 does), then hands the unset value on anyway, which transformers refuses. The upstream fix
# is this one word; the grep fails the build once a release carries it, so this gets deleted.
RUN if [ "$STAGES_EXTRA" = "evaluate" ]; then \
      CLI="$(find /app/.venv -path '*/speculators/train/cli.py')" \
      && grep -q 'head_dim=head_dim,' "$CLI" \
      && sed -i 's/head_dim=head_dim,/head_dim=resolved_head_dim,/' "$CLI"; \
    fi
# BFCL pins libraries the other harnesses can't share, so it gets a virtualenv of its own, with
# the platform's code to drive it; its qwen-agent imports soundfile without depending on it.
RUN if [ -n "$BFCL_VERSION" ]; then \
      uv venv /opt/bfcl && uv pip install --python /opt/bfcl/bin/python "bfcl-eval==$BFCL_VERSION" soundfile; \
    fi
COPY packages ./packages
RUN uv sync --frozen --no-dev --package mlp-stages ${STAGES_EXTRA:+--extra "$STAGES_EXTRA"} \
    && if [ -n "$BFCL_VERSION" ]; then \
      uv pip install --python /opt/bfcl/bin/python --no-deps ./packages/core ./packages/stages; \
    fi
ENV PATH="/app/.venv/bin:$PATH"
# Unsloth reports usage statistics unless told not to.
ENV UNSLOTH_DISABLE_STATISTICS=1
# Lets fetch download with Xet at full speed.
ENV HF_XET_HIGH_PERFORMANCE=1
