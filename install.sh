#!/usr/bin/env bash
# Installs or upgrades the platform on a single-node host. Usage: ./install.sh <dev|prod>
set -euo pipefail

ENVIRONMENT=${1:-}
[[ $ENVIRONMENT == dev || $ENVIRONMENT == prod ]] || { echo "usage: ./install.sh <dev|prod>" >&2; exit 1; }
source "$(dirname "$0")/deploy/values.sh"

K3S_SOCKET=/run/k3s/containerd/containerd.sock

# Writes stdin to a root-owned file and succeeds only if its content changed.
update_root_file() {
  local content
  content=$(cat)
  [[ -f $1 && $(sudo cat "$1") == "$content" ]] && return 1
  sudo mkdir -p "$(dirname "$1")"
  printf '%s\n' "$content" | sudo tee "$1" >/dev/null
}

install_nvidia_container_toolkit() {
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -q && sudo apt-get install -y -q nvidia-container-toolkit
}

install_gvisor() {
  curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor --yes -o /usr/share/keyrings/gvisor-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" \
    | sudo tee /etc/apt/sources.list.d/gvisor.list >/dev/null
  sudo apt-get update -q && sudo apt-get install -y -q runsc
}

install_nerdctl() {
  local version
  version=$(value .versions.nerdctl)
  curl -fsSL "https://github.com/containerd/nerdctl/releases/download/v$version/nerdctl-$version-linux-amd64.tar.gz" \
    | sudo tar -xz -C /usr/local/bin nerdctl
}

# BuildKit's containerd worker builds straight into k3s's image store.
install_buildkit() {
  local version
  version=$(value .versions.buildkit)
  curl -fsSL "https://github.com/moby/buildkit/releases/download/$version/buildkit-$version.linux-amd64.tar.gz" \
    | sudo tar -xz -C /usr/local
  update_root_file /etc/systemd/system/buildkit.service <<EOF || true
[Unit]
Description=BuildKit
After=k3s.service

[Service]
ExecStart=/usr/local/bin/buildkitd --oci-worker=false --containerd-worker=true --containerd-worker-addr=$K3S_SOCKET --containerd-worker-namespace=k8s.io

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now buildkit
}

install_helm() {
  curl -fsSL "https://get.helm.sh/helm-$(value .versions.helm)-linux-amd64.tar.gz" \
    | sudo tar -xz -C /usr/local/bin --strip-components=1 linux-amd64/helm
}

prepare_host() {
  log "Preparing the host (only what is missing)"
  nvidia-smi >/dev/null || die "nvidia-smi failed: install the NVIDIA driver first"
  local restart_k3s=false
  if ! command -v nvidia-container-runtime >/dev/null; then install_nvidia_container_toolkit; restart_k3s=true; fi
  if ! command -v runsc >/dev/null; then install_gvisor; restart_k3s=true; fi

  update_root_file /etc/rancher/k3s/config.yaml.d/mlp.yaml <<EOF && restart_k3s=true
secrets-encryption: true
default-runtime: nvidia
EOF
  # k3s finds the nvidia runtime by itself, but gVisor needs a containerd template.
  update_root_file /var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.tmpl <<'EOF' && restart_k3s=true
{{ template "base" . }}

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.runsc]
  runtime_type = "io.containerd.runsc.v1"
EOF

  if ! command -v k3s >/dev/null; then
    curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$(value .versions.k3s)" sh -
  elif $restart_k3s; then
    sudo systemctl restart k3s
  fi
  command -v nerdctl >/dev/null || install_nerdctl
  command -v buildkitd >/dev/null || install_buildkit
  command -v helm >/dev/null || install_helm

  mkdir -p "$(dirname "$KUBECONFIG")"
  sudo k3s kubectl config view --raw > "$KUBECONFIG"
  chmod 600 "$KUBECONFIG"
  kubectl wait --for=condition=Ready node --all --timeout=5m
  until kubectl get crd ingressroutes.traefik.io >/dev/null 2>&1; do sleep 5; done
}

build_images() {
  TAG=$(git -C "$ROOT" rev-parse --short HEAD)
  [[ -z $(git -C "$ROOT" status --porcelain) ]] || TAG+="-dirty-$(date +%Y%m%d%H%M%S)"
  log "Building images with tag $TAG"
  # The Python images build from the whole uv workspace, the Web UI from its own directory.
  build_image api "$ROOT" "$ROOT/packages/api/Dockerfile"
  build_image stages "$ROOT" "$ROOT/packages/stages/Dockerfile"
  build_image webui "$ROOT/packages/interface/webui" "$ROOT/packages/interface/webui/Dockerfile"
}

build_image() {
  sudo nerdctl --address "$K3S_SOCKET" --namespace k8s.io build \
    --build-arg PYTHON_IMAGE="$(value .images.python)" --build-arg UV_IMAGE="$(value .images.uv)" \
    --build-arg NODE_IMAGE="$(value .images.node)" --build-arg NGINX_IMAGE="$(value .images.nginx)" \
    -t "$(value ".images.$1"):$TAG" -f "$3" "$2"
}

# Generated once and reused; Kubeflow and MLflow get copies in their own namespaces.
create_credentials() {
  log "Creating namespaces and credentials"
  local platform kubeflow mlflow
  platform=$(value .namespaces.platform) kubeflow=$(value .namespaces.kubeflow) mlflow=$(value .namespaces.mlflow)
  for namespace in "$platform" "$kubeflow" "$mlflow"; do
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
  done
  # Traefik (in kube-system) forwards presigned downloads to SeaweedFS.
  kubectl label namespace "$platform" "$mlflow" "$(value .namespaces.kubeSystem)" mlp-object-store=allowed --overwrite

  kubectl -n "$platform" get secret mlp-credentials >/dev/null 2>&1 || kubectl -n "$platform" create secret generic mlp-credentials \
    --from-literal=postgresPassword="$(openssl rand -hex 24)" \
    --from-literal=s3AccessKey="$(openssl rand -hex 12)" \
    --from-literal=s3SecretKey="$(openssl rand -hex 24)" \
    --from-literal=jwtSecret="$(openssl rand -hex 32)"
  credential() { kubectl -n "$platform" get secret mlp-credentials -o "jsonpath={.data.$1}" | base64 -d; }

  kubectl -n "$kubeflow" create secret generic mlpipeline-minio-artifact --dry-run=client -o yaml \
    --from-literal=accesskey="$(credential s3AccessKey)" --from-literal=secretkey="$(credential s3SecretKey)" \
    | kubectl apply -f -
  kubectl -n "$mlflow" create secret generic mlflow-credentials --dry-run=client -o yaml \
    --from-literal=username="$(value .postgres.user)" --from-literal=password="$(credential postgresPassword)" \
    --from-literal=AWS_ACCESS_KEY_ID="$(credential s3AccessKey)" --from-literal=AWS_SECRET_ACCESS_KEY="$(credential s3SecretKey)" \
    | kubectl apply -f -
}

install_cert_manager() {
  log "Installing cert-manager"
  helm upgrade --install cert-manager cert-manager --repo https://charts.jetstack.io \
    --version "$(value .versions.certManager)" --namespace "$(value .namespaces.certManager)" --create-namespace \
    --set crds.enabled=true --set crds.keep=false --wait
}

install_platform_chart() {
  log "Installing the platform chart (API, Web UI, Postgres, TLS, routes)"
  helm upgrade --install mlp "$ROOT/deploy/chart" --namespace "$(value .namespaces.platform)" \
    -f "$VALUES" --set images.tag="$TAG" --wait --timeout 10m
}

# Asked on the first run only; the API stores just its hash, which survives re-installs.
set_shared_password() {
  local platform accounts password confirmation
  platform=$(value .namespaces.platform)
  # A separate assignment, so a failing kubectl stops the install instead of resetting the password.
  accounts=$(kubectl -n "$platform" exec statefulset/postgres -- psql -U "$(value .postgres.user)" \
    -d "$(value .postgres.database)" -tAc 'SELECT count(*) FROM account')
  [[ $accounts == 1 ]] && return
  log "Setting the shared password"
  password=${MLP_PASSWORD:-}
  if [[ -z $password ]]; then
    read -rsp "Shared password: " password && echo
    read -rsp "Repeat it: " confirmation && echo
    [[ $password == "$confirmation" ]] || die "the passwords differ"
  fi
  printf '%s\n' "$password" | kubectl -n "$platform" exec -i deploy/api -- set-password
}

install_kubeflow_pipelines() {
  log "Installing Kubeflow Pipelines"
  local kubeflow
  kubeflow=$(value .namespaces.kubeflow)
  kubectl apply -k "$(kfp_manifests cluster-scoped-resources)"
  kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
  mkdir -p "$WORK/kfp"
  cat > "$WORK/kfp/kustomization.yaml" <<EOF
resources:
  - $(kfp_manifests env/platform-agnostic)
components:
  - $(realpath --relative-to="$WORK/kfp" "$ROOT/deploy/kfp")
patches:
  - target:
      kind: PersistentVolumeClaim
      name: seaweedfs-pvc
    patch: '[{"op": "replace", "path": "/spec/resources/requests/storage", "value": "$(value .settings.object_store_size)"}]'
EOF
  kubectl apply -k "$WORK/kfp"
  kubectl -n "$kubeflow" rollout status deploy/seaweedfs --timeout=10m
  for bucket in $(value '.buckets[]'); do
    kubectl -n "$kubeflow" exec deploy/seaweedfs -- sh -c \
      "echo s3.bucket.list | weed shell | grep -qw $bucket || echo 's3.bucket.create --name $bucket' | weed shell"
  done
  kubectl -n "$kubeflow" wait --for=condition=Available deploy --all --timeout=15m
}

install_mlflow() {
  log "Installing MLflow"
  value .mlflow > "$WORK/mlflow.yaml"
  helm upgrade --install mlflow mlflow --repo https://community-charts.github.io/helm-charts \
    --version "$(value .versions.mlflowChart)" --namespace "$(value .namespaces.mlflow)" -f "$WORK/mlflow.yaml" \
    --set backendStore.postgres.host="postgres.$(value .namespaces.platform).svc.cluster.local" \
    --set backendStore.postgres.port="$(value .ports.postgres)" \
    --set extraEnvVars.MLFLOW_S3_ENDPOINT_URL="http://seaweedfs.$(value .namespaces.kubeflow).svc.cluster.local:$(value .ports.seaweedfs)" \
    --wait --timeout 10m
}

write_ca_certificate() {
  local cert_manager
  cert_manager=$(value .namespaces.certManager)
  kubectl -n "$cert_manager" wait --for=condition=Ready certificate/mlp-ca --timeout=2m
  kubectl -n "$cert_manager" get secret mlp-ca -o 'jsonpath={.data.ca\.crt}' | base64 -d > "$ROOT/ca.crt"
}

print_urls() {
  local domain
  domain=$(value .domain)
  log "Platform is up"
  echo "Web UI:    https://$domain/"
  echo "API:       https://$domain/health"
  echo "KFP UI:    https://$domain/pipeline/"
  echo "MLflow UI: https://$domain/mlflow/"
  echo "CA:        $ROOT/ca.crt (import it into your OS trust store once)"
}

install_yq
load_values "$ROOT/deploy/values-$ENVIRONMENT.yaml"
prepare_host
build_images
create_credentials
kubectl apply -f "$(device_plugin_manifest)"
install_cert_manager
install_platform_chart
set_shared_password
install_kubeflow_pipelines
install_mlflow
write_ca_certificate
print_urls
