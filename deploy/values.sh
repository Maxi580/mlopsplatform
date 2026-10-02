# shellcheck shell=bash
# Reads deploy/values*.yaml for install.sh and uninstall.sh, plus the upstream manifest URLs named there.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
VALUES="$WORK/values.yaml"
export KUBECONFIG="$HOME/.kube/mlp.yaml"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# Merges deploy/values.yaml with the given per-environment overrides into $VALUES.
load_values() {
  yq eval-all '. as $file ireduce ({}; . * $file)' "$ROOT/deploy/values.yaml" "$@" > "$VALUES"
}

value() { yq "$1" "$VALUES"; }

# Ubuntu's apt package named yq is a different tool, so check for mikefarah's.
install_yq() {
  yq --version 2>/dev/null | grep -q mikefarah && return
  sudo curl -fsSL -o /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
  sudo chmod +x /usr/local/bin/yq
}

kfp_manifests() { echo "github.com/kubeflow/pipelines/manifests/kustomize/$1?ref=$(value .versions.kfp)"; }

device_plugin_manifest() {
  echo "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/$(value .versions.nvidiaDevicePlugin)/deployments/static/nvidia-device-plugin.yml"
}
