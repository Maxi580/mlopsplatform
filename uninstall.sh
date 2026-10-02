#!/usr/bin/env bash
# Deletes everything the platform created, including all data. Keeps k3s, nerdctl/BuildKit and the driver.
set -euo pipefail

source "$(dirname "$0")/deploy/values.sh"
install_yq
load_values
platform=$(value .namespaces.platform) kubeflow=$(value .namespaces.kubeflow)
mlflow=$(value .namespaces.mlflow) cert_manager=$(value .namespaces.certManager)

domain=$(helm get values mlp --namespace "$platform" 2>/dev/null | yq '.domain // ""')
[[ -n $domain ]] || domain=$platform
read -rp "This deletes the platform and ALL its data. Type '$domain' to confirm: " answer
[[ $answer == "$domain" ]] || die "aborted"

log "Removing releases and manifests"
helm uninstall mlp --namespace "$platform" --ignore-not-found --wait
helm uninstall mlflow --namespace "$mlflow" --ignore-not-found --wait
kubectl delete -k "$(kfp_manifests cluster-scoped-resources)" --ignore-not-found
helm uninstall cert-manager --namespace "$cert_manager" --ignore-not-found --wait
kubectl delete -f "$(device_plugin_manifest)" --ignore-not-found

log "Removing namespaces and volumes"
namespaces=("$platform" "$kubeflow" "$mlflow" "$cert_manager")
kubectl delete namespace "${namespaces[@]}" --ignore-not-found
for namespace in "${namespaces[@]}"; do
  kubectl get pv -o yaml \
    | yq ".items[] | select(.spec.claimRef.namespace == \"$namespace\") | .metadata.name" \
    | xargs -r kubectl delete pv
done
# A host directory, so deleting its volume leaves the files.
sudo rm -rf "$(value .modelCacheHostPath)"

log "Removing images and build cache"
sudo k3s crictl rmi --prune
sudo buildctl prune --all
rm -f "$ROOT/ca.crt"
log "Platform removed"
