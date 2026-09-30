# Which Kubeflow components earn their place?

Research for [#2](https://github.com/Maxi580/mlopsplatform/issues/2). Facts checked on 2026-10-01 against these versions: Kubeflow Pipelines (KFP) 2.17.2, Kubeflow Trainer v2.3.0, Katib v0.19.0, KServe v0.21.0, Kueue v0.20.0 and Kubeflow community-distribution 26.03.1 / master. Terms follow `CONTEXT.md`.

## Question

Which Kubeflow components should the platform use, and which should it leave out? The candidates are Kubeflow Pipelines v2, Kubeflow Trainer (formerly Training Operator), Katib and KServe, on k3s and on a single-node GPU VM. For each one we need the install footprint, whether it gives us the pipeline graph and log view, how it requests and waits for GPUs, its licence and its maintenance state. The answer must name the minimal set, and say whether Stages should be plain pipeline components or hand off to Trainer or KServe.

## Recommendation

**Minimal set: Kubeflow Pipelines standalone, plus KServe in Standard mode for the `serve` Stage. Leave out Trainer, Katib and the rest of the Kubeflow platform. Add Kueue for GPU queueing. Kueue is a Kubernetes SIG project, not a Kubeflow one.**

1. **Kubeflow Pipelines 2.17.x, standalone, single-user.** It is the part the boss values: the run graph and the per-task log view. It does not need Istio, Dex, oauth2-proxy or the Central Dashboard. Install the `env/platform-agnostic` overlay, not `env/dev`. The `env/dev` overlay retags every KFP image to `master` and adds a GCP inverse proxy. KFP brings Argo Workflows (controller only), ML Metadata (MLMD), SeaweedFS and MySQL. Whether we can swap SeaweedFS and MySQL for the shared object store and Postgres belongs to #3. One caveat for #3 already: KFP's PostgreSQL support is not production-ready yet (see Findings).
2. **Stages are plain KFP container components.** `distill`, `finetune` and `evaluate` each run as one KFP task, which is one pod. The pod requests its GPU with `set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(n)`. Only this keeps every Stage's logs in the KFP log view and its node in the graph. It also lets KFP 2.17's built-in MLflow plugin create one MLflow Run per Pipeline and a nested Run per Stage. A single-node, multi-GPU finetune runs `torchrun`/`accelerate` inside that one pod, and needs no Trainer.
3. **Don't hand off to Trainer in v1.** Trainer's value is multi-node distributed training (TrainJob on JobSet). A single VM doesn't need it. KFP has no native TrainJob task: a component must create the CR itself and poll it. The training pods' logs would then live outside the KFP log view.
   **The `finetune` Stage:** #4 chose TRL + PEFT on Accelerate. Trainer's only BuiltinTrainer is TorchTune, and TorchTune says it "is no longer actively maintained" ([torchtune README](https://github.com/meta-pytorch/torchtune/blob/main/README.md)). So Trainer could only run our own container, under its `torch_distributed` or `deepspeed_distributed` runtime ([runtimes](https://github.com/kubeflow/trainer/tree/v2.3.0/manifests/base/runtimes)). The answer by case:
   - **Single GPU (now):** a plain KFP container step. It uses our TRL image and `set_accelerator_limit(1)`.
   - **Multi-GPU on one node (later, if the VM has more than one GPU):** still a plain KFP container step. It uses `set_accelerator_limit(n)` and runs `accelerate launch --num_processes n` (or DeepSpeed through Accelerate) inside the one pod. Trainer adds nothing on a single node.
   - **Multi-node (only if the platform ever spans several GPU nodes):** at that point the `finetune` component creates a TrainJob with the same custom TRL image on the `torch_distributed` runtime. It forwards resources through `dsl.TaskConfig`, and Kueue queues the TrainJob natively. The Stage's inputs and outputs stay the same, so the switch touches only that one component.
4. **`serve` hands off to KServe.** A served model must outlive its Pipeline, so a KFP task can't hold it. The `serve` Stage is a short KFP component. It applies an `InferenceService` in **Standard (raw Deployment) mode**, waits until it is Ready and outputs the endpoint URL. Standard mode needs only cert-manager and a Gateway API or Ingress controller. It needs no Knative and no Istio. This choice is provisional: if #6 picks an engine that KServe's runtimes don't fit, the fallback is a plain Deployment that the same component manages. What happens to the served model afterwards is #11.
5. **Leave out:** Katib (sweeps are out of scope, and the project has been quiet since its last release in Oct 2025), Trainer (see point 3), the Central Dashboard, Notebooks, Kubeflow Hub (model registry), Spark Operator, Istio, Dex, oauth2-proxy and Knative.
6. **GPU queueing: Kueue with its `pod` and `deployment` integrations.** Label each GPU Stage's pod with `kueue.x-k8s.io/queue-name` through `kfp.kubernetes.add_pod_label`. One ClusterQueue holds an `nvidia.com/gpu` quota equal to the node's GPUs. Kueue gates each labelled pod until quota is free and admits pods in queue order. Serving Deployments can be labelled too, so a running model counts against the same GPU quota. Without Kueue, a GPU pod just sits `Pending` until the kube-scheduler finds a free GPU. That works on one GPU, but it gives no ordering, no queue position to show the user, and no priorities between `serve` and training. The known limitation: the queue works per pod, not per Pipeline. Earlier CPU-only tasks run, and then the Pipeline waits at its GPU Stage.

Rough footprint of this set, from the community-distribution resource table plus Kueue's chart defaults: KFP 970m CPU / 3.5 GiB RAM / 35 GB PVC, KServe 600m / 1.2 GiB, Kueue about 500m / 512 MiB requested, plus cert-manager 3m / 128 MiB. That is about 2 CPU and 5.5 GiB. The full Kubeflow platform needs about 4.4 CPU and 12 GiB.

## Findings per component

### Kubeflow Pipelines (KFP) v2: keep

- **Version and maintenance.** Latest release 2.17.2 (2026-09-04), 2.17.0 on 2026-07-09. The repo is pushed daily. A `2.18.0` milestone is open, and a `KFP 3.0` milestone is due 2026-11-06. [releases](https://github.com/kubeflow/pipelines/releases), [milestones](https://github.com/kubeflow/pipelines/milestones)
- **Standalone install, no Istio or Dex.** The official docs offer "Kubeflow Pipelines standalone" as an alternative to the full distribution. The install is two `kubectl apply -k` steps: `cluster-scoped-resources`, then an env overlay. Istio is needed only for the *multi-user* Kubernetes-native overlay. [KFP installation docs](https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/operator-guides/installation/_index.md)
- **What the install contains.** `env/platform-agnostic` = generic KFP install + MLMD + Argo Workflows (namespace install, **without Argo Server**) + SeaweedFS + MySQL. [platform-agnostic kustomization](https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/env/platform-agnostic/kustomization.yaml), [argo namespace install](https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/third-party/argo/installs/namespace/kustomization.yaml). The images come from `ghcr.io/kubeflow/kfp-*` pinned to `2.17.2`, `ghcr.io/chrislusf/seaweedfs:4.34`, `mysql:8.4` and `gcr.io/tfx-oss-public/ml_metadata_store_server:1.14.0`. [pipeline kustomization](https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/base/pipeline/kustomization.yaml), [seaweedfs deployment](https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/third-party/seaweedfs/base/seaweedfs/seaweedfs-deployment.yaml), [mysql deployment](https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/third-party/mysql/base/mysql-deployment.yaml), [metadata-grpc deployment](https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/base/metadata/base/metadata-grpc-deployment.yaml)
- **Watch out: the documented `env/dev` overlay retags all KFP images to `master`** and adds `gcp/inverse-proxy`. Use `env/platform-agnostic` instead. [env/dev kustomization](https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/env/dev/kustomization.yaml)
- **Footprint.** 970m CPU, 3552 Mi memory, 35 GB PVC ("maximum of actual usage and configured requests"). [community-distribution README](https://github.com/kubeflow/community-distribution/blob/master/README.md#kubeflow-components-versions)
- **Database.** The docs say: "PostgreSQL configuration is available only for development and evaluation; it is not supported for production KFP 2.18 deployments ... Use the default MySQL-backed deployment for production." The cause is MLMD's PostgreSQL backend, where concurrent type creation can fail runs. [installation docs](https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/operator-guides/installation/_index.md), [kubeflow/pipelines#14353](https://github.com/kubeflow/pipelines/issues/14353) (open, 2026-09-09). The `platform-agnostic-postgresql` overlay already exists in 2.17.2. [overlay](https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/env/platform-agnostic-postgresql/kustomization.yaml). MLMD removal is planned. The proposal says MLMD "blocks MySQL upgrades beyond 8.x and PostgreSQL support" and is "in maintenance mode". [MLMD removal proposal](https://github.com/kubeflow/pipelines/blob/master/proposals/12147-mlmd-removal/README.md). This matters for #3.
- **Kubernetes-native API mode** stores pipeline definitions as CRs rather than in the database. It needs cert-manager, and the docs label it "non-production". [installation docs](https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/operator-guides/installation/_index.md)
- **Graph and log view.** The KFP UI (`ml-pipeline-ui`) is part of the standalone install and is reached by port-forward or ingress. [installation docs](https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/operator-guides/installation/_index.md). Logs stay available after pods are cleaned up, for two reasons. First, KFP configures Argo with `archiveLogs: true` to the S3 artifact repository. Second, it sets a 1-hour workflow `ttlStrategy` and a default `retryStrategy` limit of 2 `OnError`. [workflow-controller configmap patch](https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/third-party/argo/base/workflow-controller-configmap-patch.yaml). Since 2.17.0 the launcher also writes per-retry `executor-logs-N` artifacts to the object store. [kubeflow/pipelines#13175](https://github.com/kubeflow/pipelines/pull/13175). Only pods that KFP itself runs appear in the log view.
- **MLflow.** KFP 2.17.0 merged a built-in MLflow plugin ([#13441](https://github.com/kubeflow/pipelines/pull/13441), [2.17.0 notes](https://github.com/kubeflow/pipelines/releases/tag/2.17.0)). With `plugins.mlflow` in the API server config, KFP creates an MLflow parent Run per pipeline run and a nested Run per task. It logs scalar metrics and parameters, and injects `MLFLOW_TRACKING_URI`/`MLFLOW_RUN_ID` into task containers. Auth can be `kubernetes`, `basic-auth`, `bearer` or `none`. [MLflow plugin operator guide](https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/operator-guides/mlflow-plugin.md), [KEP-12862](https://github.com/kubeflow/pipelines/blob/master/proposals/12862-mlflow-integration/README.md)
- **GPU requests.** In the SDK, `PipelineTask` has `set_accelerator_type`, `set_accelerator_limit`, CPU/memory request and limit setters, and `set_retry`. [pipeline_task.py](https://github.com/kubeflow/pipelines/blob/2.17.2/sdk/python/kfp/dsl/pipeline_task.py). `kfp-kubernetes` 2.17.0 adds `add_pod_label`, `add_toleration`, `add_node_selector`, `mount_pvc`, `use_secret_as_env` and `set_timeout`. It has **no runtimeClassName setter**. [kfp.kubernetes `__init__`](https://github.com/kubeflow/pipelines/blob/2.17.2/kubernetes_platform/python/kfp/kubernetes/__init__.py). DRA support was merged to master on 2026-09-10, after 2.17.2, so it is not in a release yet. [kubeflow/pipelines#14243](https://github.com/kubeflow/pipelines/pull/14243)
- **Handing off to other CRs.** KFP's documented pattern is a Python component that creates a TrainJob through the Kubernetes client. `dsl.TaskConfig` / `task_config_passthroughs` forward the task's resources, tolerations and volumes into that CR. [task_config.py](https://github.com/kubeflow/pipelines/blob/2.17.2/sdk/python/kfp/dsl/task_config.py), [lightweight components guide](https://github.com/kubeflow/pipelines/blob/master/docs/user-guides/components/lightweight-python-components.md)
- **Waiting for a GPU.** KFP has no queue of its own. Pods go to the kube-scheduler. See Kueue below.
- **Licence:** Apache-2.0. [LICENSE](https://github.com/kubeflow/pipelines/blob/master/LICENSE)

### Kubeflow Trainer v2: leave out of v1

- **Version and maintenance.** v2.3.0 released 2026-08-07. v1 (Training Operator) is legacy but still gets patches (v1.9.4, 2026-08-18). [releases](https://github.com/kubeflow/trainer/releases), [legacy v1 docs](https://github.com/kubeflow/trainer/blob/v2.3.0/docs/legacy-v1/index.md)
- **Install.** Helm chart `oci://ghcr.io/kubeflow/charts/kubeflow-trainer`. It depends on JobSet 0.12.0, and on LeaderWorkerSet 0.8.0 only when the data cache is on. [v2.3.0 release notes](https://github.com/kubeflow/trainer/releases/tag/v2.3.0), [Chart.yaml](https://github.com/kubeflow/trainer/blob/v2.3.0/charts/kubeflow-trainer/Chart.yaml). Footprint 8m CPU / 143 Mi. [community-distribution README](https://github.com/kubeflow/community-distribution/blob/master/README.md#kubeflow-components-versions)
- **Purpose.** Trainer covers "distributed AI" and "multi-node, multi-GPU distributed jobs" via TrainJob and Runtimes. Its only BuiltinTrainer is TorchTune, and TorchTune is no longer actively maintained ([torchtune README](https://github.com/meta-pytorch/torchtune/blob/main/README.md)). Any other trainer must be a custom container on a runtime such as `torch_distributed` or `deepspeed_distributed`. It works with Kueue for topology-aware scheduling. [README](https://github.com/kubeflow/trainer/blob/v2.3.0/README.md), [builtin-trainer docs](https://github.com/kubeflow/trainer/tree/v2.3.0/docs/user-guides/builtin-trainer)
- **GPU requests.** `trainer.resourcesPerNode` and `numNodes` on the TrainJob. Kueue can queue TrainJobs natively. [Kueue TrainJob docs](https://github.com/kubernetes-sigs/kueue/blob/v0.20.0/site/content/en/docs/tasks/run/trainjobs.md)
- **Graph and log view.** None of its own. Pods launched by a TrainJob are not KFP tasks, so their logs don't show in the KFP log view (see the hand-off pattern above).
- **Why it doesn't earn its place yet.** On one node, the same finetuning runs as a single pod with `torchrun --nproc_per_node`. Trainer would add JobSet, a second CR lifecycle to track and cancel, and would split the logs away from KFP.
- **Licence:** Apache-2.0. [repo](https://github.com/kubeflow/trainer)

### Katib: leave out

- **Out of scope:** hyperparameter sweeps are excluded by the map (#1).
- **Maintenance.** The last release was v0.19.0 on 2025-10-30. The repo has had only 6 commits since 2026-01-01. [releases](https://github.com/kubeflow/katib/releases), [commits](https://github.com/kubeflow/katib/commits/master). Trainer merged KEP-3562 "Introduce OptimizationJob CRD" (2026-07-20), which moves on from Katib's `Experiment` CRD. [kubeflow/trainer#3565](https://github.com/kubeflow/trainer/pull/3565)
- **Footprint:** 13m / 476 Mi / 10 GB PVC, with its own database. [community-distribution README](https://github.com/kubeflow/community-distribution/blob/master/README.md#kubeflow-components-versions)
- **Licence:** Apache-2.0. [repo](https://github.com/kubeflow/katib)

### KServe: keep, in Standard mode, for `serve` only

- **Version and maintenance.** v0.21.0 released 2026-09-25, v0.20.0 on 2026-08-06. Active development, including `LLMInferenceService` and DRA `resourceClaims`. [releases](https://github.com/kserve/kserve/releases), [v0.21.0 notes](https://github.com/kserve/kserve/releases/tag/v0.21.0)
- **Install.** Standard mode uses Deployment, Service, Ingress/Gateway API and HPA. It needs Kubernetes 1.32+, cert-manager 1.15+ and a Gateway API or Ingress controller. It needs **no Knative and no Istio**. KEDA is optional, and there is no scale-from-zero. The docs call Standard mode "the recommended approach for generative inference workloads". [Kubernetes deployment install guide](https://github.com/kserve/website/blob/main/docs/admin-guide/kubernetes-deployment.md)
- **Footprint:** 600m / 1200 Mi. [community-distribution README](https://github.com/kubeflow/community-distribution/blob/master/README.md#kubeflow-components-versions)
- **GPU requests.** Normal container `resources` on the predictor. It creates a Deployment, which Kueue's `deployment` integration can gate pod by pod. [Kueue Deployment docs](https://github.com/kubernetes-sigs/kueue/blob/v0.20.0/site/content/en/docs/tasks/run/deployment.md)
- **Graph and log view.** None. KServe is reached from the `serve` Stage component. The server's own logs live on the Deployment, not in KFP.
- **Licence:** Apache-2.0. [repo](https://github.com/kserve/kserve)

### Kueue (Kubernetes SIG, not Kubeflow): add for GPU queueing

- **Version and maintenance.** v0.20.0 released 2026-09-30. [releases](https://github.com/kubernetes-sigs/kueue/releases)
- **Plain pods.** The `pod` integration is on by default since v0.16. Kueue adds a `kueue.x-k8s.io/admission` scheduling gate to labelled pods and removes it once the pod's Workload is admitted. [plain pods docs](https://github.com/kubernetes-sigs/kueue/blob/v0.20.0/site/content/en/docs/tasks/run/plain_pods.md). `manageJobsWithoutQueueName` defaults to false, so unlabelled pods (KFP drivers, system pods) are left alone. [configuration_types.go](https://github.com/kubernetes-sigs/kueue/blob/v0.20.0/apis/config/v1beta2/configuration_types.go)
- **KFP/Argo.** "Kueue doesn't support Argo Workflows ... directly, but you can take advantage of ... plain pods". The limits are that each pod is its own Workload, and that "Argo Workflows will run all previous steps and then wait for quota". [Argo Workflow docs](https://github.com/kubernetes-sigs/kueue/blob/v0.20.0/site/content/en/docs/tasks/run/external_workloads/argo_workflow.md)
- **Footprint.** The controller requests 500m / 512 Mi by default. [values.yaml](https://github.com/kubernetes-sigs/kueue/blob/v0.20.0/charts/kueue/values.yaml)
- **Licence:** Apache-2.0. [repo](https://github.com/kubernetes-sigs/kueue)

### Everything else in the Kubeflow platform: leave out

The full community distribution adds the Dashboard, Notebooks, Kubeflow Hub, Spark Operator, Istio (750m / 2.3 GiB), Knative (1450m / 1 GiB), Dex and oauth2-proxy, for a total of 4377m / 12 GiB / 65 GB. Its install guide recommends "16+ GB of RAM" and "8 CPU cores". [community-distribution README](https://github.com/kubeflow/community-distribution/blob/master/README.md). Authentication is out of scope for v1, and nothing in this set needs Istio.

### k3s GPU note (applies to every component)

k3s detects the NVIDIA container runtime and registers a `nvidia` RuntimeClass. The docs say: "If you have not changed the default runtime on your GPU nodes, you must explicitly request the NVIDIA runtime by setting `runtimeClassName: nvidia`". The NVIDIA device plugin is installed separately. [k3s advanced docs](https://docs.k3s.io/advanced#nvidia-container-runtime-support). KFP's SDK can't set `runtimeClassName`, so the GPU node's default runtime must be `nvidia`.

## Licensing table

| Component | Version checked | Licence | Public images | Flag |
|---|---|---|---|---|
| Kubeflow Pipelines | 2.17.2 | Apache-2.0 | `ghcr.io/kubeflow/kfp-*` | none |
| Argo Workflows (bundled with KFP) | v4.0.5 | Apache-2.0 ([repo](https://github.com/argoproj/argo-workflows)) | yes | none |
| ML Metadata server (bundled with KFP) | 1.14.0 | Apache-2.0 ([repo](https://github.com/google/ml-metadata)) | `gcr.io/tfx-oss-public` | In maintenance mode, per the KFP removal proposal |
| SeaweedFS (bundled with KFP) | 4.34 | Apache-2.0 ([repo](https://github.com/seaweedfs/seaweedfs)) | `ghcr.io/chrislusf/seaweedfs` | none (KFP no longer ships MinIO) |
| MySQL (bundled with KFP) | 8.4 | GPL-2.0 ([LICENSE](https://github.com/mysql/mysql-server/blob/trunk/LICENSE)) | `mysql` on Docker Hub | GPL, not AGPL. Fine to run unmodified, but it is a second database next to Postgres (#3) |
| Kubeflow Trainer | v2.3.0 | Apache-2.0 | yes | not used |
| JobSet (Trainer dependency) | 0.12.0 | Apache-2.0 ([repo](https://github.com/kubernetes-sigs/jobset)) | yes | not used |
| Katib | v0.19.0 | Apache-2.0 | yes | not used; low activity |
| KServe | v0.21.0 | Apache-2.0 | yes | none |
| Kueue | v0.20.0 | Apache-2.0 | `registry.k8s.io` | none |
| cert-manager (KServe, KFP k8s-native mode) | 1.15+ | Apache-2.0 ([repo](https://github.com/cert-manager/cert-manager)) | yes | none |

None of the selected components is AGPL, SSPL, BSL or source-only.

## Open questions for the grilling tickets

- **#3 (one object store, one Postgres):** KFP on Postgres is "not supported for production" while MLMD remains ([#14353](https://github.com/kubeflow/pipelines/issues/14353)). Options: run KFP on its bundled MySQL in v1 (two databases), accept dev-grade Postgres, or wait for MLMD removal in KFP 3.0 (milestone due 2026-11-06). SeaweedFS could also *be* the one S3 store for everyone, instead of being swapped out.
- **#6 (serving engine):** does the chosen engine fit a KServe runtime / `LLMInferenceService`, or does `serve` fall back to a plain Deployment?
- **#11 (served model lifecycle):** the `serve` Stage creates an InferenceService that outlives the Pipeline. Who deletes it, and does it keep holding GPU quota in Kueue?
- **#8 (Pipeline Request shape):** GPU count per Stage becomes `set_accelerator_limit`. Should the Request expose it, or should the API derive it?
- **#12 (Smoke Test / CI):** KFP compiles pipelines without a cluster, so CI can compile and diff the pipeline spec. Anything that needs Kueue or KServe needs a cluster.

## Sources

- KFP releases: https://github.com/kubeflow/pipelines/releases (2.17.2, 2026-09-04)
- KFP 2.17.0 release notes: https://github.com/kubeflow/pipelines/releases/tag/2.17.0
- KFP milestones: https://github.com/kubeflow/pipelines/milestones
- KFP installation docs: https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/operator-guides/installation/_index.md
- KFP MLflow plugin guide: https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/operator-guides/mlflow-plugin.md
- KFP KEP-12862 MLflow integration: https://github.com/kubeflow/pipelines/blob/master/proposals/12862-mlflow-integration/README.md
- KFP MLMD removal proposal: https://github.com/kubeflow/pipelines/blob/master/proposals/12147-mlmd-removal/README.md
- KFP Postgres/MLMD bug: https://github.com/kubeflow/pipelines/issues/14353
- KFP manifests (2.17.2): https://github.com/kubeflow/pipelines/tree/2.17.2/manifests/kustomize
- KFP Argo configmap patch: https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/third-party/argo/base/workflow-controller-configmap-patch.yaml
- KFP executor log per retry: https://github.com/kubeflow/pipelines/pull/13175
- KFP SDK PipelineTask: https://github.com/kubeflow/pipelines/blob/2.17.2/sdk/python/kfp/dsl/pipeline_task.py
- kfp-kubernetes API: https://github.com/kubeflow/pipelines/blob/2.17.2/kubernetes_platform/python/kfp/kubernetes/__init__.py
- KFP TaskConfig: https://github.com/kubeflow/pipelines/blob/2.17.2/sdk/python/kfp/dsl/task_config.py
- KFP DRA PR: https://github.com/kubeflow/pipelines/pull/14243
- Kubeflow community distribution (component table, requirements): https://github.com/kubeflow/community-distribution/blob/master/README.md
- Trainer v2.3.0 release: https://github.com/kubeflow/trainer/releases/tag/v2.3.0
- Trainer README / Chart: https://github.com/kubeflow/trainer/blob/v2.3.0/README.md, https://github.com/kubeflow/trainer/blob/v2.3.0/charts/kubeflow-trainer/Chart.yaml
- Trainer OptimizationJob KEP: https://github.com/kubeflow/trainer/pull/3565
- Katib releases: https://github.com/kubeflow/katib/releases
- KServe v0.21.0: https://github.com/kserve/kserve/releases/tag/v0.21.0
- KServe Standard mode install: https://github.com/kserve/website/blob/main/docs/admin-guide/kubernetes-deployment.md
- Kueue releases: https://github.com/kubernetes-sigs/kueue/releases (v0.20.0, 2026-09-30)
- Kueue plain pods / Argo / Deployment / TrainJob docs: https://github.com/kubernetes-sigs/kueue/tree/v0.20.0/site/content/en/docs/tasks/run
- Kueue configuration API: https://github.com/kubernetes-sigs/kueue/blob/v0.20.0/apis/config/v1beta2/configuration_types.go
- Kueue Helm values: https://github.com/kubernetes-sigs/kueue/blob/v0.20.0/charts/kueue/values.yaml
- k3s NVIDIA runtime: https://docs.k3s.io/advanced#nvidia-container-runtime-support
- MySQL licence: https://github.com/mysql/mysql-server/blob/trunk/LICENSE
