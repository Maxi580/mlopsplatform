# Storage and metadata: one object store, one Postgres?

Research ticket: [#3](https://github.com/Maxi580/mlopsplatform/issues/3) (map: [#1](https://github.com/Maxi580/mlopsplatform/issues/1)).
Facts checked on **2026-10-01** against: Kubeflow Pipelines (KFP) **2.17.2** (latest release, 2026-09-04) and `master`; kubeflow/manifests **26.03.1**; MLflow **3.16.1**; SeaweedFS **4.48**; MinIO repo state as of 2026-10-01.

## Question

Can Kubeflow Pipelines, MLflow and the platform API share **one S3-compatible object store** and **one Postgres** server? If so, which object store? Sub-questions: MinIO's 2026 status, SeaweedFS as KFP's default, alternatives, whether KFP (and ML Metadata, MLMD) can run on Postgres, how MLflow is wired to Postgres and S3, and whether one store can be partitioned per tool.

## Recommendation

**Object store: SeaweedFS (Apache-2.0), one instance, shared by everything.** It is what KFP already deploys by default since 2.15 [S3][S4], its images are published on GHCR and Docker Hub [S10][S12], it runs as a single Deployment on a k3s `local-path` PVC [S5][S23], and it offers bucket-scoped IAM users [S6][S9]. Reuse KFP's SeaweedFS instance rather than installing a second one. MinIO is ruled out (see Findings).

**SQL: one Postgres server, one database per tool, with a timing caveat for KFP.**

| Database (on the one Postgres server) | Owner | Contents |
| --- | --- | --- |
| `platform` | platform API | Pipelines, their Owner and stored Pipeline Requests, plus pointers to Runs and Kubeflow runs |
| `mlflow` | MLflow tracking server | Runs, experiments, and the model registry (the registry defaults to the same store) [S19] |
| `mlpipeline`, `cachedb` | KFP API server, KFP cache server | pipeline definitions, run history, and the execution cache [S13][S16] |

- **KFP can technically use Postgres, but it is not production-supported until KFP 3.0.** KFP's own docs say: "PostgreSQL is for development and evaluation only; it is not supported for production KFP 2.18 deployments … PostgreSQL will not be supported until KFP 3.0, when MLMD is removed" [S15]. The reason is an MLMD concurrency bug on Postgres [S17]. MLMD removal is already merged on `master` [S18], and the "KFP 3.0" milestone is due **2026-11-06** [S20].
- **So:** design for one Postgres and **target KFP 3.0**. If the platform must ship before 3.0 is released, the only allowed exception is KFP's bundled MySQL (`mysql:8.4`, GPLv2 with the FOSS exception [S14][S27]) used by KFP alone. Nothing else goes on it, and it is removed on the move to 3.0. Don't run MLMD on Postgres in production.
- Use one Postgres **16 or 17** server, e.g. managed by CloudNativePG (Apache-2.0 [S28]), not KFP's bundled `postgres:14.7-alpine3.17` [S13]. The engine version is an open question below.

**Bucket layout: one bucket per tool, not prefixes in one shared bucket.**

| Bucket | Writer | Layout |
| --- | --- | --- |
| `mlpipeline` | KFP API server, Argo, KFP launcher | Keep KFP's defaults: `pipelines/` (pipeline specs, `ObjectStoreConfig.PipelinePath`) [S8], `v2/artifacts/` (default pipeline root) [S13], `private-artifacts/<namespace>/<name>/<Y>/<m>/<d>/<pod>` (Argo logs and artifacts) [S7] |
| `mlflow` | MLflow server only, in proxied mode (`--artifacts-destination s3://mlflow`) | MLflow's own `<experiment_id>/<run_id>/artifacts/...` layout |
| `platform` | platform API and Stage Jobs | `datasets/`, `distillation-datasets/`, `models/`, `adapters/`, with the exact keys and versioning left to grilling ticket #10 |

Separate buckets give each tool its own SeaweedFS IAM identity, limited to that bucket [S6][S9], and their own lifecycle rules [S9]. They also keep KFP's `mlpipeline` name, which KFP hardcodes in several places: the `bucketName` install parameter, the SeaweedFS bootstrap and the Argo config [S5][S7][S13]. Prefixes inside one bucket would also work technically, since all three tools accept a prefix [S7][S13][S21], but with a single credential a bug in any tool could overwrite another tool's data.

## Findings

### 1. MinIO: archived, source-only, no free public images, and AGPL anyway

- `minio/minio` is **archived** on GitHub. The README opens with "THIS REPOSITORY IS NO LONGER MAINTAINED" and points to the proprietary "AIStor Free" and "AIStor Enterprise" [S1]. The README went into maintenance mode on 2025-12-03 [S2].
- On 2025-10-15 the README was changed to say "The MinIO community edition is now distributed as source code only. We will no longer provide pre-compiled binary releases", with `go install` or building the Dockerfile yourself as the only options [S1][S2]. The last GitHub release is `RELEASE.2025-10-15T17-29-55Z`, published 2025-10-16 [S2].
- **The user's report is correct, and today the situation is worse.** On 2026-10-01 the Docker Hub API returns `object not found` for `minio/minio`, the `minio` namespace no longer lists a `minio` repository, and `quay.io/minio/minio` requires authentication [S11].
- SeaweedFS's README (maintained by a competitor, so treat it as a secondary claim) says MinIO "ceased development" on 2026-04-25 [S10]. The GitHub repo's last push, on 2026-04-24, is consistent with that [S1].
- The license is **AGPL-3.0** [S1]. MinIO's own README says any use "requires validation against AGPLv3 obligations" [S1]. For unmodified internal use, AGPL's network clause is usually not triggered, but the map requires AGPL to be flagged, so it gets flagged here.
- **AIStor Free** is not an open-source option. MinIO's pricing page says it is "commercially licensed", "Redistribution is prohibited", and it is a single-node "standalone deployment" [S22]. That fails the "free for commercial use" rule, since the terms are vendor-controlled.
- **Verdict:** excluded, both for licensing (AGPL, or proprietary AIStor) and because it is distributed as source only, with no maintained free images.

### 2. SeaweedFS: KFP's default, Apache-2.0, fine on k3s

- **KFP default since 2.15.0 (2025-11-25):** "The default object store deployment has been changed to SeaweedFS, replacing the previous deployment of MinIO … MinIO remains fully supported (as is any S3-compliant object store within KFP)" [S3]. Since then KFP has also swapped its MinIO client for the AWS SDK v2, in 2.16 [S3]. kubeflow/manifests 26.03.1 ships KFP 2.17.2 with SeaweedFS as the default artifact store across all overlays [S4].
- **How KFP runs it** [S5]: a single-replica `Deployment` running `ghcr.io/chrislusf/seaweedfs:4.34` with `weed server -s3 -iam -filer`, a 20 GiB `ReadWriteOnce` PVC, and a `postStart` hook that creates the `mlpipeline` bucket and a `kubeflow-admin` IAM user from the `mlpipeline-minio-artifact` Secret. The comment in the manifest reads "Single container setup not scalable". The Service is `seaweedfs` on port 8333, and port **9000** is kept for MinIO-era configs.
- **NetworkPolicy:** KFP's manifest only admits S3 traffic from the same namespace, from `kubeflow-profile` namespaces and from `istio-system` [S5]. k3s enforces NetworkPolicies through its embedded kube-router controller [S24], so an MLflow or platform API **in another namespace will be blocked** until that policy is extended.
- **License and distribution:** Apache-2.0, actively released (4.48 on 2026-09-28), with images on GHCR and Docker Hub (`chrislusf/seaweedfs`, updated 2026-09-30) [S10][S12]. There is also a paid "SeaweedFS Enterprise" edition with extra features: data recovery, self-healing and customizable erasure coding [S10]. None of these matter for a single node.
- **S3 compatibility:** the README lists 73 bucket and object operations, including multipart uploads, presigned URLs, versioning, lifecycle, bucket policies, and IAM users, groups and policies. It says AWS SDKs "work unchanged", and the S3 compatibility suite runs in CI [S10]. Multipart matters for multi-GB model weights. KFP itself runs its end-to-end tests against SeaweedFS [S3].
- **k3s:** the only storage requirement is an RWO PVC. k3s ships Rancher's Local Path Provisioner, which stores volumes under `/var/lib/rancher/k3s/storage` [S23]. That fits both the WSL dev box and the single-node VM. SeaweedFS also offers an official Helm chart and `weed mini`, which "is fine for single-node production" [S10], if the platform later wants to run it outside KFP's manifests.

### 3. Alternatives

| Store | License | Distribution | Fit |
| --- | --- | --- | --- |
| **Garage** | AGPL-3.0 [S25] | Images published (`dxflrs/garage`, 2026-09-30) [S11] | Excluded by the AGPL rule |
| **Ceph RGW via Rook** | Ceph: LGPL-2.1/3 [S26]; Rook: Apache-2.0 [S26] | Public images | License fine, but Rook needs raw devices or unformatted partitions for OSDs [S26]. Heavy for one node with local-path storage. |
| **RustFS** | Apache-2.0 [S25] | Docker images built in CI [S25] | A MinIO reimplementation. 1.0.0 GA only on 2026-09-16, and patches still ship as `-preview` releases [S25]. Too young, and not what KFP tests against. |
| **MinIO / AIStor Free** | AGPL-3.0 / proprietary | Source only / vendor download | Excluded (see section 1) |

### 4. KFP and SQL

- **Postgres support exists in code and manifests.** In 2.17.2 the API server has a Postgres (`pgx`/gorm) driver, configured through `DBConfig.PostgreSQLConfig.{Host,Port,User,Password,DBName}` and `DBDriverName` [S8][S16]. There are overlays `env/platform-agnostic-postgresql` and `env/platform-agnostic-multi-user-postgresql` [S13]. The Postgres install config uses logical databases `mlpipeline`, `cachedb` and `metadb` against host `postgres` [S13].
- **But it is only for development and evaluation.** The KFP operator guide, last updated 2026-09-30, says Postgres is not supported for production 2.18 and will be supported from KFP 3.0, "when MLMD is removed". It also notes that 2.18 release images "will not include PostgreSQL configuration", so the dev path uses `master` images [S15]. The 2.18 release branch runs its deployment CI on MySQL only [S19b].
- **MLMD on Postgres:** in 2.17.2 the Postgres overlay runs MLMD's gRPC server with `--metadata_source_config_type=postgresql` against a **second** Postgres Deployment, `metadata-postgres-db`, using database `mlmdpostgres` [S14b]. It can be pointed at the shared server, but MLMD's Postgres backend lacks the row lock that MySQL uses for type lookups. Concurrent first runs can fail and leave duplicate type records (issue #14353, open) [S17].
- **MLMD is gone on `master`.** PR #13986, merged 2026-09-14, "replace[s] MLMD tracking with native task and artifact APIs" and removes the metadata-writer, envoy and gRPC manifests [S18]. The `master` Postgres overlay no longer includes `base/metadata` [S13]. `release-2.18` was cut before that PR and keeps MLMD [S18b]. The "KFP 3.0" milestone is due 2026-11-06 [S20].
- **In short:** with KFP ≤ 2.18 the only production-safe arrangement is MySQL for KFP, which means a second SQL server. From KFP 3.0 on, KFP, MLflow and the platform can all share one Postgres.

### 5. MLflow wiring

- **Backend store:** `--backend-store-uri postgresql://user:pass@host:5432/mlflow`. SQLAlchemy supports sqlite, postgresql, mysql and mssql. Run `mlflow db upgrade <uri>` before upgrades, because migrations can cause downtime [S21]. The model registry needs a database-backed store [S21], and the official Helm chart defaults the registry URI to the backend store URI [S19].
- **Artifacts, proxied mode (recommended):** `mlflow server --backend-store-uri postgresql://… --artifacts-destination s3://mlflow`. The server proxies uploads and downloads, so clients such as Stage Jobs need no S3 credentials [S29].
- **S3 endpoint:** set `MLFLOW_S3_ENDPOINT_URL=http://seaweedfs.kubeflow.svc:8333`, plus `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for the bucket-scoped SeaweedFS user, on the **server only**. MLflow's docs warn that setting it on both server and client produces invalid paths [S30]. Prefixes such as `s3://bucket/prefix` are supported [S30].
- **Image:** `ghcr.io/mlflow/mlflow` installs plain `mlflow` [S31]. `psycopg2-binary` comes from the `db` extra and `boto3` from the `extras` extra [S32]. Build a thin image with `mlflow[db,extras]`, or use the `Dockerfile.full` variant, which installs `mlflow[extras,azure,db,gateway,genai,auth]` [S31]. MLflow publishes an official Helm chart (`charts/`, v0.1.1, appVersion 3.16.1) [S19].
- **KFP ↔ MLflow:** KFP 2.17 ships an MLflow plugin that registers each KFP run with MLflow [S33]. KFP's own 2.18 CI keeps "all eight MLflow combinations" with MLflow on Postgres [S19b]. This is relevant to ticket #2.

### 6. Partitioning one store and one server safely

- **Object store:** SeaweedFS IAM can scope a user to a bucket or prefix, e.g. `s3.configure -user <u> -actions Read:<bucket>/<prefix>,Write:<bucket>/<prefix>` [S6], and it supports bucket policies [S10]. KFP needs `mlpipeline` (or whatever `bucketName` is set to) and puts all its keys under it [S7][S8][S13]. Argo's `keyFormat` places logs under `private-artifacts/<namespace>/...` [S7]. Separate buckets per tool are therefore safe, and so are per-tool prefixes. Buckets are preferred for isolation and lifecycle rules.
- **Postgres:** one database per tool with its own role. Each tool runs its own migrations (KFP's gorm migration [S3], `mlflow db upgrade` [S21]), so they must not share a database or schema.

## Licensing table

| Component | License | Public images | Commercial use | Flag |
| --- | --- | --- | --- | --- |
| SeaweedFS 4.48 | Apache-2.0 [S10] | Yes: GHCR and Docker Hub [S5][S12] | Yes | Paid Enterprise edition exists, not needed |
| PostgreSQL | PostgreSQL License (permissive) [S27] | Yes: official `postgres` image | Yes | none |
| CloudNativePG 1.30.1 (optional operator) | Apache-2.0 [S28] | Yes | Yes | none |
| MLflow 3.16.1 | Apache-2.0 [S34] | Yes: `ghcr.io/mlflow/mlflow` [S31] | Yes | none |
| Kubeflow Pipelines 2.17.2 | Apache-2.0 [S34] | Yes: `ghcr.io/kubeflow/*` | Yes | none |
| Argo Workflows (under KFP) | Apache-2.0 [S34] | Yes | Yes | none |
| MySQL 8.4 (KFP's interim DB) | GPLv2 + Universal FOSS Exception [S14] | Yes: `mysql:8.4` | Yes, for unmodified internal use | **Copyleft (GPLv2)**, and a second SQL server |
| MinIO (community) | AGPL-3.0 [S1] | **No**: source only, repo archived [S1][S11] | Risky | **AGPL + source-only + unmaintained** |
| MinIO AIStor Free | Proprietary [S22] | Vendor download | Restricted ("Redistribution is prohibited") | **Proprietary** |
| Garage | AGPL-3.0 [S25] | Yes | Risky | **AGPL** |
| Ceph / Rook | LGPL-2.1/3 / Apache-2.0 [S26] | Yes | Yes | Weak copyleft; operationally heavy |
| RustFS | Apache-2.0 [S25] | Yes | Yes | Immature (GA 2026-09-16) |

## Tools that can't share, and why

1. **KFP (≤ 2.18) with Postgres in production.** It is dev and evaluation only because MLMD's Postgres backend has a type-creation race [S15][S17]. KFP ≤ 2.18 therefore needs its own MySQL. This goes away with KFP 3.0 (MLMD removed [S18], milestone due 2026-11-06 [S20]).
2. **MLMD (KFP ≤ 2.18).** Even on Postgres, the stock overlay deploys a separate `metadata-postgres-db` server [S14b]. It can be repointed, but it isn't production-safe on Postgres. In 3.0 it disappears.
3. **Nothing else.** MLflow (Postgres and S3 [S21][S29]), the platform API (our own code), and KFP and Argo on SeaweedFS [S5][S7] all share the one object store. SeaweedFS's own filer metadata sits in an embedded store on its PVC and is not a separate SQL server [S5]. It could live in Postgres [S10], but there is no need.

## Open questions for the grilling tickets

- **KFP version and the MySQL exception (map / #2):** do we wait for KFP 3.0 (due 2026-11-06, not yet released), or ship on 2.17/2.18 with KFP's MySQL as a temporary second SQL server? If we ship early, is losing KFP run history acceptable when moving MySQL to Postgres, or must a migration be planned?
- **KFP's MLflow plugin (#2):** should the platform API create Runs itself, or let KFP's MLflow plugin create them [S33]? This decides who writes to the `mlflow` bucket and database.
- **Where SeaweedFS lives (install, not yet specified):** keep it inside KFP's `kubeflow` namespace and its manifests (single replica, 20 GiB PVC, `minio`/`minio123` default keys [S5]), or deploy it from the SeaweedFS Helm chart and point KFP at it? Either way the NetworkPolicy must admit MLflow and the platform API [S5][S24], and the default credentials must be replaced.
- **Capacity (Target VM, not yet specified):** models, adapters, Distillation Datasets and KFP artifacts all land on one local-path PVC. The size needs to be well above KFP's 20 GiB default. Retention or lifecycle rules per bucket would help (SeaweedFS supports lifecycle rules [S10]).
- **Model Cache vs object store (#7):** is the Model Cache a bucket in SeaweedFS, or a separate PVC? A PVC would not count as a second object store, but the ticket should decide.
- **Key layout and versioning in the `platform` bucket (#10):** e.g. `models/<pipeline-id>/…` vs content-addressed keys, and whether the MLflow model registry points at `platform` objects or MLflow copies them into `mlflow`.
- **Backups (new):** with one SeaweedFS and one Postgres holding everything, what is the backup story? Postgres dumps or CloudNativePG backups to…? A single object store can't back itself up.
- **Postgres engine version and operator (install):** CloudNativePG versus a plain StatefulSet, and Postgres 16 vs 17. KFP's bundled `postgres:14.7` is old [S13].

## Sources

- [S1] MinIO repo README (archived; "no longer maintained"; "Source-Only Distribution"; AGPL-3.0): https://github.com/minio/minio (checked 2026-10-01) · README: https://github.com/minio/minio/blob/master/README.md
- [S2] MinIO README commit history (2025-10-15 source-only, 2025-12-03 maintenance mode, 2026-02-12 state clarified) and releases: https://github.com/minio/minio/commits/master/README.md · https://github.com/minio/minio/releases
- [S3] KFP 2.15.0 release notes (SeaweedFS default; gorm migration), 2.16.0 (AWS SDK v2 replacing the MinIO client, "Remove minio"), 2.17.0 (SeaweedFS image bump): https://github.com/kubeflow/pipelines/releases/tag/2.15.0 · https://github.com/kubeflow/pipelines/releases/tag/2.16.0 · https://github.com/kubeflow/pipelines/releases/tag/2.17.0
- [S4] kubeflow/manifests README (KFP 2.17.2; SeaweedFS as default artifact store): https://github.com/kubeflow/manifests/blob/master/README.md · release 26.03.1: https://github.com/kubeflow/manifests/releases/tag/26.03.1
- [S5] KFP 2.17.2 SeaweedFS manifests (deployment, service, PVC, networkpolicy, secret): https://github.com/kubeflow/pipelines/tree/2.17.2/manifests/kustomize/third-party/seaweedfs/base/seaweedfs
- [S6] KFP SeaweedFS README (per-user `s3.configure` with bucket/prefix actions): https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/third-party/seaweedfs/README.md
- [S7] KFP Argo `workflow-controller-configmap` patch (SeaweedFS endpoint, `keyFormat` = `private-artifacts/...`): https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/third-party/argo/base/workflow-controller-configmap-patch.yaml
- [S8] KFP API server default config (`ObjectStoreConfig`, `PipelinePath: pipelines`, `PostgreSQLConfig`, MLflow plugin block): https://github.com/kubeflow/pipelines/blob/2.17.2/backend/src/apiserver/config/config.json
- [S9] SeaweedFS README, S3 section (IAM, bucket policies, lifecycle, versioning, multipart): https://github.com/seaweedfs/seaweedfs#the-most-complete-s3-api
- [S10] SeaweedFS README (Apache-2.0, Helm, `weed mini`, Enterprise edition, filer stores incl. PostgreSQL, "Compared to MinIO, RustFS") and releases (4.48, 2026-09-28): https://github.com/seaweedfs/seaweedfs/blob/master/README.md · https://github.com/seaweedfs/seaweedfs/releases
- [S11] Docker Hub API: `https://hub.docker.com/v2/namespaces/minio/repositories/minio` returns `object not found`, and the namespace listing (`/v2/namespaces/minio/repositories`) has no `minio` repo. Quay: `https://quay.io/api/v1/repository/minio/minio/tag/` returns 401 "Requires authentication". Garage image: `https://hub.docker.com/v2/namespaces/dxflrs/repositories/garage`. All checked 2026-10-01.
- [S12] SeaweedFS Docker Hub image `chrislusf/seaweedfs` (last updated 2026-09-30): https://hub.docker.com/r/chrislusf/seaweedfs
- [S13] KFP Postgres overlays and install config: 2.17.2 https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/env/platform-agnostic-postgresql/kustomization.yaml · https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/base/installs/generic/postgres/pipeline-install-config.yaml · `master` overlay (no `base/metadata`): https://github.com/kubeflow/pipelines/blob/master/manifests/kustomize/env/platform-agnostic-postgresql/kustomization.yaml · MySQL default: https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/env/platform-agnostic/kustomization.yaml · bundled Postgres image: https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/third-party/postgresql/base/pg-deployment.yaml
- [S14] KFP MySQL image `mysql:8.4`: https://github.com/kubeflow/pipelines/blob/2.17.2/manifests/kustomize/third-party/mysql/base/mysql-deployment.yaml · MySQL license (GPLv2 + FOSS exception): https://github.com/mysql/mysql-server/blob/trunk/LICENSE
- [S14b] KFP 2.17.2 MLMD Postgres overlay (separate `metadata-postgres-db`, `mlmdpostgres`): https://github.com/kubeflow/pipelines/tree/2.17.2/manifests/kustomize/base/metadata/overlays/postgres
- [S15] KFP operator guide "Database Configuration" (Postgres dev/eval only until 3.0; updated 2026-09-30): https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/operator-guides/configure-database.md · https://www.kubeflow.org/docs/components/pipelines/operator-guides/configure-database/
- [S16] KFP API server client manager (Postgres config keys, gorm Postgres driver, object store config): https://github.com/kubeflow/pipelines/blob/2.17.2/backend/src/apiserver/client_manager/client_manager.go
- [S17] kubeflow/pipelines#14353 "concurrent MLMD type creation can fail PostgreSQL pipeline runs in 2.18" (open) and the maintainer decision: https://github.com/kubeflow/pipelines/issues/14353 · https://github.com/kubeflow/pipelines/issues/14353#issuecomment-5639278081
- [S18] kubeflow/pipelines#13986 "replace MLMD tracking with native task and artifact APIs" (merged 2026-09-14): https://github.com/kubeflow/pipelines/pull/13986
- [S18b] kubeflow/pipelines#14392 (release-2.18 cut before MLMD removal): https://github.com/kubeflow/pipelines/pull/14392
- [S19] MLflow official Helm chart (`charts/`, `registryStoreUri` defaults to `backendStoreUri`, `artifactsDestination`): https://github.com/mlflow/mlflow/tree/master/charts · https://github.com/mlflow/mlflow/blob/master/charts/values.yaml
- [S19b] kubeflow/pipelines#14405 (2.18 CI restricted to MySQL; MLflow stores stay on Postgres): https://github.com/kubeflow/pipelines/pull/14405
- [S20] KFP "KFP 3.0" milestone (due 2026-11-06): https://github.com/kubeflow/pipelines/milestones
- [S21] MLflow docs, backend store: https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/
- [S22] MinIO pricing page (AIStor Free: commercially licensed, standalone, redistribution prohibited): https://www.min.io/pricing
- [S23] k3s docs, storage (Local Path Provisioner, `/var/lib/rancher/k3s/storage`): https://docs.k3s.io/add-ons/storage · source https://github.com/k3s-io/docs/blob/main/docs/add-ons/storage.md
- [S24] k3s docs, embedded network policy controller: https://docs.k3s.io/networking/networking-services
- [S25] Garage license (AGPL-3.0): https://git.deuxfleurs.fr/Deuxfleurs/garage/src/branch/main-v1/LICENSE · releases (v2.4.1): https://git.deuxfleurs.fr/Deuxfleurs/garage/releases · RustFS repo (Apache-2.0, 1.0.0 on 2026-09-16, `1.0.1-preview.*`): https://github.com/rustfs/rustfs · https://github.com/rustfs/rustfs/releases
- [S26] Ceph license (LGPL-2.1 or LGPL-3): https://github.com/ceph/ceph/blob/main/COPYING · Rook (Apache-2.0) prerequisites (raw devices/partitions): https://github.com/rook/rook/blob/master/Documentation/Getting-Started/Prerequisites/prerequisites.md
- [S27] PostgreSQL license: https://github.com/postgres/postgres/blob/master/COPYRIGHT · https://www.postgresql.org/about/licence/
- [S28] CloudNativePG (Apache-2.0, v1.30.1 on 2026-09-23): https://github.com/cloudnative-pg/cloudnative-pg
- [S29] MLflow docs, tracking server (proxied artifact access, `--artifacts-destination`, Postgres + S3 example): https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/
- [S30] MLflow docs, artifact store (S3-compatible: `MLFLOW_S3_ENDPOINT_URL`, prefix support, don't set the endpoint on both server and client): https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/
- [S31] MLflow Dockerfiles (v3.16.1): https://github.com/mlflow/mlflow/blob/v3.16.1/docker/Dockerfile · https://github.com/mlflow/mlflow/blob/v3.16.1/docker/Dockerfile.full
- [S32] MLflow `pyproject.toml` extras (`db` = PyMySQL, psycopg2-binary, pymssql; `extras` includes boto3): https://github.com/mlflow/mlflow/blob/v3.16.1/pyproject.toml
- [S33] KFP operator guide "MLflow Plugin Configuration": https://github.com/kubeflow/website/blob/master/content/en/docs/components/pipelines/operator-guides/mlflow-plugin.md
- [S34] Repo licenses via the GitHub API: https://github.com/mlflow/mlflow (Apache-2.0) · https://github.com/kubeflow/pipelines (Apache-2.0) · https://github.com/argoproj/argo-workflows (Apache-2.0)
