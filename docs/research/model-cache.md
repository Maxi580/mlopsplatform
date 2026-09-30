# Model Cache: storage, sharing and eviction

Research for [#7](https://github.com/Maxi580/mlopsplatform/issues/7). Facts were checked on 2026-10-01 against `huggingface_hub` v2.0.0 (released 2026-09-24), `datasets` 5.0.1, vLLM v0.30.0, Kubernetes docs (v1.37), k3s docs (`main`), local-path-provisioner v0.0.37 and Longhorn 1.13.0. Numbered references point to **Sources**.

## Question

How should the **Model Cache** work, so each Base Model and dataset is downloaded from Hugging Face once and the cache never fills the disk? This covers where it lives (shared PVC, object store or a hybrid), what happens when two Jobs want the same model at once, how eviction works, and how to download quickly with an HF token, including gated models. Every tool must be free for commercial use.

## Recommendation

**Where the cache lives:** in one PersistentVolumeClaim that holds the standard Hugging Face cache layout (`HF_HOME`, and so `$HF_HOME/hub` and `$HF_HOME/datasets`). It sits on a **dedicated disk or partition** on the GPU node, not on the node's root filesystem. A full root filesystem triggers kubelet's `nodefs.available<10%` eviction of every Pod on the node [23].
- On the single-node targets (dev k3s and the production VM), use k3s's bundled local-path provisioner [9][10]. A `ReadWriteOnce` claim is enough, because every Pod runs on the one node and RWO "can allow multiple pods to access (read from or write to) that volume when the pods are running on the same node" [8]. Point local-path's storage path, or a static PV, at the dedicated mount.
- Keep the cache off the object store. Base Models in the Model Cache are a re-downloadable cache, not a record. The object store (likely SeaweedFS, Apache-2.0 [22]) keeps what cannot be re-downloaded: finetuned weights, Distillation Datasets, KFP and MLflow artifacts. [#10](https://github.com/Maxi580/mlopsplatform/issues/10) decides the details.
- If the platform ever goes multi-node, change only the StorageClass to an RWX one, such as Longhorn RWX (NFSv4 share-manager, Apache-2.0) [11]. The cache layout and the code stay the same.

**How Jobs share it:**
1. The first step of every Pipeline is a platform-owned **fetch** step, one KFP component. It resolves the requested revision to a commit hash with `resolve_revision()`, then runs `snapshot_download()` into the Model Cache [1]. It is the only Job that gets the HF token.
2. Every later Stage (`distill`, `finetune`, `evaluate`, `serve`) mounts the same PVC, for example with `kfp.kubernetes.mount_pvc` [16]. It runs with `HF_HUB_OFFLINE=1` and the pinned commit hash, so it makes no network calls and needs no token [2]. vLLM, Transformers and `datasets` all resolve through the same `HF_HUB_CACHE` [2][14].
3. Record the resolved commit hash on the Run in MLflow, so the Pipeline is reproducible even if `main` moves.

**How eviction works:** the platform builds a small **cache janitor** on top of `huggingface_hub`'s own cache API (`scan_cache_dir()`, `delete_revisions()`, `hf cache prune`) [1]. Nothing off the shelf does the policy part.
- **Admission:** before downloading, the fetch step sums the repo's file sizes from the Hub metadata and compares that with free space minus a reserve. If it doesn't fit, it evicts. If it still doesn't fit, it fails the Pipeline with a clear error, which matches the map's "show the error to the user" rule. Don't rely on `huggingface_hub` for this: its disk check only logs a warning [4].
- **Pinning (leases):** a revision is protected while any non-terminal Pipeline references it, while it belongs to the Smoke Test, or while an operator has pinned it. The API already knows which Pipelines are live, so it keeps these leases in Postgres.
- **LRU:** after each fetch, and nightly as a CronJob, evict unpinned revisions, least recently used first, until usage is under a high-water mark such as 80% of the volume. "Last used" means the platform's own last-lease timestamp. Filesystem atime is only a fallback, because it is coarse under `relatime` [19].
- **Garbage:** `hf cache prune` removes detached revisions, leftover `.incomplete` files and unreferenced shared Xet blobs [1].

## Options compared

| | **A. Shared PVC with the HF cache** (recommended) | **B. Object store only, pulled or streamed per Job** | **C. Hybrid: object store as source of truth, PVC as local cache** |
|---|---|---|---|
| How it works | One PVC mounted at `HF_HOME` in every Stage Job. The HF libraries use it natively [1][2]. | Mirror each HF repo into S3 once. `serve` streams safetensors from `s3://` with `--load-format runai_streamer` [12]. `finetune` and `evaluate` must download to local disk first. | Mirror HF into S3, then hydrate a node-local PVC from S3 on a cache miss. |
| Downloads from HF | Once per revision | Once per revision, but every Job still copies from S3 to the node (except the streamed `serve` path) | Once from HF, plus one S3 to PVC copy per node |
| Library support | Native in Transformers, `datasets`, vLLM, TRL: set env vars and you're done | Only vLLM streams from S3 directly [12]. vLLM still copies config and tokenizer files into a local temp directory [13]. Training libraries need our own download code. | We write both the mirroring and the hydration logic |
| Eviction | We build the policy on `hf cache` [1] | S3 lifecycle rules, plus scratch space per Job | Both |
| Disk safety | Needs a dedicated volume and our admission check, because local-path ignores PVC capacity [9] | Disk use per Job is temporary, but the object store grows | Both |
| Fit for single node | Best: zero network hops, fastest loads | Adds object-store traffic on the same box for nothing | Complexity pays off only on a multi-node cluster |
| Multi-node later | Swap the StorageClass for Longhorn RWX or NFS [11] | Works as is | Designed for it |

**On k3s storage backends:**
- **local-path** (bundled with k3s, Apache-2.0): it stores data under `/var/lib/rancher/k3s/storage` by default, which you can change with `--default-local-storage-path` [10]. It binds the Pod to the node [10]. It supports only RWO, unless you use `sharedFileSystemPath` [9]. "No support for the volume capacity limit currently" [9]: the PVC size is not enforced, hence the dedicated disk and our own admission check.
- **Longhorn RWX:** it exposes a Longhorn volume through an NFSv4 server in a share-manager Pod, and every node needs an NFSv4 client [11]. That adds overhead on a single node. Keep it for multi-node.
- **NFS:** it works, but `flock` may be a silent no-op on some NFS mounts. `huggingface_hub` copes with that (see below) [3].
- **KServe LocalModelCache** (Apache-2.0) caches models on node disks with a DaemonSet agent. It needs KServe, and it supports only `InferenceService`, not training Jobs [15]. It could complement the `serve` Stage if [#6](https://github.com/Maxi580/mlopsplatform/issues/6) picks KServe, but it can't be the Model Cache.

## Concurrency behaviour

All of this applies to `huggingface_hub` v2.0.0 [3][5]:

- **Two Jobs downloading the same file:** each download takes a lock at `$HF_HUB_CACHE/.locks/<repo>/<etag>.lock` through `WeakFileLock`, which is `filelock`'s `FileLock` and falls back to `SoftFileLock` if `flock` is unsupported [3][5]. The second Job waits, logs "Still waiting to acquire lock" every 10 s, then finds the blob present and just creates its snapshot symlink [3][5].
- **Correctness does not depend on the lock:** "the lock is best-effort … each download writes to a process-unique temporary file that is atomically renamed into place". On filesystems where `flock` silently succeeds (Lustre, GPFS, some NFS), "a broken lock costs only duplicated bandwidth" [3]. So a shared PVC cannot be corrupted by concurrent downloads.
- **Lock files are group-writable** (`0o664`), so Pods running as different UIDs in the same group can wait on each other's locks [5]. Give all Stage Pods the same `fsGroup`.
- **Different files of the same repo** download in parallel (`snapshot_download` defaults to `max_workers=8`) [7].
- **Deletion is not coordinated with readers.** `hf cache rm` and `delete_revisions()` take no locks. The strategy just deletes snapshot folders and unreferenced blobs [1][6]. A running Job whose model is evicted keeps its already-open file handles on Linux, but any file it opens later is gone. This is why eviction must honour **leases**, and why the fetch step runs first so a Pipeline cannot start against a half-evicted model.
- **Offline Stages cannot race against the Hub.** With the commit hash pinned and `HF_HUB_OFFLINE=1`, later Stages never re-resolve `main`. `snapshot_download` raises `IncompleteSnapshotError` instead of returning a partial folder if files are missing [1].

## Eviction tooling

**What exists:**

| Tool | What it does | Gap for us |
|---|---|---|
| `hf cache ls` with `--filter "size>1GB"`, `--filter "accessed>30d"`, `--sort accessed`, `--format json`, `-q` [1] | Lists repos and revisions with size, last accessed and last modified | Read-only |
| `hf cache rm <repo or rev> -y` and `hf cache rm $(hf cache ls --filter "accessed>30d" -q) -y` [1] | Deletes repos, revisions or single files, and blobs only when unreferenced | No quota, no LRU-to-target, no pinning, no lock |
| `hf cache prune` [1] | Deletes detached revisions, `.incomplete` files and unused shared Xet blobs | Doesn't touch revisions that are still referenced |
| `scan_cache_dir()`, `HFCacheInfo.delete_revisions()`, `DeleteCacheStrategy.expected_freed_size` [1][6] | The Python API behind the CLI, with a dry-run size estimate | Same gaps |
| `hf cache verify` [1] | Checks cached files against Hub checksums | — |
| `Dataset.cleanup_cache_files()` [14] | Cleans the processed Arrow cache in `HF_DATASETS_CACHE` | `hf cache` doesn't manage this cache, so the janitor must cover `$HF_HOME/datasets` too |
| Xet `chunk_cache` | Disabled by default since `hf_xet` 1.2.0. If enabled, it uses random eviction and is capped by `HF_XET_CHUNK_CACHE_SIZE_BYTES` [1][2] | Leave it disabled |

`huggingface-cli delete-cache` is the older interactive command. `hf cache rm` and `hf cache prune` replace it in current releases [1].

Notes on the signals:
- `last_accessed` comes from blob `st_atime` [6]. Under the Linux default `relatime`, atime is updated only if it is older than mtime or ctime, or more than a day old [19]. That's fine for day-grained LRU, but not for "is someone using this right now". Leases answer that question.
- Shared Xet blobs are deduplicated across repos. Per-repo sizes are logical, and only the cache-wide `size_on_disk` is physical [1]. So the janitor should re-scan after each deletion and not add up per-repo sizes.

**What we'd build** (roughly 150 lines of Python in the API's codebase, reusing `huggingface_hub`):
1. A `model_cache_leases` table in Postgres: `(repo_type, repo_id, commit_hash, pipeline_id or "smoke-test" or "pinned", created_at, released_at)`.
2. An admission function for the fetch step: estimate the size, check free space, call evict, then download or fail.
3. `evict(target_bytes)`: `scan_cache_dir()`, drop leased revisions, sort by last lease time (atime as fallback), `delete_revisions(...).execute()` until under target, then prune.
4. A nightly CronJob calling the same function, plus a datasets Arrow cache sweep.
5. Metrics: cache size, free space and evictions, exported to MLflow or logs.

## Download speed and HF token handling

- **Xet replaces `hf_transfer`.** `HF_HUB_ENABLE_HF_TRANSFER` is deprecated: "all file transfers go through the `hf-xet` binary package … `hf_transfer` can't be used anymore" [2]. `hf-xet` is used automatically when installed [2], and `hf-xet` / xet-core is Apache-2.0 [21].
- **Tuning:** `HF_XET_HIGH_PERFORMANCE=1` saturates bandwidth and uses all CPU cores, and is "analogous to the legacy `HF_HUB_ENABLE_HF_TRANSFER=1`". `HF_XET_NUM_CONCURRENT_RANGE_GETS` (default 16) sets parallelism per file. Set `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY` only on HDDs [2]. Set the high-performance flag on the fetch step only.
- **Cross-repo dedup:** Xet files are stored once in a cache-wide shared blob store and symlinked into each repo. That saves disk and downloads when two repos share weights [1][2]. It needs symlinks, which local-path on ext4 or xfs supports. Don't set `HF_HUB_DISABLE_SYMLINKS` [2].
- **Why a token speeds things up:** Hub rate limits apply per 5-minute window on "Resolvers" (file downloads). Anonymous use gets 3,000 per IP and a free user 5,000 (September 2025 figures, "subject to change"). HF's first advice when throttled is "make sure you always pass a `HF_TOKEN`" [17]. `huggingface_hub` ≥ 1.2.0 automatically waits out 429s using the `RateLimit` header [17].
- **Gated models:** access must be requested in the browser by a user account. Downloads then need that user's token [18]. HF recommends **fine-grained tokens** for production, scoped to the needed repos [20].
- **Token handling in this design:** only the fetch step receives the token, as an env var from a Kubernetes Secret created for that Pipeline (for example `kfp.kubernetes.use_secret_as_env` [16]), and the Secret is deleted afterwards. All other Stages run with `HF_HUB_OFFLINE=1` and no token. Never set `HF_TOKEN_PATH` or `HF_HOME/token` on the shared volume, because the token would then be stored in the cache [2]. [#9](https://github.com/Maxi580/mlopsplatform/issues/9) owns the Secret lifecycle.

## Open questions for the grilling tickets

- **[#9](https://github.com/Maxi580/mlopsplatform/issues/9), gated access leaks through the cache.** Once one user's token has downloaded a gated model, any later Pipeline can read it offline from the shared cache without a token or its own license acceptance. Is that acceptable while there is a single Owner? When multi-user arrives, does the fetch step re-check access with the requesting user's token (a cheap metadata call) even on a cache hit?
- **[#9](https://github.com/Maxi580/mlopsplatform/issues/9), cache hits without a token.** For an ungated, already-cached model, should the fetch step still call the Hub to resolve `main`, or accept a pinned commit hash and skip the network entirely?
- **[#10](https://github.com/Maxi580/mlopsplatform/issues/10), boundary between Model Cache and object store.** Are Distillation Datasets and finetuned outputs ever read back through the Model Cache, or only from the object store? Must a Base Model revision used by a registered model be kept, or only its commit hash recorded?
- **[#11](https://github.com/Maxi580/mlopsplatform/issues/11), serve leases.** A long-running `serve` Deployment holds a lease on its Base Model, and on the adapter's base too. Should the lease end when the Pipeline finishes or when the Deployment is deleted? The answer decides whether a served model can ever be evicted.
- **[#8](https://github.com/Maxi580/mlopsplatform/issues/8), Pipeline Request fields.** Is `revision` optional, defaulting to `main` and then pinned? Can the user pass `allow_patterns` so a Pipeline doesn't pull every GGUF or `.bin` duplicate in a repo?
- **[#12](https://github.com/Maxi580/mlopsplatform/issues/12), Smoke Test.** Its tiny model should be permanently pinned. Does CI (no cluster) need to test the janitor logic against a fake cache directory?
- **Not yet covered by any ticket: cache sizing and operator controls.** Who sets the cache volume size and high-water mark on the target VM, and is there an API or CLI command to pin, list or evict cache entries by hand?

## Sources

1. Hugging Face, *Understand caching* (huggingface_hub v2.0.0): https://huggingface.co/docs/huggingface_hub/guides/manage-cache
2. Hugging Face, *Environment variables* (huggingface_hub v2.0.0): https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
3. `huggingface_hub` source, `file_download.py` (locking, `_download_to_tmp_and_move`), main at v2.0.0: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py
4. Same file, `_check_disk_space` (logs a warning only): https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py
5. `huggingface_hub` source, `utils/_fixes.py` (`WeakFileLock`): https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_fixes.py
6. `huggingface_hub` source, `utils/_cache_manager.py` (`scan_cache_dir`, atime-based `last_accessed`, delete strategy): https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_cache_manager.py
7. `huggingface_hub` source, `_snapshot_download.py` (`max_workers=8`): https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_snapshot_download.py
8. Kubernetes docs, *Persistent Volumes: Access Modes*: https://kubernetes.io/docs/concepts/storage/persistent-volumes/#access-modes
9. rancher/local-path-provisioner README (v0.0.37, Apache-2.0): https://github.com/rancher/local-path-provisioner
10. k3s docs, *Volumes and Storage*: https://docs.k3s.io/add-ons/storage
11. Longhorn docs, *ReadWriteMany (RWX) Volume* (v1.13.0): https://longhorn.io/docs/latest/nodes-and-volumes/volumes/rwx-volumes/
12. vLLM docs, *Loading models with Run:ai Model Streamer* (vLLM v0.30.0): https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer/ (streamer is Apache-2.0, v0.16.1: https://github.com/run-ai/runai-model-streamer)
13. vLLM source, `vllm/transformers_utils/runai_utils.py` (`ObjectStorageModel` mirrors non-weight files to a local directory): https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/runai_utils.py
14. Hugging Face `datasets` docs, *Cache management* (datasets 5.0.1): https://huggingface.co/docs/datasets/cache
15. KServe docs, *LocalModel Installation* (v0.20): https://github.com/kserve/website/blob/main/docs/install/localmodel-install.md
16. Kubeflow Pipelines `kfp-kubernetes`, `volume.py` (`mount_pvc`) and `secret.py` (`use_secret_as_env`): https://github.com/kubeflow/pipelines/tree/master/kubernetes_platform/python/kfp/kubernetes
17. Hugging Face Hub docs, *Rate limits*: https://huggingface.co/docs/hub/rate-limits
18. Hugging Face Hub docs, *Gated models*: https://huggingface.co/docs/hub/models-gated
19. Linux `mount(8)` man page, `relatime`: https://man7.org/linux/man-pages/man8/mount.8.html
20. Hugging Face Hub docs, *User access tokens*: https://huggingface.co/docs/hub/security-tokens
21. huggingface/xet-core (Apache-2.0): https://github.com/huggingface/xet-core
22. seaweedfs/seaweedfs (Apache-2.0): https://github.com/seaweedfs/seaweedfs
23. Kubernetes docs, *Node-pressure Eviction* (default hard threshold `nodefs.available<10%`, the reason to keep the cache off the root filesystem): https://kubernetes.io/docs/concepts/scheduling-eviction/node-pressure-eviction/
