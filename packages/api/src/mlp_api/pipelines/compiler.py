import json
import os
from dataclasses import dataclass, field
from itertools import pairwise

from google.protobuf import json_format
from kfp import dsl, kubernetes

from mlp_core import config
from mlp_core.pipeline_request.references import checkpoint_prefix, split_endpoint_reference
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_core.settings import Settings


@dataclass(frozen=True)
class StepEnvironment:
    """The images and storage of the steps, from the API's environment; tests make their own."""

    stages_image: str
    # Training backend -> its trainer image.
    trainer_images: dict[str, str]
    model_cache_pvc: str
    object_store_url: str
    object_store_bucket: str
    # How the `serve` step reaches the API inside the cluster.
    api_url: str
    sandbox_url: str
    # Where `evaluate` reaches an Endpoint's Service.
    platform_namespace: str

    @classmethod
    def from_environment(cls) -> "StepEnvironment":
        return cls(
            stages_image=os.environ["STAGES_IMAGE"],
            trainer_images={
                "hf": os.environ["TRAINER_HF_IMAGE"],
                "unsloth": os.environ["TRAINER_UNSLOTH_IMAGE"],
            },
            model_cache_pvc=os.environ["MODEL_CACHE_PVC"],
            object_store_url=os.environ["S3_ENDPOINT_URL"],
            object_store_bucket=os.environ["S3_BUCKET"],
            api_url=os.environ["API_URL"],
            sandbox_url=os.environ["SANDBOX_URL"],
            platform_namespace=os.environ["PLATFORM_NAMESPACE"],
        )


@dataclass(frozen=True)
class Resume:
    """What a resumed Pipeline reuses from the Pipelines it continues; stored with it."""

    # The Model Versions its first Phases registered, in Phase order; those Phases are skipped.
    model_versions: list[str] = field(default_factory=list)
    # The Dataset Version its `distill` Stage registered; `distill` is then skipped.
    distilled_dataset: str | None = None
    # Where the first unfinished Phase's Checkpoint lies, which that Phase continues from.
    checkpoint: str | None = None
    # The best parameters its `sweep` Stage found; `sweep` is then skipped.
    swept_parameters: dict | None = None


def compile_pipeline(
    pipeline_id: int,
    request: PipelineRequest,
    fetched: list[str],
    steps: StepEnvironment,
    settings: Settings,
    resume: Resume | None = None,
) -> dict:
    """The Kubeflow pipeline spec for a resolved request, in the form KFP's run API takes."""
    resume = resume or Resume()

    @dsl.container_component
    def cleanup():
        return dsl.ContainerSpec(image=steps.stages_image, command=["mlp-stage", "cleanup"])

    @dsl.pipeline(name=request.name)
    def pipeline():
        cleanup_task = cleanup().set_caching_options(False)
        with dsl.ExitHandler(cleanup_task):
            # 1. fetch fills the Model Cache with the Base Models and benchmarks the Stages need.
            previous = None
            if fetched:
                previous = fetch_step(fetched, steps, settings)
                # Only fetch gets the token, from the Secret; a pipeline parameter would be logged.
                kubernetes.use_secret_as_env(
                    previous,
                    config.PIPELINE_SECRET_NAME.format(id=pipeline_id),
                    {"hf_token": config.SECRET_ENV_VARS["hf_token"]},
                    optional=True,
                )

            # 2. The Stages, one after another in their fixed order; `@distill` is distill's output,
            # `@sweep` sweep's, `@finetune` the last Phase's or, once every Phase finished, the
            # last one reused.
            stages, distilled, quantized = [], resume.distilled_dataset or "", ""
            swept = json.dumps(resume.swept_parameters) if resume.swept_parameters else ""
            finetuned = (resume.model_versions or [""])[-1]
            gpus = settings.gpus_per_stage
            if request.distill and not distilled:
                stages.append(distill_step(pipeline_id, request, steps, gpus))
                distilled = stages[-1].outputs["dataset"]
            if request.sweep and not swept:
                stages.append(sweep_step(pipeline_id, request, steps, settings, distilled))
                swept = stages[-1].outputs["best_parameters"]
            if request.finetune:
                phases = finetune_steps(
                    pipeline_id, request, steps, settings, distilled, resume, swept=swept
                )
                stages += phases
                if phases:
                    finetuned = phases[-1].outputs["model_version"]
            if request.quantize:
                stages.append(quantize_step(pipeline_id, request, finetuned, steps, gpus))
                quantized = stages[-1].outputs["model_version"]
            if request.speculate:
                stages.append(
                    speculate_step(
                        pipeline_id, request, steps, settings, finetuned, quantized, distilled
                    )
                )
            if request.evaluate:
                evaluated = evaluated_request(request, resume)
                stages.append(evaluate_step(pipeline_id, evaluated, steps, gpus))
            if request.serve:
                stages.append(serve_step(pipeline_id, steps))
            for stage in stages:
                if previous:
                    stage.after(previous)
                previous = stage

    return pipeline_run_spec(pipeline)


def evaluated_request(request: PipelineRequest, resume: Resume) -> PipelineRequest:
    """The request with `@finetune` as the last reused Model Version if every Phase finished."""
    finished = request.finetune and len(resume.model_versions) == len(request.finetune.phases)
    if not finished or request.evaluate.model != config.FINETUNE_OUTPUT:
        return request
    evaluate = request.evaluate.model_copy(update={"model": resume.model_versions[-1]})
    return request.model_copy(update={"evaluate": evaluate})


def compile_smoke_test(
    pipeline_id: int,
    name: str,
    fetched: list[str],
    sandbox: bool,
    cases: dict[str, PipelineRequest],
    steps: StepEnvironment,
    settings: Settings,
) -> dict:
    """The Smoke Test's Kubeflow pipeline spec: one node per case, named after it, in a chain."""

    @dsl.pipeline(name=name)
    def pipeline():
        previous = fetch_step(fetched, steps, settings).set_display_name("fetch")
        # Each later case runs once the one before it ended, even if that failed; no data passes
        # between them.
        if sandbox:
            previous = (
                sandbox_step(steps, settings)
                .set_display_name(config.SMOKE_TEST_SANDBOX_CASE)
                .after(previous)
                .ignore_upstream_failure()
            )
        gpus = settings.gpus_per_stage
        for case, request in cases.items():
            # A case of several steps (distill, Phases, serve) passes or fails with its last one,
            # and each of its steps runs only once the one before it passed.
            case_steps, distilled, swept = [], "", ""
            if request.distill:
                case_steps.append(("distill", distill_step(pipeline_id, request, steps, gpus)))
                distilled = case_steps[-1][1].outputs["dataset"]
            if request.sweep:
                case_steps.append(("sweep", sweep_step(pipeline_id, request, steps, settings)))
                swept = case_steps[-1][1].outputs["best_parameters"]
            if request.finetune and case == config.SMOKE_TEST_RESUME_CASE:
                # The first step stops after a Checkpoint, as a cancel would; the next continues it.
                [interrupted] = finetune_steps(
                    pipeline_id, request, steps, settings, distilled, interrupt=True
                )
                resume = Resume(checkpoint=checkpoint_prefix(pipeline_id, 0))
                [resumed] = finetune_steps(pipeline_id, request, steps, settings, distilled, resume)
                case_steps += [("interrupted", interrupted), ("finetune", resumed)]
            elif request.finetune:
                names = [f"finetune-{phase.algorithm}" for phase in request.finetune.phases]
                tasks = finetune_steps(
                    pipeline_id, request, steps, settings, distilled, swept=swept
                )
                case_steps += zip(names, tasks, strict=True)
            if request.quantize:
                case_steps.append(
                    ("quantize", quantize_step(pipeline_id, request, "", steps, gpus))
                )
            if request.speculate:
                speculated = speculate_step(pipeline_id, request, steps, settings)
                case_steps.append(("speculate", speculated))
            if request.evaluate:
                case_steps.append(("evaluate", evaluate_step(pipeline_id, request, steps, gpus)))
            if request.serve:
                case_steps.append(("serve", serve_step(pipeline_id, steps)))
            for step_name, task in case_steps[:-1]:
                task.set_display_name(f"{case}-{step_name}")
            case_steps[0][1].after(previous).ignore_upstream_failure()
            for (_, earlier), (_, task) in pairwise(case_steps):
                task.after(earlier)
            previous = case_steps[-1][1].set_display_name(case)

    return pipeline_run_spec(pipeline)


def fetch_step(references: list[str], steps: StepEnvironment, settings: Settings):
    @dsl.container_component
    def fetch(model_cache_size: str, references: str):
        return dsl.ContainerSpec(
            image=steps.stages_image,
            command=["mlp-stage", "fetch"],
            args=[model_cache_size, references],
        )

    task = fetch(model_cache_size=settings.model_cache_size, references=",".join(references))
    use_model_cache(task, steps)
    # A benchmark's download scores one sample, which for a coding benchmark runs in the Sandbox.
    task.set_env_variable("SANDBOX_URL", steps.sandbox_url)
    return task


def distill_step(
    pipeline_id: int, request: PipelineRequest, steps: StepEnvironment, gpus_per_stage: int
):
    """The step that asks the Teacher about every prompt and registers the replies via the API."""

    @dsl.container_component
    def distill(
        pipeline_id: str, request: str, teacher_url: str, gpus: str, dataset: dsl.OutputPath(str)
    ):
        return dsl.ContainerSpec(
            image=steps.stages_image,
            command=["mlp-stage", "distill"],
            args=[pipeline_id, request, teacher_url, gpus, dataset],
        )

    # 1. Only a Base Model or Model Version Teacher runs here, on GPUs; the others have a URL.
    api_url = request.distill.api_url
    teacher_url = api_url or endpoint_reference_url(request.distill.teacher, steps)
    gpus = 0 if teacher_url else gpus_per_stage

    # 2. The step, with the step token to register the replies and an API Teacher's key.
    task = distill(
        pipeline_id=str(pipeline_id),
        request=request.model_dump_json(),
        teacher_url=teacher_url,
        gpus=str(gpus),
    )
    secrets = {"step_token": config.SECRET_ENV_VARS["step_token"]}
    if api_url:
        secrets["teacher_api_key"] = config.SECRET_ENV_VARS["teacher_api_key"]
    kubernetes.use_secret_as_env(task, config.PIPELINE_SECRET_NAME.format(id=pipeline_id), secrets)
    task.set_env_variable("API_URL", steps.api_url)
    use_vllm(task, steps, gpus)
    return task


def sweep_step(
    pipeline_id: int,
    request: PipelineRequest,
    steps: StepEnvironment,
    settings: Settings,
    distilled_dataset="",
):
    """The step that trains the Sweep's Trials one after another and reports the best parameters."""
    backend = request.sweep.backend

    @dsl.container_component
    def sweep(
        pipeline_id: str,
        request: str,
        distilled_dataset: str,
        best_parameters: dsl.OutputPath(str),
    ):
        return dsl.ContainerSpec(
            image=steps.trainer_images[backend],
            command=["mlp-stage", "sweep"],
            args=[pipeline_id, request, distilled_dataset, best_parameters],
        )

    task = sweep(
        pipeline_id=str(pipeline_id),
        request=request.model_dump_json(),
        distilled_dataset=distilled_dataset,
    )
    use_trainer(task, steps, trainer_gpus(backend, settings))
    # It reports the best parameters to the API with the step token.
    task.set_env_variable("API_URL", steps.api_url)
    kubernetes.use_secret_as_env(
        task,
        config.PIPELINE_SECRET_NAME.format(id=pipeline_id),
        {"step_token": config.SECRET_ENV_VARS["step_token"]},
    )
    return task


def finetune_steps(
    pipeline_id: int,
    request: PipelineRequest,
    steps: StepEnvironment,
    settings: Settings,
    distilled_dataset="",
    resume: Resume | None = None,
    interrupt: bool = False,
    swept="",
) -> list:
    """One step per unfinished Phase, handed the previous Model Version and `swept` if it asks."""
    backend = request.finetune.backend
    trainer_image = steps.trainer_images[backend]

    @dsl.container_component
    def finetune(
        pipeline_id: str,
        phase_index: str,
        request: str,
        distilled_dataset: str,
        previous_model_version: str,
        swept_parameters: str,
        checkpoint_minutes: str,
        resume_checkpoint: str,
        stop_after_checkpoint: str,
        model_version: dsl.OutputPath(str),
    ):
        return dsl.ContainerSpec(
            image=trainer_image,
            command=["mlp-stage", "finetune"],
            args=[
                pipeline_id,
                phase_index,
                request,
                distilled_dataset,
                previous_model_version,
                swept_parameters,
                checkpoint_minutes,
                resume_checkpoint,
                stop_after_checkpoint,
                model_version,
            ],
        )

    # A resume skips the Phases that registered a Model Version and continues the next one.
    resume = resume or Resume()
    finished = len(resume.model_versions)
    tasks, previous_model_version = [], (resume.model_versions or [""])[-1]
    for index, phase in list(enumerate(request.finetune.phases))[finished:]:
        # The request holds no Secret value, so it can be a parameter.
        task = finetune(
            pipeline_id=str(pipeline_id),
            phase_index=str(index),
            request=request.model_dump_json(),
            distilled_dataset=distilled_dataset,
            previous_model_version=previous_model_version,
            swept_parameters=swept if phase.params_from else "",
            # The Smoke Test's interrupted step saves at its first step.
            checkpoint_minutes="0" if interrupt else str(settings.checkpoint_minutes),
            resume_checkpoint=(index == finished and resume.checkpoint) or "",
            stop_after_checkpoint="true" if interrupt else "",
        ).set_display_name(f"finetune-{phase.algorithm}")
        use_trainer(task, steps, trainer_gpus(backend, settings))
        tasks.append(task)
        previous_model_version = task.outputs["model_version"]
    return tasks


def trainer_gpus(backend: str, settings: Settings) -> int:
    return 1 if backend in config.SINGLE_GPU_BACKENDS else settings.gpus_per_stage


# A step that trains on its GPUs, offline, reading Model Versions and Datasets from the object
# store; `grpo` and `rloo` score their completions with their rewards in the Sandbox.
def use_trainer(task, steps: StepEnvironment, gpus: int) -> None:
    use_model_cache(task, steps)
    task.set_env_variable("HF_HUB_OFFLINE", "1")
    task.set_accelerator_type(config.GPU_RESOURCE)
    task.set_accelerator_limit(gpus)
    use_object_store(task, steps)
    task.set_env_variable("SANDBOX_URL", steps.sandbox_url)


def quantize_step(
    pipeline_id: int, request: PipelineRequest, finetuned, steps: StepEnvironment, gpus: int
):
    """The step that quantizes the model and registers it; `finetuned` is `@finetune`'s output."""

    @dsl.container_component
    def quantize(
        pipeline_id: str, request: str, finetuned: str, model_version: dsl.OutputPath(str)
    ):
        return dsl.ContainerSpec(
            image=steps.stages_image,
            command=["mlp-stage", "quantize"],
            args=[pipeline_id, request, finetuned, model_version],
        )

    task = quantize(
        pipeline_id=str(pipeline_id), request=request.model_dump_json(), finetuned=finetuned
    )
    use_vllm(task, steps, gpus)
    return task


def speculate_step(
    pipeline_id: int,
    request: PipelineRequest,
    steps: StepEnvironment,
    settings: Settings,
    finetuned="",
    quantized="",
    distilled="",
):
    """The step that trains a Speculator and registers it, handed the earlier Stages' outputs."""

    @dsl.container_component
    def speculate(
        pipeline_id: str,
        request: str,
        finetuned: str,
        quantized: str,
        distilled_dataset: str,
        gpus: str,
        dataloader_workers: str,
        model_version: dsl.OutputPath(str),
    ):
        return dsl.ContainerSpec(
            image=steps.stages_image,
            command=["mlp-stage", "speculate"],
            args=[
                pipeline_id,
                request,
                finetuned,
                quantized,
                distilled_dataset,
                gpus,
                dataloader_workers,
                model_version,
            ],
        )

    task = speculate(
        pipeline_id=str(pipeline_id),
        request=request.model_dump_json(),
        finetuned=finetuned,
        quantized=quantized,
        distilled_dataset=distilled,
        gpus=str(settings.gpus_per_stage),
        dataloader_workers=str(settings.speculate_dataloader_workers),
    )
    use_vllm(task, steps, settings.gpus_per_stage)
    return task


def evaluate_step(
    pipeline_id: int, request: PipelineRequest, steps: StepEnvironment, gpus_per_stage: int
):
    """The step that runs the request's benchmarks, offline, against its model."""

    @dsl.container_component
    def evaluate(pipeline_id: str, request: str, endpoint_url: str, gpus: str):
        return dsl.ContainerSpec(
            image=steps.stages_image,
            command=["mlp-stage", "evaluate"],
            args=[pipeline_id, request, endpoint_url, gpus],
        )

    # 1. An Endpoint already serves its model; any other gets a vLLM of its own, on GPUs.
    endpoint_url = endpoint_reference_url(request.evaluate.model, steps)
    gpus = 0 if endpoint_url else gpus_per_stage

    # 2. The step.
    task = evaluate(
        pipeline_id=str(pipeline_id),
        request=request.model_dump_json(),
        endpoint_url=endpoint_url,
        gpus=str(gpus),
    )
    use_vllm(task, steps, gpus)
    task.set_env_variable("SANDBOX_URL", steps.sandbox_url)
    return task


def endpoint_reference_url(model: str, steps: StepEnvironment) -> str:
    """Where steps reach the `endpoint:` Reference's Service; empty for any other model."""
    if not model.startswith("endpoint:"):
        return ""
    return endpoint_service_url(split_endpoint_reference(model), steps.platform_namespace)


def endpoint_service_url(name: str, namespace: str) -> str:
    """Where the API and Pipeline steps reach the Endpoint's vLLM inside the cluster."""
    object_name = config.ENDPOINT_OBJECT_NAME.format(name=name)
    return config.ENDPOINT_SERVICE_URL.format(
        object_name=object_name, namespace=namespace, port=config.VLLM_PORT
    )


# A step that loads models, e.g. on vLLM: it reads the Model Cache offline and Model Versions from
# the object store, on its GPUs if it has any.
def use_vllm(task, steps: StepEnvironment, gpus: int) -> None:
    use_model_cache(task, steps)
    task.set_env_variable("HF_HUB_OFFLINE", "1")
    task.set_env_variable("HF_DATASETS_OFFLINE", "1")
    if gpus:
        task.set_accelerator_type(config.GPU_RESOURCE)
        task.set_accelerator_limit(gpus)
    use_object_store(task, steps)


def serve_step(pipeline_id: int, steps: StepEnvironment):
    @dsl.container_component
    def serve(pipeline_id: str):
        return dsl.ContainerSpec(
            image=steps.stages_image, command=["mlp-stage", "serve"], args=[pipeline_id]
        )

    # The API starts the Endpoint from the request it stored; the step only asks, with the serve
    # token from the Secret.
    task = serve(pipeline_id=str(pipeline_id))
    task.set_caching_options(False)
    task.set_env_variable("API_URL", steps.api_url)
    kubernetes.use_secret_as_env(
        task,
        config.PIPELINE_SECRET_NAME.format(id=pipeline_id),
        {"step_token": config.SECRET_ENV_VARS["step_token"]},
    )
    return task


# Runs snippets in the Sandbox, which only Pipeline steps can reach.
def sandbox_step(steps: StepEnvironment, settings: Settings):
    @dsl.container_component
    def sandbox(sandbox_memory_mb: str):
        return dsl.ContainerSpec(
            image=steps.stages_image,
            command=["mlp-stage", "check-sandbox"],
            args=[sandbox_memory_mb],
        )

    task = sandbox(sandbox_memory_mb=str(settings.sandbox_memory_mb))
    task.set_caching_options(False)
    task.set_env_variable("SANDBOX_URL", steps.sandbox_url)
    return task


def pipeline_run_spec(pipeline) -> dict:
    return {
        "pipeline_spec": json_format.MessageToDict(pipeline.pipeline_spec),
        "platform_spec": json_format.MessageToDict(pipeline.platform_spec),
    }


def use_model_cache(task, steps: StepEnvironment) -> None:
    task.set_caching_options(False)
    task.set_env_variable("HF_HOME", config.MODEL_CACHE_PATH)
    kubernetes.mount_pvc(task, steps.model_cache_pvc, config.MODEL_CACHE_PATH)


# Reads Dataset Versions with the object store keys KFP's own steps use.
def use_object_store(task, steps: StepEnvironment) -> None:
    task.set_env_variable("S3_ENDPOINT_URL", steps.object_store_url)
    task.set_env_variable("S3_BUCKET", steps.object_store_bucket)
    kubernetes.use_secret_as_env(
        task, config.OBJECT_STORE_SECRET, config.OBJECT_STORE_SECRET_ENV_VARS
    )
