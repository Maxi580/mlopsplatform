import gc
import importlib
import json
import logging
import os
import tempfile
from pathlib import Path

import mlflow
import optuna
from datasets import Dataset

from mlp_core import api_paths, config
from mlp_core.pipeline_request.schema import PipelineRequest, Sweep, SweepParameters
from mlp_stages.dataset_versions import download_dataset_version
from mlp_stages.model_versions import model_files, model_with_adapter
from mlp_stages.platform_api import call_step_route


def sweep(pipeline_id: str, request: str, distilled_dataset: str, best_parameters: str) -> None:
    """Writes the best Trial's parameters to `best_parameters`, once reported to the API."""
    # 1. The `sweep` block of the resolved request; `@distill` is the Dataset Version KFP hands
    # over.
    sweep = PipelineRequest.model_validate_json(request).sweep
    if sweep.dataset == config.DISTILL_OUTPUT:
        sweep.dataset = distilled_dataset
    # Out of the environment, so the Trials' Runs start under the step's Run instead of resuming it.
    run_id = os.environ.pop("MLFLOW_RUN_ID")

    # 2. The backend, imported by name before any training library, as finetune does.
    backend = importlib.import_module(f"mlp_stages.finetune.backends.{sweep.backend}")

    with tempfile.TemporaryDirectory() as directory:
        # 3. The model every Trial starts from, the Teacher a `distillation` Sweep loads too, and
        # the rows Trials train on and are measured on.
        scratch = Path(directory)
        model_directory = model_files(sweep.model, scratch / "model")
        teacher = sweep.teacher and list(model_with_adapter(sweep.teacher, scratch / "teacher")[1:])
        tokenizer = backend.load_tokenizer(model_directory)
        train_rows, eval_rows = trial_datasets(sweep, scratch)

        # 4. The Trials, each a Run nested under the step's; a failed one is logged and skipped.
        def trial_objective(trial: optuna.Trial) -> float:
            # Only now is a failed Trial's model unreferenced, its traceback handled by Optuna.
            free_gpu_memory()
            parameters = suggested_parameters(trial, sweep.parameters)
            trial.set_user_attr("parameters", parameters)
            phase = sweep.trial_phase(parameters)
            experiment_id = mlflow.MlflowClient().get_run(run_id).info.experiment_id
            run = mlflow.start_run(
                parent_run_id=run_id, experiment_id=experiment_id, run_name=f"trial-{trial.number}"
            )
            # Its weights and Checkpoints stay in its output directory, deleted after it.
            with run, tempfile.TemporaryDirectory() as output_directory:
                mlflow.log_params(trial.params)
                trainer = backend.build_trainer(
                    model_directory,
                    phase,
                    tokenizer,
                    train_rows,
                    output_directory,
                    None,
                    teacher,
                    eval_dataset=eval_rows,
                )
                value = trained_objective(trainer, sweep.objective.metric)
                mlflow.log_metric(sweep.objective.metric, value)
                return value

        logging.getLogger("optuna").setLevel(logging.INFO)
        study = optuna.create_study(direction=sweep.objective.goal, sampler=sampler(sweep))
        study.optimize(trial_objective, n_trials=sweep.trials, catch=(Exception,))

    # 5. The best Trial's parameters and objective in the step's Run, then to the API and KFP,
    # which hands them to Phases with `params_from: @sweep`.
    completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if not completed:
        raise SystemExit(f"Every Trial failed; see the Runs of the {sweep.trials} Trials")
    best = study.best_trial
    client = mlflow.MlflowClient()
    for name, value in best.params.items():
        client.log_param(run_id, f"best/{name}", value)
    client.log_metric(run_id, f"best/{sweep.objective.metric}", best.value)
    print(f"Best of {len(completed)} of {sweep.trials} Trials: {best.params} -> {best.value}")
    parameters = best.user_attrs["parameters"]
    call_step_route(
        api_paths.SWEEP_PIPELINE,
        pipeline_id,
        json={"parameters": parameters, "objective": best.value},
    )
    Path(best_parameters).parent.mkdir(parents=True, exist_ok=True)
    Path(best_parameters).write_text(json.dumps(parameters))


def trial_datasets(sweep: Sweep, scratch: Path) -> tuple[Dataset, Dataset]:
    """The rows Trials train on, and those they are measured on: `eval_dataset` or held out."""
    rows = read_rows(sweep.dataset, scratch / "train")
    if sweep.eval_dataset:
        return rows, read_rows(sweep.eval_dataset, scratch / "eval")
    split = rows.train_test_split(test_size=sweep.eval_split, seed=config.SWEEP_SEED)
    return split["train"], split["test"]


def read_rows(reference: str, directory: Path) -> Dataset:
    directory.mkdir(parents=True)
    path = download_dataset_version(reference, directory)
    return Dataset.from_json(str(path), keep_in_memory=True)


def sampler(sweep: Sweep) -> optuna.samplers.BaseSampler:
    if sweep.sampler == "grid":
        search_space = {
            f"{block}.{name}": parameter.values
            for block, parameters in sweep.parameters
            for name, parameter in parameters.items()
        }
        return optuna.samplers.GridSampler(search_space, seed=config.SWEEP_SEED)
    if sweep.sampler == "random":
        return optuna.samplers.RandomSampler(seed=config.SWEEP_SEED)
    return optuna.samplers.TPESampler(seed=config.SWEEP_SEED)


def suggested_parameters(trial: optuna.Trial, searched: SweepParameters) -> dict:
    """The Trial's values, `{settings: …, lora: …}`, each from its parameter's range or values."""
    suggested = {}
    for block, parameters in searched:
        suggested[block] = {}
        for name, parameter in parameters.items():
            key, log = f"{block}.{name}", parameter.scale == "log"
            if parameter.values is not None:
                value = trial.suggest_categorical(key, parameter.values)
            elif isinstance(parameter.min, int) and isinstance(parameter.max, int):
                value = trial.suggest_int(key, parameter.min, parameter.max, log=log)
            else:
                value = trial.suggest_float(key, parameter.min, parameter.max, log=log)
            suggested[block][name] = value
    return suggested


def trained_objective(trainer, metric: str) -> float:
    """The metric's last value once the Trial trained and, for an `eval_` metric, was measured."""
    trainer.train()
    if metric.startswith("eval_"):
        trainer.evaluate()
    values = [entry[metric] for entry in trainer.state.log_history if metric in entry]
    if not values:
        raise ValueError(f"the Trial logged no {metric}")
    return values[-1]


# The next Trial loads its model on the same GPUs.
def free_gpu_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
