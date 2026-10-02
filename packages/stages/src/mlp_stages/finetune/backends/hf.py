import trl
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlp_core import config
from mlp_core.pipeline_request.references import split_base_model_reference
from mlp_core.pipeline_request.schema import Phase


def load_tokenizer(base_model: str):
    """The Base Model's tokenizer and chat template, from the Model Cache."""
    repo, commit = split_base_model_reference(base_model)
    return AutoTokenizer.from_pretrained(repo, revision=commit)


def build_trainer(base_model: str, phase: Phase, tokenizer, dataset, output_directory: str):
    """The catalog row's TRL trainer, training a fresh LoRA Adapter on the Base Model with PEFT."""
    algorithm = config.ALGORITHMS[phase.algorithm]
    repo, commit = split_base_model_reference(base_model)
    # The weights fetch put in the Model Cache, at their saved precision (usually bfloat16).
    model = AutoModelForCausalLM.from_pretrained(repo, revision=commit, dtype="auto")
    return getattr(trl, algorithm["trainer"])(
        model=model,
        args=getattr(trl, algorithm["config"])(
            **{**algorithm["defaults"], **phase.settings.model_dump()},
            output_dir=output_directory,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        # TRL freezes the Base Model's weights and trains only this Adapter.
        peft_config=LoraConfig(**{**config.LORA_DEFAULTS, **phase.lora.model_dump()}),
    )
