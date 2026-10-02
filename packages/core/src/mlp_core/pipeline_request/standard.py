from mlp_core.pipeline_request.schema import (
    Finetune,
    LoraSettings,
    PipelineRequest,
    SftPhase,
    SftSettings,
)

# A good starting request (values from #22) for users to copy and adapt.
STANDARD_PIPELINE_REQUEST = PipelineRequest(
    name="qwen-sft",
    finetune=Finetune(
        base_model="hf:Qwen/Qwen2.5-0.5B-Instruct",
        backend="hf",
        phases=[
            SftPhase(
                algorithm="sft",
                dataset="dataset:my-chat",
                method="lora",
                settings=SftSettings(
                    learning_rate=1e-4,
                    num_train_epochs=3,
                    per_device_train_batch_size=8,
                    gradient_accumulation_steps=1,
                    max_length=1024,
                ),
                lora=LoraSettings(
                    r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear"
                ),
            )
        ],
    ),
)
