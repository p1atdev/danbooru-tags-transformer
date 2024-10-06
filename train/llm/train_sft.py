import sys

sys.path.append(".")

import torch
import torch.nn as nn

from datasets import Dataset, load_from_disk, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from accelerate import Accelerator
from trl import DataCollatorForCompletionOnlyLM

from src.tags import InstructionTokens

import wandb

SEED = 20241006

# pretrained model
BASE_MODEL_NAME = "p1atdev/dart-v3-llama-8L-241005"

DATASET_NAME = "p1atdev/dart-v3-20241006-sft"

PROJECT_NAME = "danbooru-tags-transformer-v3"
PUSH_HUB_NAME = "p1atdev/dart-v3-llama-8L-241005_241006-sft"
SAVE_DIR = "./output/dart-llama-8L-241005_241006-sft"

NUM_PROC = 4

INPUT_END = InstructionTokens.INPUT_END  # for sft


def prepare_models():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, padding="max_length", truncation=True, max_length=256
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )

    return tokenizer, model


def prepare_dataset():
    ds = load_dataset(DATASET_NAME)

    return ds


def main():
    set_seed(SEED)

    tokenizer, model = prepare_models()

    dataset = prepare_dataset()

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
        response_template=INPUT_END,  # <|input_end|>
    )

    accelerator = Accelerator()
    model.to(accelerator.device)

    wandb.init(project=PROJECT_NAME)
    train_args = TrainingArguments(
        output_dir=SAVE_DIR,
        overwrite_output_dir=True,
        num_train_epochs=4,
        # auto_find_batch_size=True,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=1000,
        weight_decay=0.01,
        optim="ademamix",
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 1},
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=10,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=accelerator.num_processes,
        neftune_noise_alpha=5,
        torch_compile=True,
        bf16=True,
        report_to=["wandb"],
        hub_model_id=PUSH_HUB_NAME,
        hub_private_repo=True,
        push_to_hub=True,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,  # type: ignore
        tokenizer=tokenizer,
        args=train_args,
        # dataset_text_field="text",
        # dataset_num_proc=NUM_PROC,
        train_dataset=dataset["train"],  # type: ignore
        eval_dataset=dataset["test"],  # type: ignore
        data_collator=data_collator,
        # max_seq_length=256,
    )

    trainer.train(
        # resume_from_checkpoint=True,
    )

    trainer.push_to_hub()


if __name__ == "__main__":
    main()

    # debug
    _tokenizer, model = prepare_models()
    print(model)

    model.push_to_hub(PUSH_HUB_NAME, private=True)
