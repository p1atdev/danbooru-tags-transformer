import torch

from datasets import Dataset, load_from_disk, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    set_seed,
    OPTConfig,
    OPTForCausalLM,
    DataCollatorForLanguageModeling,
)
from accelerate import Accelerator

import wandb

SEED = 20240419

TOKENIZER_NAME = "p1atdev/dart-tokenizer-v2-encode"
DATASET_NAME = "p1atdev/202403-at20240423-tokenized-shuffle"
CONFIG_PATH = "./config/opt/125m.json"

PROJECT_NAME = "dart2vec_opt_1"
PUSH_HUB_NAME = "p1atdev/dart2vec-opt_8"
SAVE_DIR = "./dart2vec_opt_8"


def prepare_models():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, padding="max_length", truncation=True, max_length=256
    )

    config = OPTConfig.from_json_file(CONFIG_PATH)
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    model = OPTForCausalLM._from_config(config)
    model.to(torch.bfloat16)

    # freeze the position embeddings
    model.model.decoder.embed_positions.requires_grad_(False)
    # init the weights with zeros
    model.model.decoder.embed_positions.weight.data.zero_()

    return tokenizer, model


def prepare_dataset():
    ds = load_dataset(DATASET_NAME)

    return ds


def main():
    set_seed(SEED)

    tokenizer, model = prepare_models()

    dataset = prepare_dataset()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    accelerator = Accelerator()
    model.to(accelerator.device)

    wandb.init(project=PROJECT_NAME)
    train_args = TrainingArguments(
        output_dir=SAVE_DIR,
        overwrite_output_dir=True,
        num_train_epochs=5,
        # auto_find_batch_size=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=8e-4,
        warmup_steps=100,
        weight_decay=0.0,
        optim="adamw_torch_fused",
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs={
            "patience": 1,
            "threshold": 1e-4,
        },
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=10,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=accelerator.num_processes,
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
        tokenizer=tokenizer,  # can't upload tokenizer because it has custom decoder
        args=train_args,
        train_dataset=dataset["train"],  # type: ignore
        eval_dataset=dataset["test"],  # type: ignore
        data_collator=data_collator,
    )

    trainer.train(
        # resume_from_checkpoint=True,
    )

    trainer.push_to_hub()


if __name__ == "__main__":
    main()
