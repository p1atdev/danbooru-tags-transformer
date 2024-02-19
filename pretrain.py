import torch

from transformers import (
    TrainingArguments,
    AutoTokenizer,
    set_seed,
    DataCollatorForLanguageModeling,
    Trainer,
    GenerationConfig,
)
from datasets import Dataset, load_from_disk, load_dataset
from dart.configuration_dart import DartConfig
from dart.modeling_dart import DartForCausalLM

import wandb


CACHE_DIR = "/huggingface/cache"

DATASET_NAME = "p1atdev/dart-tokenized-pretrain-20240219"

# BASE_MODEL_NAME = "FacebookAI/roberta-base"
TOKENIZER_NAME = "p1atdev/dart-tokenizer-v1"

# MAX_LENGTH = 256

SEED = 202340214

PROJECT_NAME = "danbooru-tag-transformer"
OUTPUT_DIR = "./dart-test-3"
HUB_MODEL_ID = "p1atdev/dart-test-3"


def get_models():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    config = DartConfig(vocab_size=tokenizer.vocab_size)
    model = DartForCausalLM(config)
    model.to(torch.bfloat16)

    generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=128,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=20,
        num_beams=1,
    )
    model.generation_config = generation_config

    return tokenizer, model


def main():
    set_seed(SEED)

    dataset = load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)
    assert isinstance(dataset, Dataset)
    dataset = dataset.train_test_split(test_size=2000, shuffle=True, seed=SEED)

    print(dataset)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    tokenizer, model = get_models()
    model.config.use_cache = False  # type: ignore
    # model.resize_token_embeddings(len(tokenizer))  # type: ignore

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    wandb.init(project=PROJECT_NAME)
    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        # auto_find_batch_size=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=500,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=10,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_num_workers=4,
        # torch_compile=True,
        report_to=["wandb"],
        hub_model_id=HUB_MODEL_ID,
        hub_private_repo=True,
        push_to_hub=True,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,  # type: ignore
        # tokenizer=tokenizer, # can't upload tokenizer because it has custom decoder
        args=train_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=test_dataset,  # type: ignore
        data_collator=collator,
    )

    trainer.train(
        # resume_from_checkpoint=True,
    )

    trainer.save_model(OUTPUT_DIR)
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
