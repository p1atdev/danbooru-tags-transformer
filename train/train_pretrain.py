import torch
import torch.nn as nn

from datasets import Dataset, load_from_disk, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    set_seed,
    DataCollatorForLanguageModeling,
)
from accelerate import Accelerator

from preset import get_model_config

# import wandb

SEED = 20240425

EMBEDDING_MODEL = "p1atdev/dart-v2-vectors"

TOKENIZER_NAME = "p1atdev/dart-v2-tokenizer"
DATASET_NAME = "p1atdev/dart-v2-20240424-pretrain"
MODEL_TYPE = "llama"
MODEL_SIZE = "100m"

PROJECT_NAME = "danbooru-tags-transformer-v2"
PUSH_HUB_NAME = "p1atdev/dart-v2-100m-llama"
SAVE_DIR = "./dart-100m-llama"


def prepare_models():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, padding="max_length", truncation=True, max_length=256
    )

    config, MODEL_CLASS = get_model_config(MODEL_TYPE, MODEL_SIZE)
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    model: PreTrainedModel = MODEL_CLASS._from_config(config)
    model.to(torch.bfloat16)

    vector_model = AutoModel.from_pretrained(
        EMBEDDING_MODEL, torch_dtype=torch.bfloat16
    )
    vector_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    vector_embeddings = vector_model.get_input_embeddings()
    vector_vocab = vector_tokenizer.get_vocab()

    # vector モデルの embedding と新しいモデルではトークンIDが異なるので、トークン名を参照してコピーする
    assert config.hidden_size == vector_model.config.hidden_size
    input_embeddings = nn.Embedding(
        tokenizer.vocab_size, config.hidden_size, padding_idx=tokenizer.pad_token_id
    )
    input_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
    with torch.no_grad():
        for token, token_id in tokenizer.get_vocab().items():
            if token in vector_vocab:
                input_embeddings.weight[token_id] = vector_embeddings.weight[
                    vector_vocab[token]
                ].to(torch.bfloat16)

    # use the vector model's position embeddings
    model.set_input_embeddings(input_embeddings)

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

    # wandb.init(project=PROJECT_NAME)
    train_args = TrainingArguments(
        output_dir=SAVE_DIR,
        overwrite_output_dir=True,
        num_train_epochs=5,
        # auto_find_batch_size=True,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        warmup_steps=1000,
        weight_decay=0.0,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
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
        report_to=[],
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

    # debug
    # tokenizer, model = prepare_models()
    # print(model)
    # print(model.get_input_embeddings())

    # model.push_to_hub(PUSH_HUB_NAME, private=True)
    # tokenizer.push_to_hub(PUSH_HUB_NAME, private=True)
