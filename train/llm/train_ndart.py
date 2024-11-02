import sys

sys.path.append(".")

import torch
import torch.nn as nn
import numpy as np

from datasets import Dataset, load_from_disk, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForTextEncoding,
    set_seed,
)
from accelerate import Accelerator

from src.tags import InstructionTokens, MultiModalTokens
from models.ndart.processing_ndart import NDartProcessor
from models.ndart.modeling_ndart import NDartForConditionalGeneration
from models.ndart.configuration_ndart import NDartConfig

import wandb

SEED = 20241006

# pretrained model
BASE_ENCODER_MODEL_NAME = "intfloat/multilingual-e5-base"
BASE_DECODER_MODEL_NAME = "p1atdev/dart-v3-llama-8L6QKV-241029_241102-sft-1"

DATASET_NAME = "p1atdev/dart-v3-20241102-ndart-1"

PROJECT_NAME = "danbooru-tags-transformer-v3-natural"
PUSH_HUB_NAME = "p1atdev/dart-v3-llama-8L6QKV-241029_241102_241102-NL-1"
SAVE_DIR = "./output/dart-v3-llama-8L6QKV-241029_241102_241102-NL-1"

NUM_PROC = 4

INPUT_END = InstructionTokens.INPUT_END  # for sft
NATURAL_PLACEHOLDER = MultiModalTokens.NATURAL_PLACEHOLDER
IGNORE_INDEX = -100

TORCH_DTYPE = torch.bfloat16

TRAIN_ENCODER = False
TRAIN_PROJECTOR = True
TRAIN_DECODER = True


def prepare_models():
    # register custom code
    NDartConfig.register_for_auto_class()
    NDartForConditionalGeneration.register_for_auto_class("AutoModelForPreTraining")
    NDartProcessor.register_for_auto_class()

    processor = NDartProcessor(
        encoder_tokenizer=AutoTokenizer.from_pretrained(BASE_ENCODER_MODEL_NAME),
        decoder_tokenizer=AutoTokenizer.from_pretrained(BASE_DECODER_MODEL_NAME),
        natural_token=NATURAL_PLACEHOLDER,
    )
    encoder = AutoModelForTextEncoding.from_pretrained(
        BASE_ENCODER_MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
    )
    decoder = AutoModelForCausalLM.from_pretrained(
        BASE_DECODER_MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
    )
    model = NDartForConditionalGeneration._from_config(
        NDartConfig(
            encoder_config=encoder.config,
            decoder_config=decoder.config,
            natural_token_index=processor.natural_token_id,
        ),
        torch_dtype=TORCH_DTYPE,
    )
    model.encoder_model = encoder
    model.decoder_model = decoder
    model.to(TORCH_DTYPE)

    ###! Encoder training config
    model.encoder_model.requires_grad_(TRAIN_ENCODER)
    model.encoder_model.eval()
    model.encoder_model = torch.compile(model.encoder_model)
    ###

    ###! Projector training config
    model.projection.requires_grad_(TRAIN_PROJECTOR)
    model.projection.eval()
    model.projection = torch.compile(model.projection)
    ###

    ###! Decoder training config
    model.decoder_model.requires_grad_(TRAIN_DECODER)
    model.decoder_model.train()
    # model.decoder_model = torch.compile(model.decoder_model)
    ###

    return processor, model


def prepare_dataset():
    ds = load_dataset(DATASET_NAME)

    return ds


def main():
    set_seed(SEED)

    processor, model = prepare_models()

    dataset = prepare_dataset()

    pad_token_id = processor.decoder_tokenizer.pad_token_id
    input_end_id = processor.decoder_tokenizer.convert_tokens_to_ids(INPUT_END)

    def collate_fn(examples):
        batch = {
            "input_ids": [torch.tensor(example["input_ids"]) for example in examples],
            "attention_mask": [
                torch.tensor(example["attention_mask"]) for example in examples
            ],
            "encoder_input_ids": [
                torch.tensor(example["encoder_input_ids"]) for example in examples
            ],
            "encoder_attention_mask": [
                torch.tensor(example["encoder_attention_mask"]) for example in examples
            ],
        }
        # pad
        batch = {
            "input_ids": nn.utils.rnn.pad_sequence(
                batch["input_ids"],
                batch_first=True,
                padding_value=processor.decoder_tokenizer.pad_token_id,  # type: ignore
            ),
            "attention_mask": nn.utils.rnn.pad_sequence(
                batch["attention_mask"], batch_first=True, padding_value=0
            ),
            "encoder_input_ids": nn.utils.rnn.pad_sequence(
                batch["encoder_input_ids"],
                batch_first=True,
                padding_value=processor.encoder_tokenizer.pad_token_id,  # type: ignore
            ),
            "encoder_attention_mask": nn.utils.rnn.pad_sequence(
                batch["encoder_attention_mask"], batch_first=True, padding_value=0
            ),
        }

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == pad_token_id] = IGNORE_INDEX
        batch["labels"] = labels

        for i in range(len(examples)):
            response_token_ids_start_idx = None

            for idx in np.where(batch["labels"][i] == input_end_id)[0]:
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if input_end_id == batch["labels"][i][idx : idx + 1].tolist():
                    response_token_ids_start_idx = idx

            if response_token_ids_start_idx is None:
                batch["labels"][i, :] = IGNORE_INDEX
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + 1

                # Make pytorch loss function ignore all tokens up through the end of the response key
                batch["labels"][i, :response_token_ids_end_idx] = IGNORE_INDEX

        return batch

    accelerator = Accelerator()
    model.to(accelerator.device)

    # wandb.init(project=PROJECT_NAME)
    train_args = TrainingArguments(
        output_dir=SAVE_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        # auto_find_batch_size=True,
        # per_device_train_batch_size=32,
        # per_device_eval_batch_size=32,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.01,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        adam_epsilon=1e-8,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={
            "min_lr": 5e-5,
            "num_cycles": 0.5,
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
        neftune_noise_alpha=5,
        torch_compile=False,  # compile does not work
        bf16=True,
        report_to=[],
        hub_model_id=PUSH_HUB_NAME,
        hub_private_repo=True,
        push_to_hub=True,
        save_safetensors=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,  # type: ignore
        tokenizer=processor,
        args=train_args,
        # dataset_num_proc=NUM_PROC,
        train_dataset=dataset["train"],  # type: ignore
        eval_dataset=dataset["test"],  # type: ignore
        data_collator=collate_fn,
    )

    trainer.train(
        # resume_from_checkpoint=True,
    )

    trainer.push_to_hub()


if __name__ == "__main__":
    main()

    # # debug
    # _tokenizer, model = prepare_models()
    # print(model)

    # model.push_to_hub(PUSH_HUB_NAME, private=True)
