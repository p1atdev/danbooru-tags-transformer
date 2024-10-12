from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import (
    AutoModelForTextEncoding,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    BertModel,
    PreTrainedTokenizerBase,
    AutoTokenizer,
    ProcessorMixin,
    GenerationMixin,
    BatchFeature,
    PreTrainedTokenizerFast,
)
from transformers.processing_utils import ProcessingKwargs


class NDartProcessorKwargs:
    _defaults = {
        "natural_kwargs": {
            "padding": True,
        },
        "tag_kwargs": {
            "padding": True,
        },
    }


@dataclass
class NDartProcessingOutput:
    natural: BatchFeature
    tag: BatchFeature


class NDartProcessor(ProcessorMixin):
    attributes = ["natural_tokenizer", "tag_tokenizer"]
    valid_kwargs = ["chat_template", "natural_token"]
    natural_tokenizer_class = "AutoTokenizer"
    tag_tokenizer_class = "AutoTokenizer"

    natural_tokenizer: PreTrainedTokenizerBase
    tag_tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        natural_tokenizer=None,
        tag_tokenizer=None,
        natural_token="<|natural|>",
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            natural_tokenizer, tag_tokenizer, chat_template=chat_template, **kwargs
        )
        self.natural_token_id = self.tag_tokenizer.convert_tokens_to_ids(natural_token)

    def __call__(
        self,
        natural_text: str | list[str] | None = None,
        tag_text: str | list[str] | None = None,
        **kwargs,
    ) -> NDartProcessingOutput:
        if tag_text is None:
            raise ValueError("tag_text is required for NDartProcessor")

        def _validate_text_input(text) -> str | list[str]:
            if isinstance(text, list):
                assert all(
                    isinstance(t, str) for t in text
                ), f"Expected list of str but got {type(text)}"
                assert all(len(t) > 0 for t in text), "Expected non-empty strings"
            else:
                assert isinstance(text, str), f"Expected str but got {type(text)}"
                assert len(text) > 0, "Expected non-empty string"
            return text

        def _normalize_text_input(text: str | list[str]) -> list[str]:
            if isinstance(text, str):
                return [text]
            return text

        natural_text: str | list[str] = _validate_text_input(natural_text)
        natural_text = _normalize_text_input(natural_text)
        tag_text: str | list[str] = _validate_text_input(tag_text)
        tag_text = _normalize_text_input(tag_text)

        natural_output_kwargs = {
            **NDartProcessorKwargs._defaults["natural_kwargs"],
            "return_tensors": "pt",
            **kwargs,
        }
        tag_output_kwargs = {
            **NDartProcessorKwargs._defaults["tag_kwargs"],
            "return_tensors": "pt",
            **kwargs,
        }

        natural_tokens = self.natural_tokenizer(
            natural_text,
            **natural_output_kwargs,
        )
        tag_tokens = self.tag_tokenizer(
            tag_text,
            **tag_output_kwargs,
        )

        tag_input_ids, tag_attention_mask = self.insert_encoder_tokens_batch(
            encoder_input_ids=natural_tokens["input_ids"],
            encoder_attention_mask=natural_tokens["attention_mask"],
            decoder_input_ids=tag_tokens["input_ids"],
            decoder_attention_mask=tag_tokens["attention_mask"],
        )

        return NDartProcessingOutput(
            natural=BatchFeature(data={**natural_tokens}),
            tag=BatchFeature(
                data={
                    "input_ids": tag_input_ids,
                    "attention_mask": tag_attention_mask,
                }
            ),
        )

    def insert_encoder_tokens_batch(
        self,
        encoder_input_ids: torch.LongTensor,  # (batch_size, seq_len)
        decoder_input_ids: torch.LongTensor,  # (batch_size, seq_len)
        encoder_attention_mask: torch.LongTensor | None = None,  # (batch_size, seq_len)
        decoder_attention_mask: torch.LongTensor | None = None,  # (batch_size, seq_len)
    ):
        new_input_ids = []
        new_attention_mask = []

        encoder_token_lens = (
            encoder_attention_mask.sum(dim=1)
            if encoder_attention_mask is not None
            else encoder_input_ids.size(1)
        )
        # decoder_input_ids内のnatural_token_idの位置を取得
        positions = (decoder_input_ids == self.natural_token_id).nonzero(
            as_tuple=False
        )[:, 1]  # (batch_index, in_batch_position) -> (in_batch_position)

        for i, (input_ids_i, position, encoder_len) in enumerate(
            zip(
                decoder_input_ids,
                positions,
                encoder_token_lens,
                strict=True,
            )
        ):
            # 置き換え用のトークンを準備 (1 x encoder_len の長さの self.natural_token_id のテンソル)
            replacement_tokens = torch.full(
                (encoder_len,),
                self.natural_token_id,
                dtype=torch.long,
            )

            # 新しく作成するinput_idsを作成
            new_input_ids_i = torch.cat(
                [
                    input_ids_i[:position] if position > 0 else torch.tensor([]),
                    replacement_tokens,
                    input_ids_i[position + 1 :]
                    if position + 1 < len(input_ids_i)
                    else torch.tensor([]),
                ]
            )
            new_input_ids.append(new_input_ids_i)

            if decoder_attention_mask is not None:
                attention_mask_i = decoder_attention_mask[i]
                new_attention_mask_i = torch.cat(
                    [
                        attention_mask_i[:position]
                        if position > 0
                        else torch.tensor([]),
                        attention_mask_i,
                        attention_mask_i[position + 1 :]
                        if position + 1 < len(attention_mask_i)
                        else torch.tensor([]),
                    ]
                )
                new_attention_mask.append(new_attention_mask_i)

        # padding right
        new_input_ids = nn.utils.rnn.pad_sequence(
            new_input_ids,
            batch_first=True,
            padding_value=self.tag_tokenizer.pad_token_id,
        )
        new_attention_mask = (
            nn.utils.rnn.pad_sequence(
                new_attention_mask,
                batch_first=True,
                padding_value=0,
            )
            if new_attention_mask
            else None
        )
        return new_input_ids, new_attention_mask

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->PreTrained
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tag_tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->PreTrained
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tag_tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["natural_text", "tag_text"]


if __name__ == "__main__":
    natural_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    tag_tokenizer = AutoTokenizer.from_pretrained("p1atdev/dart-v3-tokenizer-241010")
    processor = NDartProcessor(
        natural_tokenizer=natural_tokenizer,
        tag_tokenizer=tag_tokenizer,
        natural_token="<|natural|>",
    )
    output = processor("Hello, world!", "<|natural|><general>1girl, solo")
    print(output)
