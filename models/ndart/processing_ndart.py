import os
import json
import warnings
from pathlib import Path


import torch
import torch.nn as nn

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    AutoTokenizer,
    ProcessorMixin,
    BatchFeature,
)
from transformers.utils import (
    direct_transformers_import,
    PROCESSOR_NAME,
    CHAT_TEMPLATE_NAME,
)
from transformers.utils import logging
from transformers.dynamic_module_utils import custom_object_save

logger = logging.get_logger(__name__)

# Dynamically import the Transformers module to grab the attribute classes of the processor form their names.
transformers_module = direct_transformers_import(Path(__file__).parent)


class NDartProcessorKwargs:
    _defaults = {
        "encoder_kwargs": {
            "padding": True,
        },
        "decoder_kwargs": {
            "padding": True,
        },
    }


class NDartProcessor(ProcessorMixin):
    attributes = ["encoder_tokenizer", "decoder_tokenizer"]
    valid_kwargs = ["chat_template", "natural_token"]
    encoder_tokenizer_class = "AutoTokenizer"
    decoder_tokenizer_class = "AutoTokenizer"

    encoder_tokenizer: PreTrainedTokenizer
    decoder_tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        encoder_tokenizer=None,
        decoder_tokenizer=None,
        natural_token="<|natural|>",
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            encoder_tokenizer, decoder_tokenizer, chat_template=chat_template, **kwargs
        )
        self.natural_token_id: int = self.decoder_tokenizer.convert_tokens_to_ids(
            natural_token
        )

    def __call__(
        self,
        natural_text: str | list[str] | None = None,
        tag_text: str | list[str] | None = None,
        **kwargs,
    ) -> BatchFeature:
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
            **NDartProcessorKwargs._defaults["encoder_kwargs"],
            "return_tensors": "pt",
            **kwargs,
        }
        tag_output_kwargs = {
            **NDartProcessorKwargs._defaults["decoder_kwargs"],
            "return_tensors": "pt",
            **kwargs,
        }

        natural_tokens = self.encoder_tokenizer(
            natural_text,
            **natural_output_kwargs,
        )
        tag_tokens = self.decoder_tokenizer(
            tag_text,
            **tag_output_kwargs,
        )

        tag_input_ids, tag_attention_mask = self.insert_encoder_tokens_batch(
            encoder_input_ids=natural_tokens["input_ids"],
            encoder_attention_mask=natural_tokens["attention_mask"],
            decoder_input_ids=tag_tokens["input_ids"],
            decoder_attention_mask=tag_tokens["attention_mask"],
        )

        return BatchFeature(
            data={
                "input_ids": tag_input_ids,
                "attention_mask": tag_attention_mask,
                "encoder_input_ids": natural_tokens["input_ids"],
                "encoder_attention_mask": natural_tokens["attention_mask"],
            }
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

        has_decoder_attention_mask = decoder_attention_mask is not None

        encoder_token_lens = (
            encoder_attention_mask.sum(dim=1)
            if encoder_attention_mask is not None
            else encoder_input_ids.size(1)
        )
        # decoder_input_ids内のnatural_token_idの位置を取得
        natural_positions = (decoder_input_ids == self.natural_token_id).nonzero(
            as_tuple=False
        )[:, 1]  # (batch_index, in_batch_position) -> (in_batch_position)
        assert len(natural_positions) == decoder_input_ids.size(
            0
        ), f"Natural token not found in decoder_input_ids: {len(natural_positions)} != "

        for i, (input_ids_i, position, encoder_len) in enumerate(
            zip(
                decoder_input_ids,
                natural_positions,
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

            if has_decoder_attention_mask:
                attention_mask_i = decoder_attention_mask[i]
                new_attention_mask_i = torch.cat(
                    [
                        attention_mask_i[:position]
                        if position > 0
                        else torch.tensor([]),
                        torch.ones_like(replacement_tokens),
                        attention_mask_i[position + 1 :]
                        if position + 1 < len(attention_mask_i)
                        else torch.tensor([]),
                    ]
                )
                new_attention_mask.append(new_attention_mask_i)
                assert (
                    new_attention_mask_i.size(0) == new_input_ids_i.size(0)
                ), f"Attention mask size mismatch: {new_attention_mask_i.size(0)} != {new_input_ids_i.size(0)}"

        # padding right
        new_input_ids_tensor = nn.utils.rnn.pad_sequence(
            new_input_ids,
            batch_first=True,
            padding_value=self.decoder_tokenizer.pad_token_id,
        )
        new_attention_mask_tensor = (
            nn.utils.rnn.pad_sequence(
                new_attention_mask,
                batch_first=True,
                padding_value=0,
            )
            if has_decoder_attention_mask
            else None
        )
        return new_input_ids_tensor, new_attention_mask_tensor

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->PreTrained
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.decoder_tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->PreTrained
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.decoder_tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["natural_text", "tag_text"]

    # edit from: https://github.com/huggingface/transformers/blob/1d063793318b20654ebb850f48f43e0a247ab7bb/src/transformers/processing_utils.py#L980-L995
    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            subfolder = attribute_name  # subfolder is the same as attribute_name
            if isinstance(class_name, tuple):
                classes = tuple(
                    getattr(transformers_module, n) if n is not None else None
                    for n in class_name
                )
                use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = getattr(transformers_module, class_name)

            args.append(
                attribute_class.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=subfolder,
                    **kwargs,
                )
            )
        return args

    # edit from: https://github.com/huggingface/transformers/blob/1d063793318b20654ebb850f48f43e0a247ab7bb/src/transformers/processing_utils.py#L460-L560
    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
        can be reloaded using the [`~ProcessorMixin.from_pretrained`] method.

        <Tip>

        This class method is simply calling [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`]. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)
        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            attrs = [
                getattr(self, attribute_name) for attribute_name in self.attributes
            ]
            configs = [
                (a.init_kwargs if isinstance(a, PreTrainedTokenizerBase) else a)
                for a in attrs
            ]
            configs.append(self)
            custom_object_save(self, save_directory, config=configs)

        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name)
            # Include the processor class in the attribute config so this processor can then be reloaded with the
            # `AutoProcessor` API.
            if hasattr(attribute, "_set_processor_class"):
                attribute._set_processor_class(self.__class__.__name__)
            attribute.save_pretrained(
                os.path.join(
                    save_directory,
                    attribute_name,  # CHANGED: save to subfolder
                ),
            )

        if self._auto_class is not None:
            # We added an attribute to the init_kwargs of the tokenizers, which needs to be cleaned up.
            for attribute_name in self.attributes:
                attribute = getattr(self, attribute_name)
                if isinstance(attribute, PreTrainedTokenizerBase):
                    del attribute.init_kwargs["auto_map"]

        # If we save using the predefined names, we can load using `from_pretrained`
        # plus we save chat_template in its own file
        output_processor_file = os.path.join(save_directory, PROCESSOR_NAME)
        output_chat_template_file = os.path.join(save_directory, CHAT_TEMPLATE_NAME)

        processor_dict = self.to_dict()
        # Save `chat_template` in its own file. We can't get it from `processor_dict` as we popped it in `to_dict`
        # to avoid serializing chat template in json config file. So let's get it from `self` directly
        if self.chat_template is not None:
            chat_template_json_string = (
                json.dumps(
                    {"chat_template": self.chat_template}, indent=2, sort_keys=True
                )
                + "\n"
            )
            with open(output_chat_template_file, "w", encoding="utf-8") as writer:
                writer.write(chat_template_json_string)
            logger.info(f"chat template saved in {output_chat_template_file}")

        # For now, let's not save to `processor_config.json` if the processor doesn't have extra attributes and
        # `auto_map` is not specified.
        if set(processor_dict.keys()) != {"processor_class"}:
            self.to_json_file(output_processor_file)
            logger.info(f"processor saved in {output_processor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        if set(processor_dict.keys()) == {"processor_class"}:
            return []
        return [output_processor_file]
