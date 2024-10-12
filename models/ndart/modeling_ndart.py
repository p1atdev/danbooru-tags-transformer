import torch
import torch.nn as nn

from transformers import (
    AutoModelForTextEncoding,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    BertModel,
    PreTrainedTokenizerBase,
    AutoTokenizer,
    GenerationMixin,
    PreTrainedModel,
)
from transformers.activations import ACT2FN

from configuration_ndart import NDartConfig


class NDartEmbeddingProjector(nn.Module):
    linear_1: nn.Linear
    linear_2: nn.Linear
    activation: nn.Module

    def __init__(
        self,
        config: NDartConfig,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(
            config.natural_config.hidden_size,  # type: ignore
            config.tag_config.hidden_size,  # type: ignore
            bias=True,
        )
        self.activation = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.tag_config.hidden_size,  # type: ignore
            config.tag_config.hidden_size,  # type: ignore
            bias=True,
        )

    def forward(self, encoder_states: torch.FloatTensor):
        hidden_states = self.linear_1(encoder_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states


class NDartPreTrainedModel(PreTrainedModel):
    config_class = NDartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.tag_config.initializer_range
        )
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class NDartForConditionalGeneration(NDartPreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config: NDartConfig,
    ):
        super().__init__(config)

        self.encoder_model = AutoModelForTextEncoding.from_config(config.natural_config)
        self.decoder_model = AutoModelForCausalLM.from_config(config.tag_config)

        self.projection = NDartEmbeddingProjector(config)

        self.vocab_size = self.decoder_model.config.vocab_size
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        self.post_init()

    def _replace_natural_token_embeddings(
        self,
        encoder_embeds: torch.FloatTensor,
        decoder_input_ids: torch.LongTensor,
        decoder_embeds: torch.FloatTensor,
        encoder_attention_mask: torch.LongTensor | None = None,
    ):
        natural_token_mask = (
            (decoder_input_ids == self.config.natural_token_index)
            .unsqueeze(-1)
            .expand_as(decoder_embeds)
            .to(decoder_embeds.device)
        )
        _batch_size, _seq_len, dim = decoder_embeds.size()
        if encoder_attention_mask is not None:
            encoder_embeds = torch.masked_select(
                encoder_embeds,
                encoder_attention_mask.unsqueeze(-1).expand_as(encoder_embeds).bool(),
            ).view(-1, dim)

        decoder_embeds = decoder_embeds.masked_scatter(
            natural_token_mask,
            encoder_embeds.to(decoder_embeds.device, decoder_embeds.dtype),
        )

        return decoder_embeds

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        encoder_input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ):
        encoder_embed = self.encoder_model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        ).last_hidden_state
        projected_embeds = self.projection(encoder_embed)

        decoder_embeds = self.decoder_model.get_input_embeddings()(decoder_input_ids)
        decoder_embeds = self._replace_natural_token_embeddings(
            encoder_embeds=projected_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_embeds=decoder_embeds,
            encoder_attention_mask=encoder_attention_mask,
        )

        decoder_outputs = self.decoder_model(
            inputs_embeds=decoder_embeds,
            attention_mask=decoder_attention_mask,
            **kwargs,
        )

        return decoder_outputs

    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs
    # )


if __name__ == "__main__":
    from processing_ndart import NDartProcessor

    encoder_model = "intfloat/multilingual-e5-small"
    decoder_model = "p1atdev/dart-v3-llama-8L-241005_241008-sft-2"
    bert = AutoModelForTextEncoding.from_pretrained(encoder_model)
    dart = AutoModelForCausalLM.from_pretrained(decoder_model)
    processor = NDartProcessor(
        natural_tokenizer=AutoTokenizer.from_pretrained(encoder_model),
        tag_tokenizer=AutoTokenizer.from_pretrained(decoder_model),
        natural_token="<|reserved_0|>",
    )

    model = NDartForConditionalGeneration._from_config(
        NDartConfig(
            natural_config=bert.config,
            tag_config=dart.config,
            natural_token_index=processor.natural_token_id,
        )
    )
    model.encoder_model = bert
    model.decoder_model = dart
    print(model)

    tag_prompt = [
        "<|bos|><projection><|reserved_0|></projection>",
        "<projection><|reserved_0|></projection><copyright></copyright><character></character><general><|input_end|>",
    ]
    natural_text = [
        "an image",
        "黒髪ロング猫耳美少女JK",
    ]

    with torch.no_grad():
        with torch.autocast(device_type="cpu"):
            processor_output = processor(
                natural_text=natural_text,
                tag_text=tag_prompt,
                return_tensors="pt",
            )
            natural_encoded = processor_output.natural
            tag_encoded = processor_output.tag

            print(
                f"Encoder input_ids: {natural_encoded.input_ids}",
            )
            encoder_embeds = model.encoder_model(
                input_ids=natural_encoded.input_ids,
                attention_mask=natural_encoded.attention_mask,
            ).last_hidden_state
            print(
                f"Encoder embeddings: {encoder_embeds.shape} {encoder_embeds[:, :, 0]}",
            )
            print(
                f"Natural attention mask: {natural_encoded.attention_mask}",
            )

            projected_embeds = model.projection(encoder_embeds)
            print(
                f"Projected embeddings: {projected_embeds[:, :, 0]}",
            )
            print(
                f"Decoder input_ids: {tag_encoded.input_ids}",
            )
            decoder_embeds = model.decoder_model.get_input_embeddings()(
                tag_encoded.input_ids,
            )
            print(
                f"Decoder embeddings: {decoder_embeds.shape} {decoder_embeds[:, :, 0]}",
            )

            replaced_decoder_embeds = model._replace_natural_token_embeddings(
                encoder_embeds=projected_embeds,
                decoder_input_ids=tag_encoded.input_ids,
                decoder_embeds=decoder_embeds,
                encoder_attention_mask=natural_encoded.attention_mask,
            )

            print(
                f"Replaced decoder embeddings: {replaced_decoder_embeds.shape} {replaced_decoder_embeds[:, :, 0]}",
            )
