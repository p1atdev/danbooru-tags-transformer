from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from transformers import (
    AutoModelForTextEncoding,
    AutoModelForCausalLM,
    GenerationMixin,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.activations import ACT2FN

from .configuration_ndart import NDartConfig


@dataclass
class NDartCausalLMOutputWithPast(ModelOutput):
    """
    Base class for NDart causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        encoder_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, sequence_length, hidden_size)`.
            encoder_hidden_states of the model produced by the natural text encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[torch.FloatTensor] = None


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
            config.encoder_config.hidden_size,  # type: ignore
            config.decoder_config.hidden_size,  # type: ignore
            bias=True,
        )
        self.activation = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.decoder_config.hidden_size,  # type: ignore
            config.decoder_config.hidden_size,  # type: ignore
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
            else self.config.decoder_config.initializer_range
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

        self.encoder_model = AutoModelForTextEncoding.from_config(config.encoder_config)
        self.encoder_model.main_input_name = "encoder_input_ids"
        self.decoder_model = AutoModelForCausalLM.from_config(config.decoder_config)
        self.decoder_model.main_input_name = "decoder_input_ids"

        self.projection = NDartEmbeddingProjector(config)

        self.vocab_size = self.decoder_model.config.vocab_size
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        self.post_init()

    def _replace_natural_token_embeddings(
        self,
        encoder_embeds: torch.Tensor,
        decoder_input_ids: torch.LongTensor,
        decoder_embeds: torch.Tensor,
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
        input_ids: torch.LongTensor,  # for decoder
        attention_mask: torch.LongTensor,  # for decoder
        encoder_input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        encoder_inpts_embeds: torch.FloatTensor | None = None,
        projected_embeds: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,  # for decoder
        past_key_values: list[torch.FloatTensor] | None = None,
        encoder_feature_layer: int | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        num_logits_to_keep: int | None = None,
        **kwargs,
    ):
        # 0. validate inputs
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        encoder_feature_layer = (
            encoder_feature_layer
            if encoder_feature_layer is not None
            else self.config.encoder_feature_layer
        )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if (encoder_input_ids is None) ^ (encoder_inpts_embeds is not None):
            raise ValueError(
                "You must specify exactly one of encoder_input_ids or encoder_inpts_embeds"
            )

        # 1. encode natural text
        if encoder_inpts_embeds is None:
            encoder_inpts_embeds = self.encoder_model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
            ).last_hidden_state

        # 2. project encoder embeddings
        if projected_embeds is None:
            projected_embeds = self.projection(encoder_inpts_embeds)
        assert projected_embeds is not None

        # 3. get decoder embeddings
        if inputs_embeds is None:
            inputs_embeds = self.decoder_model.get_input_embeddings()(input_ids)
        assert inputs_embeds is not None

        # 4. replace natural token embeddings
        decoder_inputs_embeds = self._replace_natural_token_embeddings(
            encoder_embeds=projected_embeds,
            decoder_input_ids=input_ids,
            decoder_embeds=inputs_embeds,
            encoder_attention_mask=encoder_attention_mask,
        )

        # 5. forward decoder
        outputs = self.decoder_model(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        logits = outputs[0]

        # 6. loss calculation
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(
                    logits.device
                )
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            num_items = kwargs.pop("num_items", None)
            loss = nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
                reduction="sum",
            )
            loss = loss / num_items

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return NDartCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            encoder_hidden_states=projected_embeds,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = self.decoder_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        return model_inputs
