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

    def fill_encoder_embedding_to_decoder_embed(
        self,
        input_ids: torch.LongTensor,
        decoder_embed: torch.FloatTensor,
        decoder_attention_mask: torch.LongTensor,
        encoder_embed: torch.FloatTensor,  # this is padded
        encoder_attention_mask: torch.LongTensor,
    ):
        """
        Fill the encoder embedding to the decoder embedding.

        Given the input_ids and replace the natural_token_index with the encoder embedding.

        Example inputs:
        - natural_token_index = 32000
        - input_ids = [
            [1, 2, 3, 32000, 4, 5, 6, 7],           # 8 tokens
            [1, 2, 32000, 8, 9, 0, 0, 0],           # 5 tokens with 3 padding (0 is pad token id)
            [1, 32000, 10, 11, 12, 0, 0, 0],        # 5 tokens with 3 padding
        ]
        - decoder_embed = [
            [1.1, 1.2, 1.3, 0.0, 1.4, 1.5, 1.6, 1.7],           # 8 tokens
            [1.1, 1.2, 0.0, 1.3, 1.4, 0.0, 0.0, 0.0],                          # 5 tokens with 3 padding
            [1.1, 0.0, 1.2, 1.3, 1.4, 0.0, 0.0, 0.0],                          # 5 tokens with 3 padding
        ]
        - decoder_attention_mask = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ]
        - encoder_embed = [
            [0.1, 0.2, 0.3, 0.4],           # 4 tokens
            [0.4, 0.5, 0.0, 0.0],           # 2 tokens with 2 padding
            [0.7, 0.8, 0.9, 0.0],           # 3 tokens with 1 padding
        ]
        - encoder_attention_mask = [
            [1, 1, 1, 1],                   # 4 tokens
            [1, 1, 0, 0],                   # 2 tokens with 2 padding
            [1, 1, 1, 0],                   # 3 tokens with 1 padding
        ]

        Return:
        - filled_decoder_embed = [
            [1.1, 1.2, 1.3, 0.1, 0.2, 0.3, 0.4, 1.4, 1.5, 1.6, 1.7],            # 8-1+4 tokens
            [1.1, 1.2, 0.4, 0.5, 1.3, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0],            # 5-1+2 tokens with 5 padding
            [1.1, 0.7, 0.8, 0.9, 1.2, 1.3, 1.4, 0.0, 0.0, 0.0, 0.0],            # 5-1+3 tokens with 4 padding
        ]
        - filled_decoder_attention_mask = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],                                  # 8-1+4 tokens
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],                                  # 5-1+2 tokens with 5 padding
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],                                  # 5-1+3 tokens with 4 padding
        ]
        """
        batch_size, seq_len = input_ids.shape
        embed_dim = decoder_embed.size(-1)
        special_token_mask = (
            input_ids == self.config.encoder_token_index
        )  # (batch_size, seq_len)
        num_special_tokens_per_sample = special_token_mask.sum(dim=1)  # (batch_size,)

        encoder_seq_lens = encoder_attention_mask.sum(dim=1)  # (batch_size,)
        new_seq_lens = seq_len - num_special_tokens_per_sample + encoder_seq_lens
        max_new_seq_len = new_seq_lens.max().item()

        filled_decoder_embed = decoder_embed.new_zeros(
            (batch_size, max_new_seq_len, embed_dim)
        )
        filled_decoder_attention_mask = decoder_attention_mask.new_zeros(
            (batch_size, max_new_seq_len)
        )

        for i in range(batch_size):
            special_positions = torch.where(special_token_mask[i])[
                0
            ]  # positions of special tokens in input_ids[i]
            dec_embed = decoder_embed[i]  # (seq_len, embed_dim)
            dec_attn_mask = decoder_attention_mask[i]  # (seq_len,)
            # dec_input_ids = input_ids[i]  # (seq_len,)

            enc_embed = encoder_embed[i]  # (encoder_seq_len, embed_dim)
            enc_seq_len = encoder_seq_lens[i].item()
            enc_embed = enc_embed[:enc_seq_len]  # (enc_seq_len, embed_dim)
            enc_attn_mask = encoder_attention_mask[i, :enc_seq_len]  # (enc_seq_len,)

            # Initialize lists to store new embeddings and attention masks
            new_embed_list = []
            new_attn_mask_list = []

            # Pointers to positions in decoder input
            prev_pos = 0

            # For each special token position
            for sp_pos in special_positions:
                sp_pos = sp_pos.item()
                # Append embeddings and attention mask from previous position to special position
                new_embed_list.append(
                    dec_embed[prev_pos:sp_pos]
                )  # embeddings from prev_pos to sp_pos-1
                new_attn_mask_list.append(dec_attn_mask[prev_pos:sp_pos])

                # Append encoder embeddings and attention mask
                new_embed_list.append(enc_embed)
                new_attn_mask_list.append(enc_attn_mask)

                # Move the pointer
                prev_pos = sp_pos + 1  # skip the special token

            # Append remaining embeddings and attention mask after the last special token
            new_embed_list.append(dec_embed[prev_pos:])
            new_attn_mask_list.append(dec_attn_mask[prev_pos:])

            # Concatenate all parts
            new_embed = torch.cat(new_embed_list, dim=0)
            new_attn_mask = torch.cat(new_attn_mask_list, dim=0)

            # Pad to max_new_seq_len
            pad_len = max_new_seq_len - new_embed.size(0)
            if pad_len > 0:
                new_embed = torch.cat(
                    [new_embed, dec_embed.new_zeros((pad_len, embed_dim))], dim=0
                )
                new_attn_mask = torch.cat(
                    [new_attn_mask, dec_attn_mask.new_zeros(pad_len)], dim=0
                )

            # Assign to filled tensors
            filled_decoder_embed[i] = new_embed
            filled_decoder_attention_mask[i] = new_attn_mask

        return filled_decoder_embed, filled_decoder_attention_mask

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        decoder_attention_mask: torch.LongTensor,
        encoder_input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        **kwargs,
    ):
        encoder_embed = self.encoder_model(encoder_input_ids).last_hidden_state
        projected_embed = self.projection(encoder_embed)

        decoder_embed = self.decoder_model.get_input_embeddings()(decoder_input_ids)
        filled_decoder_embed, filled_decoder_attention_mask = (
            self.fill_encoder_embedding_to_decoder_embed(
                input_ids=decoder_input_ids,
                decoder_embed=decoder_embed,
                decoder_attention_mask=decoder_attention_mask,
                encoder_embed=projected_embed,
                encoder_attention_mask=encoder_attention_mask,
            )
        )

        decoder_outputs = self.decoder_model(
            inputs_embeds=filled_decoder_embed,
            attention_mask=filled_decoder_attention_mask,
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
        natural_token="<|natural|>",
    )

    model = NDartForConditionalGeneration._from_config(
        NDartConfig(
            natural_config=bert.config,
            tag_config=dart.config,
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
            encoder_embed = model.encoder_model(
                natural_encoded.input_ids
            ).last_hidden_state
            print(
                f"Encoder embeddings: {encoder_embed.shape} {encoder_embed}",
            )

            projected_embed = model.projection(encoder_embed)
            print(
                f"Projected embeddings: {projected_embed}",
            )
            print(
                f"Decoder input_ids: {tag_encoded.input_ids}",
            )
            decoder_embed = model.decoder_model.get_input_embeddings()(
                tag_encoded.input_ids
            )
            print(
                f"Decoder embeddings: {decoder_embed.shape} {decoder_embed}",
            )

            filled_decoder_embed, filled_decoder_attention_mask = (
                model.fill_encoder_embedding_to_decoder_embed(
                    input_ids=tag_encoded.input_ids,
                    decoder_embed=decoder_embed,
                    decoder_attention_mask=tag_encoded.attention_mask,
                    encoder_embed=projected_embed,
                    encoder_attention_mask=natural_encoded.attention_mask,
                )
            )

            print(
                f"Filled decoder embeddings: {filled_decoder_embed.shape} {filled_decoder_embed}",
            )
