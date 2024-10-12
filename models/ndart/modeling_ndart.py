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
)
from transformers.activations import ACT2FN


class NDartEmbeddingProjector(nn.Module):
    linear_1: nn.Linear
    linear_2: nn.Linear
    activation: nn.Module

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "gelu",
    ):
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.activation = ACT2FN[activation]
        self.linear_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.FloatTensor):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x


class NDartForConditionalGeneration(nn.Module, GenerationMixin):
    """Pipeline for Natural Dart model"""

    encoder_model: BertModel
    encoder_tokenizer: PreTrainedTokenizerBase
    decoder_model: LlamaForCausalLM
    decoder_tokenizer: PreTrainedTokenizerBase

    projection: NDartEmbeddingProjector

    encoder_token_index: int  # <|natural|>
    ignore_index: int = -100

    def __init__(
        self,
        encoder_model: BertModel,
        encoder_tokenizer: PreTrainedTokenizerBase,
        decoder_model: LlamaForCausalLM,
        decoder_tokenizer: PreTrainedTokenizerBase,
        encoder_token_index: int,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__()

        self.encoder_model = encoder_model
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_model = decoder_model
        self.decoder_tokenizer = decoder_tokenizer
        self.encoder_token_index = encoder_token_index
        self.ignore_index = ignore_index

        self.projection = NDartEmbeddingProjector(
            input_dim=encoder_model.config.hidden_size,
            hidden_dim=decoder_model.config.hidden_size,
            output_dim=decoder_model.config.hidden_size,
        )

    @classmethod
    def from_pretrained(
        cls,
        decoder_model_name: str,
        encoder_model_name: str | None = None,
        encoder_tokenizer_folder: str | None = "tokenizer_encoder",
        natural_token: str | None = "<|natural|>",
        ignore_index: int = -100,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        if encoder_model_name is None:
            encoder_model_name = decoder_model_name
        if natural_token is None:
            natural_token = "<|natural|>"

        encoder_model = AutoModelForTextEncoding.from_pretrained(
            encoder_model_name, torch_dtype=torch_dtype
        )
        encoder_tokenizer = AutoTokenizer.from_pretrained(
            encoder_model_name, subfolder=encoder_tokenizer_folder
        )
        decoder_model = AutoModelForCausalLM.from_pretrained(
            decoder_model_name, torch_dtype=torch_dtype
        )
        decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

        return cls(
            encoder_model=encoder_model,
            encoder_tokenizer=encoder_tokenizer,
            decoder_model=decoder_model,
            decoder_tokenizer=decoder_tokenizer,
            encoder_token_index=decoder_tokenizer.convert_tokens_to_ids(natural_token),
            ignore_index=ignore_index,
            **kwargs,
        )

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
            input_ids == self.encoder_token_index
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
            dec_input_ids = input_ids[i]  # (seq_len,)

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
    # Example usage
    pipe = NDartForConditionalGeneration.from_pretrained(
        decoder_model_name="p1atdev/dart-v3-llama-8L-241005_241008-sft-2",
        encoder_model_name="intfloat/multilingual-e5-small",
        encoder_tokenizer_folder=None,
    )
    print(pipe)

    tag_prompt = [
        "<|bos|><projection><|reserved_0|></projection>",
        "<projection><|reserved_0|></projection><copyright></copyright><character></character><general><|input_end|>",
    ]
    natural_text = [
        "an image",
        "黒髪ロング超絶美少女JK",
    ]

    with torch.no_grad():
        with torch.autocast(device_type="cpu"):
            encoder_tokens = pipe.encoder_tokenizer(
                natural_text,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
            encoder_input_ids = encoder_tokens.input_ids
            encoder_attention_mask = encoder_tokens.attention_mask
            encoder_embed = pipe.encoder_model(encoder_input_ids).last_hidden_state

            projected_embed = pipe.projection(encoder_embed)

            decoder_tokens = pipe.decoder_tokenizer(
                tag_prompt,
                return_tensors="pt",
                padding=True,
            )
            decoder_input_ids = decoder_tokens.input_ids
            decoder_attention_mask = decoder_tokens.attention_mask
            decoder_embed = pipe.decoder_model.get_input_embeddings()(decoder_input_ids)

            print(
                f"Encoder input_ids: {encoder_input_ids}",
            )
            print(
                f"Encoder embeddings: {encoder_embed.shape} {encoder_embed}",
            )
            print(
                f"Projected embeddings: {projected_embed}",
            )
            print(
                f"Decoder input_ids: {decoder_input_ids}",
            )
            print(
                f"Decoder embeddings: {decoder_embed.shape} {decoder_embed}",
            )
            filled_decoder_embed, filled_decoder_attention_mask = (
                pipe.fill_encoder_embedding_to_decoder_embed(
                    input_ids=decoder_input_ids,
                    decoder_embed=decoder_embed,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_embed=projected_embed,
                    encoder_attention_mask=encoder_attention_mask,
                )
            )

            print(
                f"Filled decoder embeddings: {filled_decoder_embed.shape} {filled_decoder_embed}",
            )
