from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING


class NDartConfig(PretrainedConfig):
    model_type = "ndart"
    is_composition = True

    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        ignore_index=-100,
        natural_token_index=32000,
        projector_hidden_act="gelu",
        encoder_feature_layer=-1,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.natural_token_index = natural_token_index
        self.projector_hidden_act = projector_hidden_act
        self.encoder_feature_layer = encoder_feature_layer

        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = (
                encoder_config["model_type"]
                if "model_type" in encoder_config
                else "bert"  # e5
            )
        elif encoder_config is None:
            encoder_config = CONFIG_MAPPING["bert"](  # intfloat/multilingual-e5-small
                hidden_act="gelu",
                hidden_size=384,
                intermediate_size=1536,
                layer_norm_eps=1e-12,
                max_position_embeddings=512,
                num_attention_heads=12,
                num_hidden_layers=12,
                pad_token_id=0,
                position_embedding_type="absolute",
                type_vocab_size=2,
                use_cache=True,
                vocab_size=250037,
            )

        self.encoder_config = encoder_config

        if isinstance(decoder_config, dict):
            decoder_config["model_type"] = (
                decoder_config["model_type"]
                if "model_type" in decoder_config
                else "llama"  # dart
            )
        elif decoder_config is None:
            decoder_config = CONFIG_MAPPING["llama"](  # dart
                attention_bias=False,
                head_dim=96,
                hidden_act="silu",
                hidden_size=768,
                intermediate_size=3072,
                max_position_embeddings=1024,
                mlp_bias=True,
                num_attention_heads=8,
                num_hidden_layers=8,
                num_key_value_heads=1,
                rms_norm_eps=1e-5,
                tie_word_embeddings=False,
                use_cache=True,
                vocab_size=37540,
            )

        self.decoder_config = decoder_config

        super().__init__(**kwargs)
