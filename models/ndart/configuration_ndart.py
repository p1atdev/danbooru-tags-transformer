from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING


class NDartConfig(PretrainedConfig):
    model_type = "ndart"
    is_composition = True

    def __init__(
        self,
        natural_config=None,
        tag_config=None,
        ignore_index=-100,
        natural_token_index=32000,
        projector_hidden_act="gelu",
        natural_feature_layer=-1,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.natural_token_index = natural_token_index
        self.projector_hidden_act = projector_hidden_act
        self.natural_feature_layer = natural_feature_layer

        if isinstance(natural_config, dict):
            natural_config["model_type"] = (
                natural_config["model_type"]
                if "model_type" in natural_config
                else "bert"  # e5
            )
        elif natural_config is None:
            natural_config = CONFIG_MAPPING["bert"](  # intfloat/multilingual-e5-small
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

        self.natural_config = natural_config

        if isinstance(tag_config, dict):
            tag_config["model_type"] = (
                tag_config["model_type"]
                if "model_type" in tag_config
                else "llama"  # dart
            )
        elif tag_config is None:
            tag_config = CONFIG_MAPPING["llama"](  # dart
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

        self.tag_config = tag_config

        super().__init__(**kwargs)
