import os

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    GemmaConfig,
    GemmaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
)

CONFIGS = {
    "gemma": {
        "config": GemmaConfig,
        "model": GemmaForCausalLM,
    },
    "llama": {
        "config": LlamaConfig,
        "model": LlamaForCausalLM,
    },
    "mistral": {
        "config": MistralConfig,
        "model": MistralForCausalLM,
    },
    "mixtral": {
        "config": MixtralConfig,
        "model": MixtralForCausalLM,
    },
}
CONFIGS_PATH = "./config"


def get_model_config(model_type: str, model_size: str):
    config = CONFIGS[model_type]["config"].from_pretrained(
        os.path.join(CONFIGS_PATH, model_type, model_size + ".json")
    )
    return config, CONFIGS[model_type]["model"]
