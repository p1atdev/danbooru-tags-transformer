import torch
import torch.nn as nn

from typing import Optional
from dataclasses import dataclass

from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_dart2vec import Dart2VecConfig


@dataclass
class Dart2VecModelOutput(ModelOutput):
    hidden_states: torch.Tensor


@dataclass
class Dart2VecModelForFeatureExtractionOutput(ModelOutput):
    embeddings: torch.Tensor


class Dart2VecEmbeddings(nn.Module):
    def __init__(self, config: Dart2VecConfig):
        super().__init__()

        self.tag_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        if inputs_embeds is not None:
            return inputs_embeds

        embeddings = self.tag_embeddings(input_ids)

        return embeddings


class Dart2VecPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Dart2VecConfig
    base_model_prefix = "dart2vec"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class Dart2VecModel(Dart2VecPreTrainedModel):
    def __init__(self, config: Dart2VecConfig):
        super().__init__(config)

        self.config = config

        self.embeddings = Dart2VecEmbeddings(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tag_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tag_embeddings = value

    def forward(
        self, input_ids: torch.Tensor
    ) -> Dart2VecModelForFeatureExtractionOutput:
        embeddings = self.embeddings(input_ids)

        return Dart2VecModelForFeatureExtractionOutput(embeddings=embeddings)
