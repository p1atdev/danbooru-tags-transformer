from transformers.configuration_utils import PretrainedConfig


class Dart2VecConfig(PretrainedConfig):
    """Configuration for Dart2Vec model"""

    def __init__(
        self,
        vocab_size=9462,
        hidden_size=768,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
