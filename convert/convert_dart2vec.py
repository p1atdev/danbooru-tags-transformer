import sys

sys.path.append(".")

import argparse

from transformers import AutoModel, AutoTokenizer
from models.dart2vec.configuration_dart2vec import Dart2VecConfig
from models.dart2vec.modeling_dart2vec import Dart2VecModel

BASE_MODEL_NAME = "p1atdev/dart2vec-opt_8"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name",
        "-b",
        type=str,
        default=BASE_MODEL_NAME,
    )
    parser.add_argument(
        "--push_to_hub_name",
        "-p",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    base_model_name = args.base_model_name
    push_to_hub_name = args.push_to_hub_name

    print("Loading base model...")
    model = AutoModel.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # initialize
    print("Initializing Dart2Vec model...")
    Dart2VecConfig.register_for_auto_class()
    Dart2VecModel.register_for_auto_class("AutoModel")
    config = Dart2VecConfig(
        vocab_size=model.config.vocab_size,
        hidden_size=model.config.hidden_size,
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
    )
    dart2vec_model = Dart2VecModel(config)

    # load weights
    print("Copying weights...")
    dart2vec_model.set_input_embeddings(model.get_input_embeddings())

    # save
    if push_to_hub_name is not None:
        print("Saving model...")
        dart2vec_model.push_to_hub(push_to_hub_name)
        tokenizer.push_to_hub(push_to_hub_name)


if __name__ == "__main__":
    main()
