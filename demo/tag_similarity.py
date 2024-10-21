import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import gradio as gr

TOKENIZER_NAME = "p1atdev/dart-v2-moe-sft"
MODEL_NAME = "p1atdev/dart-v2-moe-sft"


def prepare_embeddings() -> torch.Tensor:
    model = AutoModel.from_pretrained(MODEL_NAME)

    embeddings = model.get_input_embeddings().weight.detach().cpu()

    return embeddings


def label_to_id(tokenizer, label: str | list[str]) -> int | list[int]:
    return tokenizer.convert_tokens_to_ids(label)


def id_to_label(tokenizer, id: int | list[int]) -> str | list[str]:
    return tokenizer.convert_ids_to_tokens(id)


def load_tags(path: str):
    with open(path, "r") as f:
        tags = f.read().splitlines()
    return tags


def demo():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    embeddings = prepare_embeddings()
    characters = load_tags("data/character_tags.txt")
    id2label = {
        id: tag for tag, id in tokenizer.get_vocab().items() if tag in characters
    }

    def get_similar_tags(tags: str, top_k=30):
        tag_list = [
            tag for tag in tokenizer.tokenize(tags) if tag != tokenizer.unk_token
        ]
        if len(tag_list) == 0:
            return f"Unknown tags: {tags}"

        tag_emb = embeddings[label_to_id(tokenizer, tag_list)].unsqueeze(1)

        similarities = torch.cosine_similarity(
            tag_emb,
            embeddings.unsqueeze(0),
            dim=2,
        ).mean(dim=0)

        similar_tags = {
            tag: sim
            for tag, sim in sorted(
                [
                    (id2label[i], sim)
                    for i, sim in enumerate(similarities.tolist())
                    if i in id2label
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
        }
        return similar_tags

    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                tag_input = gr.Textbox(
                    label="Enter tags",
                    value="blue hair, twintails",
                    placeholder="blue hair, twintails",
                )
                top_k_input = gr.Slider(
                    label="Top K",
                    minimum=2,
                    maximum=100,
                    value=30,
                    step=1,
                )

            with gr.Column():
                similar_tags_label = gr.Label(
                    label="Similar tags", value=get_similar_tags("blue hair, twintails")
                )

        tag_input.change(
            get_similar_tags,
            inputs=[tag_input, top_k_input],
            outputs=[similar_tags_label],
        )

    ui.launch()


if __name__ == "__main__":
    demo()
