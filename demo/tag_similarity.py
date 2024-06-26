import numpy as np
from transformers import AutoTokenizer, AutoModel

import gradio as gr

TOKENIZER_NAME = "p1atdev/dart-v2-vectors"
MODEL_NAME = "p1atdev/dart-v2-vectors"


def prepare_embeddings() -> dict[str, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

    embeddings: np.ndarray = model.get_input_embeddings().weight.detach().numpy()

    labels = tokenizer.get_vocab()

    label2emb = {label: embeddings[index] for label, index in labels.items()}

    return label2emb


def demo():
    label2emb = prepare_embeddings()

    def get_similar_tags(tag: str, top_k=30):
        if tag not in label2emb:
            return f"Unknown tag: {tag}"

        tag_emb = label2emb[tag]

        similarities = {}
        for label, emb in label2emb.items():
            similarities[label] = np.dot(tag_emb, emb) / (
                np.linalg.norm(tag_emb) * np.linalg.norm(emb)
            )

        return dict(
            sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )

    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                tag_input = gr.Textbox(label="Enter a tag", placeholder="blue hair")
                top_k_input = gr.Slider(
                    label="Top K",
                    minimum=2,
                    maximum=100,
                    value=30,
                    step=1,
                )

            with gr.Column():
                similar_tags_label = gr.Label(label="Similar tags")

        tag_input.change(
            get_similar_tags,
            inputs=[tag_input, top_k_input],
            outputs=[similar_tags_label],
        )

    ui.launch()


if __name__ == "__main__":
    demo()
