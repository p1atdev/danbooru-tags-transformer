import json

import numpy as np
import gradio as gr

CLUSTER = {
    "1024-1": {
        "path": "data/cluster_map_1024c1.json",
        "label2cluster": None,
        "cluster2labels": None,
    },
    "1024-2": {
        "path": "data/cluster_map_1024c2.json",
        "label2cluster": None,
        "cluster2labels": None,
    },
    "1152-1": {
        "path": "data/cluster_map_1152c1.json",
        "label2cluster": None,
        "cluster2labels": None,
    },
    "1280-1": {
        "path": "data/cluster_map_1280c1.json",
        "label2cluster": None,
        "cluster2labels": None,
    },
    "1280-2": {
        "path": "data/cluster_map_1280c2.json",
        "label2cluster": None,
        "cluster2labels": None,
    },
    "1440-3": {
        "path": "data/cluster_map_1440c3.json",
        "label2cluster": None,
        "cluster2labels": None,
    },
    # "1440-4": {
    #     "path": "data/cluster_map_1440c4.json",
    #     "label2cluster": None,
    #     "cluster2labels": None,
    # },
    # "1600": {
    #     "path": "data/cluster_map_1600c.json",
    #     "label2cluster": None,
    #     "cluster2labels": None,
    # },
    "1600-2": {
        "path": "data/cluster_map_1600c2.json",
        "label2cluster": None,
        "cluster2labels": None,
    },
    "1600-3": {
        "path": "data/cluster_map_1600c3.json",
        "label2cluster": None,
        "cluster2labels": None,
    },
}

FREQUENCY_PATH = "data/tag_frequency.json"


def prepare_cluster_map(id: str):
    with open(CLUSTER[id]["path"], "r") as f:
        label2cluster = json.load(f)

    cluster2labels = {
        cluster: [label for label, c in label2cluster.items() if c == cluster]
        for cluster in set(label2cluster.values())
    }

    CLUSTER[id]["label2cluster"] = label2cluster
    CLUSTER[id]["cluster2labels"] = cluster2labels


def prepare_frequency():
    with open(FREQUENCY_PATH, "r") as f:
        tag_to_frequency = json.load(f)

    return tag_to_frequency


def softmax(x: np.ndarray):
    # 数値の安定性を確保するために最大値を引く
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    result = exps / np.sum(exps)
    return result


def softmax_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    shifted_logits = scaled_logits - np.max(scaled_logits)
    exps = np.exp(shifted_logits)
    result = exps / np.sum(exps)
    return result


def demo():
    for key in CLUSTER:
        prepare_cluster_map(key)

    frequency = prepare_frequency()

    def get_same_cluster(tag: str, cluster: str = "1024", temperature: float = 1.0):
        if tag not in CLUSTER[cluster]["label2cluster"]:
            return f"Unknown tag: {tag}"

        cluster_id = CLUSTER[cluster]["label2cluster"][tag]
        tags = CLUSTER[cluster]["cluster2labels"][cluster_id]

        frequencies = np.array([frequency[tag] for tag in tags])
        # print(frequencies)
        # avg
        frequencies = frequencies / np.sum(frequencies)
        # print(frequencies)
        # softmax
        frequencies = softmax_with_temperature(frequencies, temperature)
        # print(frequencies)

        tag_dict = {tag: freq for tag, freq in zip(tags, frequencies)}
        return tag_dict

    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                tag_input = gr.Textbox(
                    label="Enter a tag", value="blue hair", placeholder="blue hair"
                )
                cluster_radio = gr.Radio(
                    label="Cluster size",
                    choices=list(CLUSTER.keys()),
                    value=list(CLUSTER.keys())[0],
                )
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=4.0,
                    value=1.0,
                    step=0.1,
                )

                gr.Examples(
                    examples=[
                        ["blue hair"],
                        ["white shirt"],
                        ["serafuku"],
                        ["grey background"],
                        ["smile"],
                        ["twintails"],
                        ["multiple views"],
                        ["maid"],
                        ["cityscape"],
                    ],
                    inputs=[tag_input],
                )

            with gr.Column():
                similar_tags_label = gr.Label(
                    label="Tags in the same cluster",
                    value=get_same_cluster("blue hair", list(CLUSTER.keys())[0], 1.0),
                )

        tag_input.change(
            get_same_cluster,
            inputs=[tag_input, cluster_radio, temperature_slider],
            outputs=[similar_tags_label],
        )
        cluster_radio.change(
            get_same_cluster,
            inputs=[tag_input, cluster_radio, temperature_slider],
            outputs=[similar_tags_label],
        )
        temperature_slider.change(
            get_same_cluster,
            inputs=[tag_input, cluster_radio, temperature_slider],
            outputs=[similar_tags_label],
        )

    ui.launch()


if __name__ == "__main__":
    demo()
