import json

import gradio as gr

CLUSTER = {
    "1024-1": {
        "path": "data/cluster_map_1024c1.json",
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


def prepare_cluster_map(id: str):
    with open(CLUSTER[id]["path"], "r") as f:
        label2cluster = json.load(f)

    cluster2labels = {
        cluster: [label for label, c in label2cluster.items() if c == cluster]
        for cluster in set(label2cluster.values())
    }

    CLUSTER[id]["label2cluster"] = label2cluster
    CLUSTER[id]["cluster2labels"] = cluster2labels


def demo():
    for key in CLUSTER:
        prepare_cluster_map(key)

    def get_same_cluster(tag: str, cluster: str = "1024"):
        if tag not in CLUSTER[cluster]["label2cluster"]:
            return f"Unknown tag: {tag}"

        cluster_id = CLUSTER[cluster]["label2cluster"][tag]

        tags = CLUSTER[cluster]["cluster2labels"][cluster_id]

        tag_dict = {tag: 1 for tag in tags}
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
                    value=get_same_cluster("blue hair", list(CLUSTER.keys())[0]),
                )

        tag_input.change(
            get_same_cluster,
            inputs=[tag_input, cluster_radio],
            outputs=[similar_tags_label],
        )
        cluster_radio.change(
            get_same_cluster,
            inputs=[tag_input, cluster_radio],
            outputs=[similar_tags_label],
        )

    ui.launch()


if __name__ == "__main__":
    demo()
