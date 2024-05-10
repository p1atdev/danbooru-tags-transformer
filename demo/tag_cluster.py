import json

import gradio as gr

CLUSTER = {
    "256": {
        "path": "data/cluster_map_256c.json",
        "label2cluster": None,
        "cluster2labels": None,
    },
    "512": {
        "path": "data/cluster_map_512c.json",
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
    prepare_cluster_map("256")
    prepare_cluster_map("512")

    def get_same_cluster(tag: str, cluster: str = "256"):
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
                    choices=["256", "512"],
                    value="256",
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
                    value=get_same_cluster("blue hair", "256"),
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
