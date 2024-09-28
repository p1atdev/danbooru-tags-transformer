import sys

sys.path.append(".")

import argparse

from src.cluster import TagCluster


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, default=1440)
    parser.add_argument("--n_init", type=int, default=25)
    parser.add_argument("--max_iter", type=int, default=250)
    parser.add_argument(
        "--output_path", "-o", type=str, default="data/cluster_map.json"
    )
    parser.add_argument(
        "--general_tags",
        type=str,
        default="data/general_tags.txt",
    )

    return parser.parse_args()


def train_cluster(
    model_name: str,
    n_clusters: int = 1440,
    n_init: int = 25,
    max_iter: int = 250,
    general_tags_path: str = "data/general_tags.txt",
):
    with open(general_tags_path, "r") as f:
        general_tags = f.read().splitlines()

    cluster = TagCluster.train_from_embedding_model(
        embedding_model_name=model_name,
        n_clusters=n_clusters,
        n_init=n_init,  # こっち増やした方が良さそう
        max_iter=max_iter,  # これは増やさなくて良さそう
        tag_list=general_tags,
    )

    return cluster


def main():
    args = prepare_args()
    output_path = args.output_path

    print("Training cluster map...")
    cluster = train_cluster(
        model_name=args.model_name,
        n_clusters=args.n_clusters,
        n_init=args.n_init,
        max_iter=args.max_iter,
        general_tags_path=args.general_tags,
    )

    cluster.save_pretrained(output_path)
    print(f"Cluster map saved to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
