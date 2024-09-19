import json
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from transformers import AutoTokenizer, AutoModel
from faiss import Kmeans as FaissKmeans


CLUSTER_MAP_TYPE = dict[str, int]

CLUSTRING_ALGORITHM = Literal["sklearn", "faiss"]


class TagCluster:
    """A class to represent a cluster map for tags."""

    cluster_map: CLUSTER_MAP_TYPE

    def __init__(self, cluster_map: CLUSTER_MAP_TYPE):
        self.cluster_map = cluster_map

    @classmethod
    def train_from_embedding_model(
        cls,
        embedding_model_name: str,
        n_clusters: int = 32,
        n_init: int = 10,
        max_iter: int = 1000,
        trust_remote_code: bool = True,
        algorithm: CLUSTRING_ALGORITHM = "sklearn",
    ):
        """Train a cluster map from an embedding model."""

        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        model = AutoModel.from_pretrained(
            embedding_model_name, trust_remote_code=trust_remote_code
        )
        embeddings: np.ndarray = (
            model.get_input_embeddings().weight.detach().cpu().numpy()
        )

        if algorithm == "sklearn":
            kmeans = SklearnKMeans(
                n_clusters=n_clusters, n_init=n_init, max_iter=max_iter
            )
            kmeans.fit(embeddings)
            labels = kmeans.labels_
        elif algorithm == "faiss":
            kmeans = FaissKmeans(
                d=embeddings.shape[1],
                k=n_clusters,
                niter=max_iter,
                nredo=n_init,
            )
            kmeans.train(embeddings)
            _distance, ids = kmeans.index.search(embeddings, 1)
            labels = ids.flatten()
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

        cluster_map = {
            label: int(labels[token_id])
            for label, token_id in tokenizer.get_vocab().items()
        }

        return cls(cluster_map)

    @classmethod
    def from_pretrained(cls, filename: str):
        """Load the cluster map from a file."""
        cluster_map = {}
        with open(filename, "r", encoding="utf-8") as f:
            cluster_map = json.load(f)

        return cls(cluster_map)

    def save_pretrained(self, filename: str):
        """Save the cluster map to a file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.cluster_map, f, ensure_ascii=False, indent=2)

    def __getitem__(self, key: str) -> int:
        if key in self.cluster_map:
            return self.cluster_map[key]
        else:
            raise KeyError(f"Tag {key} not found in cluster map.")
