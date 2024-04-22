import json

from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel


CLUSTER_MAP_TYPE = dict[str, int]


class TagCluster:
    """A class to represent a cluster map for tags."""

    cluster_map: CLUSTER_MAP_TYPE

    def __init__(self, cluster_map: CLUSTER_MAP_TYPE):
        self.cluster_map = cluster_map

    @classmethod
    def train_from_embedding_model(
        cls,
        embedding_model_name: str,
        model_type: str = "opt",
        n_clusters: int = 32,
        n_init: int = 10,
        max_iter: int = 1000,
    ):
        """Train a cluster map from an embedding model."""

        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        model = AutoModel.from_pretrained(embedding_model_name)
        if model_type == "opt":
            embeddings = model.decoder.embed_tokens.weight.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented.")

        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
        kmeans.fit(embeddings)

        cluster_map = {
            label: int(kmeans.labels_[token_id])
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
