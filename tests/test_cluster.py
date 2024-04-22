import sys

sys.path.append(".")

from src.cluster import TagCluster


def test_tag_cluster():
    cluster = TagCluster.train_from_embedding_model("p1atdev/dart2vec-opt_5")

    assert cluster.cluster_map is not None


def test_save_cluster_map():
    cluster = TagCluster.train_from_embedding_model(
        "p1atdev/dart2vec-opt_5", max_iter=100
    )

    cluster.save_pretrained("data/cluster_map.json")

    cluster2 = TagCluster.from_pretrained("data/cluster_map.json")

    assert cluster.cluster_map == cluster2.cluster_map


def test_get_cluster_by_tag():
    cluster = TagCluster.train_from_embedding_model(
        "p1atdev/dart2vec-opt_5", max_iter=100
    )

    assert cluster["1girl"] is not None
