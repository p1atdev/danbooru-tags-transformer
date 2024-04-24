import sys

sys.path.append(".")

from src.group import TagGroup
from src.cluster import TagCluster
from src.organizer import GroupTagOrganizer, ClusterTagOrganizer


def test_group_tag_organizer():
    group = TagGroup()
    organizer = GroupTagOrganizer(group)

    tags = [
        "1girl",  # people tag
        "solo",  # focus tag
        "solo focus",  # focus tag
        "blonde hair",  # other tag
        "bad anatomy",  # artistic error tag
        "watermark",  # watermark tag
    ]

    result = organizer.organize_tags(tags)

    assert "1girl" in result.people_tags
    assert "solo" in result.focus_tags
    assert "solo focus" in result.focus_tags
    assert "blonde hair" in result.other_tags[0]
    assert "bad anatomy" in result.artistic_error_tags
    assert "watermark" in result.watermark_tags


def test_cluster_tag_organizer():
    group = TagGroup()
    cluster = TagCluster.from_pretrained("data/cluster_map.json")
    organizer = ClusterTagOrganizer(group, cluster)

    tags = [
        "1girl",  # people tag
        "solo",  # focus tag
        "solo focus",  # focus tag
        "blonde hair",  # other tag
        "bad anatomy",  # artistic error tag
        "watermark",  # watermark tag
    ]

    result = organizer.organize_tags(tags)

    assert "1girl" in result.people_tags
    assert "solo" in result.focus_tags
    assert "solo focus" in result.focus_tags
    assert "blonde hair" in result.other_tags[0]
    assert "bad anatomy" in result.artistic_error_tags
    assert "watermark" in result.watermark_tags
