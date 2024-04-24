from abc import ABC, abstractmethod
from dataclasses import dataclass

from .group import TagGroup
from .cluster import TagCluster


@dataclass
class TagOrganizerResult:
    """A class to represent the result of organizing tags."""

    people_tags: list[str]
    focus_tags: list[str]
    watermark_tags: list[str]
    artistic_error_tags: list[str]

    # by cluster
    other_tags: list[list[str]]


class TagOrganizer(ABC):
    """An abstract class to organize tags."""

    @abstractmethod
    def organize_tags(self, tags: list[str]) -> TagOrganizerResult:
        """Organize tags by group and cluster."""
        raise NotImplementedError


class GroupTagOrganizer(TagOrganizer):
    """A class to organize tags by group."""

    group: TagGroup

    def __init__(self, group: TagGroup):
        self.group = group

    def organize_tags(self, tags: list[str]) -> TagOrganizerResult:
        """Organize tags by group."""

        people_tags: list[str] = []
        focus_tags: list[str] = []
        watermark_tags: list[str] = []
        artistic_error_tags: list[str] = []

        other_tags: list[str] = []

        for tag in tags:
            if tag in self.group.people_tags:
                people_tags.append(tag)
            elif tag in self.group.focus_tags:
                focus_tags.append(tag)
            elif tag in self.group.watermark_tags:
                watermark_tags.append(tag)
            elif tag in self.group.artistic_error_tags:
                artistic_error_tags.append(tag)
            else:
                other_tags.append(tag)

        return TagOrganizerResult(
            people_tags=people_tags,
            focus_tags=focus_tags,
            watermark_tags=watermark_tags,
            artistic_error_tags=artistic_error_tags,
            other_tags=[other_tags],
        )


class ClusterTagOrganizer(TagOrganizer):
    """A class to organize tags by group and cluster."""

    group: TagGroup
    cluster: TagCluster

    def __init__(self, group: TagGroup, cluster: TagCluster):
        self.group = group
        self.cluster = cluster

    def organize_tags(self, tags: list[str]) -> TagOrganizerResult:
        """Organize tags by group and cluster."""
        people_tags: list[str] = []
        focus_tags: list[str] = []
        watermark_tags: list[str] = []
        artistic_error_tags: list[str] = []

        clusters: dict[int, list[str]] = {}

        for tag in tags:
            # 特殊なタグならそれで分別
            if tag in self.group.people_tags:
                people_tags.append(tag)
            elif tag in self.group.focus_tags:
                focus_tags.append(tag)
            elif tag in self.group.watermark_tags:
                watermark_tags.append(tag)
            elif tag in self.group.artistic_error_tags:
                artistic_error_tags.append(tag)

            else:
                # クラスタで分別
                cluster_id = self.cluster.cluster_map.get(tag, -1)
                if cluster_id != -1:
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []

                    clusters[cluster_id].append(tag)

        # クラスタごとにリスト化
        other_tags = list(clusters.values())

        return TagOrganizerResult(
            people_tags=people_tags,
            focus_tags=focus_tags,
            watermark_tags=watermark_tags,
            artistic_error_tags=artistic_error_tags,
            other_tags=other_tags,
        )
