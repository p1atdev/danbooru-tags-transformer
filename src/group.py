import os

from pathlib import Path


def load_tags(path: str | Path):
    """Load tags from a file."""
    with open(path, "r") as f:
        tags = f.read().splitlines()

    tags = [tag.strip() for tag in tags]

    return tags


class TagGroup:
    path: str | Path

    people_tags: list[str]
    watermark_tags: list[str]
    artistic_error_tags: list[str]
    focus_tags: list[str]

    def __init__(self) -> None:
        self.path = os.path.join(os.path.dirname(__file__), "..", "tags")

        self.people_tags = load_tags(os.path.join(self.path, "people.txt"))
        self.watermark_tags = load_tags(os.path.join(self.path, "watermark.txt"))
        self.artistic_error_tags = load_tags(
            os.path.join(self.path, "artistic_error.txt")
        )
        self.focus_tags = load_tags(os.path.join(self.path, "focus.txt"))
