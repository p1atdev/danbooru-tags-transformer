import re
import numpy as np
import sys
import os

TAG_PATHS = {
    "AMBIGUOUS": "./tags/ambiguous.txt",
    "ANGLE": "./tags/angle.txt",
    "CENSORSHIP": "./tags/censorship.txt",
    "COMPOSITION": "./tags/composition.txt",
    "ERROR": "./tags/error.txt",
    "FOCUS": "./tags/focus.txt",
    "FORMAT": "./tags/format.txt",
    "FRAMING": "./tags/framing.txt",
    "GERUNDS": "./tags/gerunds.txt",
    "NON_VISUAL": "./tags/non_visual.txt",
    "PATTERN": "./tags/pattern.txt",
    "PEOPLE": "./tags/people.txt",
    "SIGNATURE": "./tags/signature.txt",
    "STYLE": "./tags/style.txt",
    "SUBJECT": "./tags/subject.txt",
    "TECHNIQUES": "./tags/techniques.txt",
    "VERBS": "./tags/verbs.txt",
}

MEDIUM_TAG_PATTERN = re.compile(r".* \(medium\)")  # 50%
STYLE_TAG_PATTERN = re.compile(r".* \(style\)")  # 50%
COSPLAY_TAG_PATTERN = re.compile(r".* \(cosplay\)")  # should be kept
MEME_TAG_PATTERN = re.compile(r".* \(meme\)")  # should be removed
OTHER_SPECIFIED_TAG_PATTERN = re.compile(r".* \(.*\)")
BACKGROUND_TAG_PATTERN = re.compile(r".* background")
LAYER_TAG_PATTERN = re.compile(r".* layer")

DROPOUT_RATES = {
    "AMBIGUOUS": 0.9,
    "ANGLE": 0.9,
    "CENSORSHIP": 0.0,
    "COMPOSITION": 0.25,  # from side
    # "ERROR": 0.0,
    "FOCUS": 0.5,  # solo focus
    "FORMAT": 0.25,  # novel cover
    "FRAMING": 0.9,  # upper body
    "GERUNDS": 0.9,  # swimming
    # "NON_VISUAL": 0.0,
    "PATTERN": 0.1,  # shima (pattern)
    "PEOPLE": 0.0,  # 2girls
    # "SIGNATURE": 0.0,
    "STYLE": 0.5,  # abstract, realistic
    "SUBJECT": 0.0,  # scenery, no humans, still life
    "TECHNIQUES": 0.9,  # depth of field
    "VERBS": 0.75,  # aiming, jumping
    "MEDIUM_TAG": 0.5,
    "STYLE_TAG": 0.5,
    "COSPLAY_TAG": 0.9,
    "MEME_TAG": 0.99,
    "OTHER_SPECIFIED_TAG": 0.9,
    "BACKGROUND_TAG": 0.75,
    "LAYER_TAG": 0.75,
}

REMOVE_TAGS = [
    "ERROR",
    "NON_VISUAL",
    "SIGNATURE",
]

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


# タグリストから読み込んで返す
def load_tags(path: str) -> list[str]:
    tags = []
    with open(os.path.join(SCRIPT_PATH, path), "r") as f:
        for line in f:
            tags.append(line.strip().lower())
    return tags


class TagManger:
    tags: dict[str, list[str]] = {}

    def __init__(self) -> None:
        for name, path in TAG_PATHS.items():
            self.tags[name] = load_tags(path)

    # タグが学習せずに取り除くべきかどうか
    def should_remove(self, tag: str) -> bool:
        for tag_category in REMOVE_TAGS:
            if tag in self.tags[tag_category]:
                return True
        return False

    def is_valid_tag(self, tag: str) -> bool:
        for tags in self.tags.values():
            if tag in tags:
                return True
        return False
