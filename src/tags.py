class CommonSpecialTokens:
    # common special tokens
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    PAD_TOKEN = "<|pad|>"
    UNK_TOKEN = "<|unk|>"
    SEP_TOKEN = "<|sep|>"
    MASK_TOKEN = "<|mask|>"


# category group
class TagCategoryTokens:
    RATING_START = "<rating>"
    RATING_END = "</rating>"
    GENERAL_START = "<general>"
    GENERAL_END = "</general>"
    CHARACTER_START = "<character>"
    CHARACTER_END = "</character>"
    COPYRIGHT_START = "<copyright>"
    COPYRIGHT_END = "</copyright>"


# rating
class RatingTokens:
    RATING_SFW = "<|rating:sfw|>"
    RATING_NSFW = "<|rating:nsfw|>"

    RATING_GENERAL = "<|rating:general|>"
    RATING_SENSITIVE = "<|rating:sensitive|>"
    RATING_QUESTIONABLE = "<|rating:questionable|>"
    RATING_EXPLICIT = "<|rating:explicit|>"


# quality
class QualityTokens:
    QUALITY_BEST = "<|quality:best|>"
    QUALITY_HIGH = "<|quality:high|>"
    QUALITY_NORMAL = "<|quality:normal|>"
    QUALITY_LOW = "<|quality:low|>"
    QUALITY_WORST = "<|quality:worst|>"


# total tags length
class LengthTokens:
    LENGTH_VERY_SHORT = "<|length:very_short|>"
    LENGTH_SHORT = "<|length:short|>"
    LENGTH_MEDIUM = "<|length:medium|>"
    LENGTH_LONG = "<|length:long|>"
    LENGTH_VERY_LONG = "<|length:very_long|>"


# aspect ratio, inspired by DanTagGen
class AspectRatioTokens:
    ASPECT_RATIO_ULTRA_WIDE = "<|aspect_ratio:ultra_wide|>"
    ASPECT_RATIO_WIDE = "<|aspect_ratio:wide|>"
    ASPECT_RATIO_SQUARE = "<|aspect_ratio:square|>"
    ASPECT_RATIO_TALL = "<|aspect_ratio:tall|>"
    ASPECT_RATIO_ULTRA_TALL = "<|aspect_ratio:ultra_tall|>"


# for SFT
class InstructionTokens:
    INPUT_START = "<|input_start|>"
    INPUT_END = "<|input_end|>"

    GROUP_START = "<group>"
    GROUP_END = "</group>"
    EXAMPLE_START = "<example>"
    EXAMPLE_END = "</example>"

    BAN_START = "<ban>"
    BAN_END = "</ban>"
    USE_START = "<use>"
    USE_END = "</use>"


# identity level
class IdentityTokens:
    FLAG_KEEP_IDENTITY = "<|keep_identity|>"
    IDENTITY_LEVEL_NONE = "<|identity:none|>"
    IDENTITY_LEVEL_LAX = "<|identity:lax|>"
    IDENTITY_LEVEL_STRICT = "<|identity:strict|>"


# multi modal inputs
class MultiModalTokens:
    IMAGE_START = "<image>"
    IMAGE_END = "</image>"
    LINEART_START = "<lineart>"
    LINEART_END = "</lineart>"
    TAGGER_START = "<tagger>"
    TAGGER_END = "</tagger>"
    PROJECTION_START = "<projection>"
    PROJECTION_END = "</projection>"
    DESCRIBE_START = "<describe>"
    DESCRIBE_END = "</describe>"


# reserved
RESERVED_TOKENS = [f"<|reserved_{i}|>" for i in range(64)]


class PresetTags:
    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            tags = f.read().splitlines()

        self.tags = [tag.strip() for tag in tags if tag.strip() != ""]

    @classmethod
    def artistic_error(cls) -> "PresetTags":
        return cls("tags/artistic_error.txt")

    @classmethod
    def background(cls) -> "PresetTags":
        return cls("tags/background.txt")

    @classmethod
    def ban_meta(cls) -> "PresetTags":
        return cls("tags/ban_meta.txt")

    @classmethod
    def color_theme(cls) -> "PresetTags":
        return cls("tags/color_theme.txt")

    @classmethod
    def displeasing_meta(cls) -> "PresetTags":
        return cls("tags/displeasing_meta.txt")

    @classmethod
    def focus(cls) -> "PresetTags":
        return cls("tags/focus.txt")

    @classmethod
    def people(cls) -> "PresetTags":
        return cls("tags/people.txt")

    @classmethod
    def usable_meta(cls) -> "PresetTags":
        return cls("tags/usable_meta.txt")

    @classmethod
    def watermark(cls) -> "PresetTags":
        return cls("tags/watermark.txt")
