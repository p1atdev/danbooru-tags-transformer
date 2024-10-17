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
    META_START = "<meta>"
    META_END = "</meta>"


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
    LENGTH_TOO_SHORT = "<|length:too_short|>"
    LENGTH_VERY_SHORT = "<|length:very_short|>"
    LENGTH_SHORT = "<|length:short|>"
    LENGTH_MEDIUM = "<|length:medium|>"
    LENGTH_LONG = "<|length:long|>"
    LENGTH_VERY_LONG = "<|length:very_long|>"
    LENGTH_TOO_LONG = "<|length:too_long|>"


# aspect ratio, inspired by DanTagGen
class AspectRatioTokens:
    ASPECT_RATIO_TOO_TALL = "<|aspect_ratio:too_tall|>"
    ASPECT_RATIO_TALL_WALLPAPER = "<|aspect_ratio:tall_wallpaper|>"
    ASPECT_RATIO_TALL = "<|aspect_ratio:tall|>"
    ASPECT_RATIO_SQUARE = "<|aspect_ratio:square|>"
    ASPECT_RATIO_WIDE = "<|aspect_ratio:wide|>"
    ASPECT_RATIO_WIDE_WALLPAPER = "<|aspect_ratio:wide_wallpaper|>"
    ASPECT_RATIO_TOO_WIDE = "<|aspect_ratio:too_wide|>"


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
    IMAGE_PLACEHOLDER = "<|image|>"
    IMAGE_END = "</image>"
    LINEART_START = "<lineart>"
    LINEART_PLACEHOLDER = "<|lineart|>"
    LINEART_END = "</lineart>"
    NATURAL_START = "<natural>"
    NATURAL_PLACEHOLDER = "<|natural|>"
    NATURAL_END = "</natural>"
    TAGGER_START = "<tagger>"
    TAGGER_PLACEHOLDER = "<|tagger|>"
    TAGGER_END = "</tagger>"
    PROJECTION_START = "<projection>"
    PROJECTION_PLACEHOLDER = "<|projection|>"
    PROJECTION_END = "</projection>"
    DESCRIBE_START = "<describe>"
    DESCRIBE_PLACEHOLDER = "<|describe|>"
    DESCRIBE_END = "</describe>"


# translation level
class TranslationTokens:
    TRANSLATE_EXACT = "<|translate:exact|>"
    TRANSLATE_APPROX = "<|translate:approx|>"
    TRANSLATE_CREATIVE = "<|translate:creative|>"


# reserved
RESERVED_TOKENS = [f"<|reserved_{i}|>" for i in range(64)]
