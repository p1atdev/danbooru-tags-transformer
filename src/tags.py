# common special tokens
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
SEP_TOKEN = "<|sep|>"
MASK_TOKEN = "<|mask|>"

# group
# RATING_START = "<rating>"
# RATING_END = "</rating>"
GENERAL_START = "<general>"
GENERAL_END = "</general>"
CHARACTER_START = "<character>"
CHARACTER_END = "</character>"
COPYRIGHT_START = "<copyright>"
COPYRIGHT_END = "</copyright>"

# rating
RATING_SFW = "<|rating:sfw|>"
RATING_NSFW = "<|rating:nsfw|>"
RATING_GENERAL = "<|rating:general|>"
RATING_SENSITIVE = "<|rating:sensitive|>"
RATING_QUESTIONABLE = "<|rating:questionable|>"
RATING_EXPLICIT = "<|rating:explicit|>"

# quality
QUALITY_BEST = "<|quality:best|>"
QUALITY_HIGH = "<|quality:high|>"
QUALITY_NORMAL = "<|quality:normal|>"
QUALITY_LOW = "<|quality:low|>"
QUALITY_WORST = "<|quality:worst|>"

# total tags legnth
LENGTH_VERY_SHORT = "<|legnth:very_short|>"
LENGTH_SHORT = "<|legnth:short|>"
LENGTH_MEDIUM = "<|legnth:medium|>"
LENGTH_LONG = "<|legnth:long|>"
LENGTH_VERY_LONG = "<|legnth:very_long|>"

# aspect ratio, inspired by DanTagGen
ASPECT_RATIO_ULTRA_WIDE = "<|aspect_ratio:ultra_wide|>"
ASPECT_RATIO_WIDE = "<|aspect_ratio:wide|>"
ASPECT_RATIO_SQUARE = "<|aspect_ratio:square|>"
ASPECT_RATIO_TALL = "<|aspect_ratio:tall|>"
ASPECT_RATIO_ULTRA_TALL = "<|aspect_ratio:ultra_tall|>"

# for SFT
INPUT_START = "<|input_start|>"
INPUT_END = "<|input_end|>"

# identity level
IDENTITY_LEVEL_NONE = "<|identity:none|>"
IDENTITY_LEVEL_LAX = "<|identity:lax|>"
IDENTITY_LEVEL_STRICT = "<|identity:strict|>"


# reserved
RESERVED_TOKENS = [f"<|reserved_{i}|>" for i in range(64)]
