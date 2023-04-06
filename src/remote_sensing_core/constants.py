from enum import Enum


class Bands(Enum):
    RGB = "RGB"
    INFRARED = "INFRARED"
    ALL = "ALL"


# Images
S1_IMG_KEY = "s1_img"
S2_IMG_KEY = "s2_img"
WORLD_COVER_IMG_KEY = "world_cover_img"
ALTITUDE_IMG_KEY = "altitude_img"
STACKED_IMAGE_KEY = "stacked_img"
# Labels
MULTICLASS_LABEL_KEY = "multiclass_label"

# File data type
NUMPY_DTYPE = "float32"
