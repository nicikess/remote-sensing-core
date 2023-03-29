from enum import Enum


class Bands(Enum):
    RGB = "RGB"
    INFRARED = "infrared"
    ALL = "all"


S1_IMG_KEY = "s1_img"
S2_IMG_KEY = "s2_img"
LABEL_KEY = "label"
NUMPY_DTYPE = "float32"
