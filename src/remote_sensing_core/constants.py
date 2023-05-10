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
MULTICLASS_NUMERIC_LABEL_KEY = "multiclass_numeric_label"
MULTICLASS_ONE_HOT_LABEL_KEY = "multiclass_one_hot_label"
ELEVATION_DIFFERENCE_LABEL_KEY = "elevation_difference_label"

# File data type
NUMPY_DTYPE = "float32"

# Temperature sentinel 2 index
TEMPERATURE_S2_INDEX = 3

# Season S2 index
SEASON_S2_INDEX = 0

# Climate zone index
CLIMATE_ZONE_INDEX = 1