from enum import Enum

class Bands(Enum):
    RGB = "RGB"
    INFRARED = "infrared"
    ALL = "all"

# Set remote file paths and directories for ben-ge (small)
class RemoteFilesAndDirectoryReferences(Enum):
    # Files
    ESA_WORLD_COVER_CSV_TRAIN = "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-train.csv"
    ESA_WORLD_COVER_CSV_VALIDATION = "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-validation.csv"
    ESA_WORLD_COVER_CSV_TEST = "/ds2/remote_sensing/ben-ge/ben-ge-s/data-index/ben-ge-s-test.csv"
    SENTINEL_1_2_METADATA_CSV = "/ds2/remote_sensing/ben-ge/ben-ge-s/ben-ge-s_sentinel12_meta.csv"

    # Directories
    SENTINEL_1_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-1/s1_npy/"
    SENTINEL_2_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge-s/sentinel-2/s2_npy/"
    ESA_WORLD_COVER_DIRECTORY = "/ds2/remote_sensing/ben-ge/ben-ge-s/esaworldcover/"
    ERA5_CSV = "/ds2/remote_sensing/ben-ge/ben-ge-s/ben-ge-s_era-5.csv"

S1_IMG_KEY = "s1_img"
S2_IMG_KEY = "s2_img"
WORLD_COVER_IMG_KEY = "world_cover_img"
LABEL_KEY = "label"
NUMPY_DTYPE = "float32"
