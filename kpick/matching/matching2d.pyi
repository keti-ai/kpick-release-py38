from _typeshed import Incomplete
from skimage import data as data
from skimage.feature import register_translation as register_translation
from skimage.transform import rotate as rotate
from skimage.util import img_as_float as img_as_float

class Matching2d:
    def __init__(self, gt_dir: Incomplete | None = ..., include_bg: bool = ...) -> None: ...
    def find_largest_mask_center(self, masks): ...
    def match(self, masks, target_rgb_img, downsample: bool = ...): ...
