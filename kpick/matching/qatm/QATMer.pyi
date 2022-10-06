import torch
from _typeshed import Incomplete
from imutils import rotate_bound as rotate_bound
from ketisdk.utils.proc_utils import Timer as Timer
from kpick.base.base import BasDetectObj as BasDetectObj
from kpick.matching.qatm.qatm import CreateModel as CreateModel, nms_multi as nms_multi, run_one_sample as run_one_sample
from pathlib import Path as Path
from torch import nn as nn
from torchvision import utils as utils

def get_qatm_default_config(): ...
def place_on_background(im, bg): ...

class TemplateDataset(torch.utils.data.Dataset):
    transform: Incomplete
    filenames: Incomplete
    thresholds: Incomplete
    images: Incomplete
    templates: Incomplete
    num_tmp: Incomplete
    colors: Incomplete
    def __init__(self, tmp_dir, transform) -> None: ...
    def __getitem__(self, idx): ...

class QATMER:
    model: Incomplete
    transform: Incomplete
    tmpDataset: Incomplete
    hard_thresh: Incomplete
    def get_model(self, tmp_dir, data_mean, data_std, alpha: int = ..., hard_thresh: float = ..., use_cuda: bool = ...) -> None: ...
    def get_scores(self, im, tmp_name: Incomplete | None = ...): ...
    def get_boxes(self, im, tmp_name: Incomplete | None = ...): ...

def get_qatm_obj(Detector=...): ...
def get_qatm_gui(Detectorobj=...): ...
def test_qatmer() -> None: ...
def demo_qatm_gui(title: str = ..., Detector=..., cfg_path: str = ..., default_cfg_path: str = ..., sensors=..., key_args=..., data_root: str = ..., rgb_formats=..., depth_formats: Incomplete | None = ...) -> None: ...
