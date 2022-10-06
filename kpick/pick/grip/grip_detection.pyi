from ketisdk.import_basic_utils import *
from _typeshed import Incomplete
from imutils import rotate as rotate
from kpick.pick.base_grasp_detection import BaseGraspDetector as BaseGraspDetector
from kpick.pick.grip.grip_proposing import find_grip_candidates_from_edges as find_grip_candidates_from_edges

MODULE_DIR: Incomplete

def get_base_cfg(): ...

BASE_CFG: Incomplete

def grip2box(Pt0, Pt1, h): ...
def grasp2TiltedRect(xc, yc, w, h, angle): ...

class GripDetector(BaseGraspDetector):
    matcher: Incomplete
    def init(self, net_args, matching_args: Incomplete | None = ..., net_train_args: Incomplete | None = ..., train: bool = ...) -> None: ...
    def find_grip_candidates(self, rgbd, net_args, rot_grip: Incomplete | None = ...): ...
    def get_grip_in_workspace(self, rgbd, Grip, all_boxes, inds): ...
    def remove_grasp_on_background(self, rgbd, Grip, all_boxes, inds, net_args): ...
    gidx: Incomplete
    def detect_grips(self, rgbd, net_args, remove_bg: bool = ...): ...
    def scoreGripWidth(self, Width, net_args): ...
    def select_best_n_grips(self, Grip, net_args): ...
    def getShowMomentScore(self, rgbd, net_args, disp_mode: str = ...): ...
    def getShowSpatialScore(self, rgbd, net_args, disp_mode: str = ...): ...
    def getShowDepthScore(self, rgbd, net_args, disp_mode: str = ...): ...
    def getShowWidthScore(self, rgbd, net_args, disp_mode: str = ...): ...
    def getShowExpScore(self, rgbd, net_args, disp_mode: str = ...): ...
    def show_grips(self, rgbd, det, net_args, args, disp_mode: str = ..., out: Incomplete | None = ...): ...
    def detect_and_show_grips(self, rgbd, net_args, args, remove_bg: bool = ..., disp_mode: str = ..., detected: Incomplete | None = ...): ...
    def matching_and_scoring(self, rgbd, disp_mode: str = ..., detected: Incomplete | None = ...): ...

def get_grip_detector_obj(Detector=...): ...
def get_grip_detector_gui(DetectorObj=...): ...
def demo_grip_gui(title: str = ..., Detector=..., cfg_path: Incomplete | None = ..., default_cfg_path: str = ..., sensors=..., key_args=..., data_root: Incomplete | None = ..., rgb_formats: Incomplete | None = ..., depth_formats: Incomplete | None = ...) -> None: ...
def demo_grip_single(rgb_path, depth_path, Detectorobj=..., cfg_path: Incomplete | None = ..., depth_min: int = ..., depth_max: int = ..., ws_pts: Incomplete | None = ...) -> None: ...
