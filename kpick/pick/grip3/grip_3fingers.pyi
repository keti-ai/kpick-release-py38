from _typeshed import Incomplete
from kpick.base.base import DetGuiObj as DetGuiObj

class ThreeFingerGraspGui(DetGuiObj):
    def get_model(self) -> None: ...
    def get_grasp_pose(self, rgbd, disp_mode: str = ...): ...
    def gui_process_single(self, rgbd, method_ind: int = ..., filename: str = ..., disp_mode: str = ..., detected: Incomplete | None = ...): ...