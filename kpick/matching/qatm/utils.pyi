from _typeshed import Incomplete
from sklearn.metrics import auc as auc

int_: Incomplete

def IoU(r1, r2): ...
def evaluate_iou(rect_gt, rect_pred): ...
def compute_score(x, w, h): ...
def locate_bbox(a, w, h): ...
def score2curve(score, thres_delta: float = ...): ...
def all_sample_iou(score_list, gt_list): ...
def plot_success_curve(iou_score, title: str = ...) -> None: ...
