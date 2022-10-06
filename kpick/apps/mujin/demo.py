import cv2
from kpick.apps.mujin.detector import MujinDetector, demo_mujin_gui
import kpick
import os
from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

KPICK_DIR = os.path.split(kpick.__file__)[0]

# paths
rgb_path = os.path.join(KPICK_DIR, 'apps/mujin/test_images/2_rgb.png')
depth_path = os.path.join(KPICK_DIR, 'apps/mujin/test_images/2_depth.png')
cfg_path = os.path.join(KPICK_DIR,'apps/mujin/config/suction.net')

# configs
depth_min = 700
depth_max =900
workspace = [(300,80), (1000, 80), (1000, 530), (300, 530)]

# load image
rgb = cv2.imread(rgb_path)[:,:,::-1]
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
rgbd = RGBD(rgb=rgb, depth=depth, depth_min=depth_min, depth_max=depth_max)
rgbd.set_workspace(pts=workspace)

# load model
detector = MujinDetector(cfg_path=cfg_path)

# predict
ret = detector.detect_and_show_poses(rgbd=rgbd, remove_bg=False)

# show
cv2.imshow('reviewer', ret['im'][:,:,::-1])
cv2.waitKey()
