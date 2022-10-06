# How to use
## 1. Copper Detector
### 1.1. Create and Load Detector - Using the default Kpick's detector
```sh
from kpick.apps.robotplus.copper_detector import create_and_load_copper_detector
detector = create_and_load_copper_detector(cfg_path=cfg_path)
```
** cfg_path is default **kpick/apps/robotplus/configs/copper_grip.cfg** when cfg_path  is None or not given

### 1.2. Modify and Load Detector -  Extending the default Kpick's detector
```sh
from kpick.apps.robotplus.copper_detector import  CopperDetector, create_and_load_copper_detector
class AppDector(CopperDetector):
    def new_function(self):
        print('new function')

detector = create_and_load_copper_detector(Detector=AppDector,cfg_path=cfg_path)
```


### 1.3. Demo on single RGB-D image
```sh
def demo_detect_copper_single():
    import cv2
    import kpick
    import os
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    KPICK_DIR = os.path.split(kpick.__file__)[0]

    # configs
    depth_min = 900
    depth_max = 1000
    workspace = [(530, 278), (1252, 286), (1248, 714), (536, 704)]

    # load image
    rgb = cv2.imread(os.path.join(KPICK_DIR, 'apps/robotplus/test_images/copper01_rgb.png'))[:, :, ::-1]
    depth = cv2.imread(os.path.join(KPICK_DIR, 'apps/robotplus/test_images/copper01_depth.png'), cv2.IMREAD_UNCHANGED)
    rgbd = RGBD(rgb=rgb, depth=depth, depth_min=depth_min, depth_max=depth_max)
    rgbd.set_workspace(pts=workspace)

    # load model
    detector = create_and_load_copper_detector()

    # predict
    detector.args.flag.show_steps = True
    ret = detector.detect_and_show_grips(rgbd=rgbd, net_args=detector.args.grip_net, args=detector.args)

    # show
    cv2.imshow('reviewer', ret['im'][:, :, ::-1])
    cv2.waitKey()
```
