![license](https://img.shields.io/badge/license-MIT-green) ![PyTorch-1.4.0](https://img.shields.io/badge/PyTorch-1.4.0-blue)
# KPICK -  AI-BASED PICKING PACKAGE

## System Requirements and prerequiste
```sh
- Ubuntu 16.04 or 18.04
- CUDA >=10.0, CUDNN>=7
```

## Installation [Link](https://github.com/keti-ai/kpick-devel/wiki/Installation-Guide)

## HOW TO USE
```sh
def demo_grip_gui(cfg_path):
    from kpick.pick.grip.grip_detection import GripGuiDetector
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.gui.gui import GUI, GuiModule

    detect_module0 = GuiModule(GripGuiDetector, type='grip_detector', name='Grip Detector',
                              category='detector', cfg_path=cfg_path, num_method=5)

    GUI(title='Grip Detection GUI',
           modules=[detect_module0] + get_realsense_modules(),
           )

```
```sh

def demo_grip_without_gui(cfg_path, rgb_path, depth_path, ws_pts=None):
    from kpick.pick.grip.grip_detection import GripDetector
    import cv2
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    # Set viewer
    cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('viewer', 1080, 720)

    # Load image
    rgb = cv2.imread(rgb_path)[:, :, ::-1]
    depth = cv2.imread(depth_path)[:, :, 0]
    rgbd = RGBD(rgb=rgb, depth=depth)

    # workspace_pts = [(550, 100), (840, 100), (840, 600), (550, 600)]
    if ws_pts is not None: rgbd.set_workspace(pts=ws_pts)

    # Initialize network
    detector = GripDetector(cfg_path=cfg_path)

    # Run
    ret = detector.detect_and_show_poses(rgbd=rgbd)

    # Show
    cv2.imshow('viewer', ret['im'][:,:,::-1])
    if cv2.waitKey()==27: exit()

```
```sh
def demo_suction_gui(cfg_path):
    from kpick.pick.suction.suction_detection import SuctionGuiDetector
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.gui.gui import GUI, GuiModule

    detect_module = GuiModule(SuctionGuiDetector, type='suction_detector', name='Suction Detector',
                              category='detector', cfg_path=cfg_path, num_method=6)

    GUI(title='Grip Detection GUI',
        modules=[detect_module,] + get_realsense_modules(),
        )
```
```sh
def demo_suction_without_gui(cfg_path, rgb_path, depth_path, ws_pts=None):
    from kpick.pick.suction.suction_detection import SuctionDetector
    import cv2
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    # Set viewer
    cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('viewer', 1080, 720)

    # Load image
    rgb = cv2.imread(rgb_path)[:, :, ::-1]
    depth = cv2.imread(depth_path)[:, :, 0]
    rgbd = RGBD(rgb=rgb, depth=depth)

    # Set workspace
    # workspace_pts = [(550, 100), (840, 100), (840, 600), (550, 600)]
    if ws_pts is not None: rgbd.set_workspace(pts=ws_pts)

    # Initialize network
    detector = SuctionDetector(cfg_path=cfg_path)

    # Run
    ret = detector.detect_and_show_poses(rgbd=rgbd)

    # Show
    cv2.imshow('viewer', ret['im'][:,:,::-1])
    if cv2.waitKey()==27: exit()
```



