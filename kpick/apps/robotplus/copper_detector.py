import os.path

from kpick.pick.grip.grip_detection import GripDetector
class CopperDetector(GripDetector):
    pass

def create_and_load_copper_detector(Detector= CopperDetector,cfg_path=None):
    from kpick.pick.grip.grip_detection import get_grip_detector_obj
    if cfg_path is None:
        import kpick
        KPICK_DIR = os.path.split(kpick.__file__)[0]
        cfg_path = os.path.join(KPICK_DIR, 'apps/robotplus/configs/copper_grip.cfg')
        if os.path.exists(cfg_path):
            print(f'{cfg_path} does not exist ...')

    return get_grip_detector_obj(Detector)(cfg_path=cfg_path)

def demo_detect_copper_gui(cfg_path=None, sensor_types=['realsense', 'zivid']):
    from kpick.pick.grip.grip_detection import demo_grip_gui, get_grip_detector_obj, get_suction_detector_gui

    demo_grip_gui(title='Copper Grasp Demo', Detector=CopperDetector,
                  cfg_path='kpick/apps/robotplus/configs/copper_grip.cfg',
                  default_cfg_path='kpick/apps/robotplus/configs/default.cfg')

def demo_modify_and_load_copper_detector():
    class AppDector(CopperDetector):
        def new_function(self):
            print('new function')

    detector = create_and_load_copper_detector(Detector=AppDector,cfg_path=cfg_path)



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


if __name__=='__main__':
    # demo_detect_copper_gui()
    demo_detect_copper_single()
    # create_and_load_copper_detector()


