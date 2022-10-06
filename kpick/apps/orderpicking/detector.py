from kpick.pick.suction.suction_detection import SuctionDetector
from ketisdk.utils.proc_utils import Timer, ProcUtils, ArrayUtils
import logging
import os
import kpick

KPICK_MODULE_DIR = os.path.split(kpick.__file__)[0]


class OrderDetector(SuctionDetector):
    pass


def create_and_load_order_detector(cfg_path=None):
    from kpick.pick.suction.suction_detection import get_suction_detector_obj
    if cfg_path is None:
        import kpick
        KPICK_DIR = os.path.split(kpick.__file__)[0]
        cfg_path = os.path.join(KPICK_DIR, 'apps/orderpicking/configs/suction.cfg')
        if os.path.exists(cfg_path):
            print(f'{cfg_path} does not exist ...')
    Module = get_suction_detector_obj(Detector=OrderDetector)

    return Module(cfg_path=cfg_path)


def demo_orderpicking_gui(cfg_path=None, default_cfg_path=None, data_root=None, rgb_formats=None, depth_formats=None):
    from kpick.pick.suction.suction_detection import demo_suction_gui

    if cfg_path is None: cfg_path = os.path.join(KPICK_MODULE_DIR, 'apps/orderpicking/configs/suction.cfg')
    if not os.path.exists(cfg_path):
        print(f'{cfg_path} does not exist ...')

    if default_cfg_path is None: default_cfg_path = os.path.join(KPICK_MODULE_DIR,
                                                                 'apps/orderpicking/configs/default.cfg')
    if not os.path.exists(default_cfg_path):
        print(f'{default_cfg_path} does not exist ...')

    demo_suction_gui(cfg_path=cfg_path, Detector=OrderDetector, default_cfg_path=default_cfg_path,
                     data_root=data_root, rgb_formats=rgb_formats, depth_formats=depth_formats,
                     )


def demo_orderpicking_single():
    import cv2
    import kpick
    import os
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    KPICK_DIR = os.path.split(kpick.__file__)[0]

    # configs
    depth_min = 700
    depth_max = 900
    workspace = [(398, 172), (949, 174), (949, 580), (398, 572)]

    # load image
    rgb = cv2.imread(os.path.join(KPICK_DIR, 'apps/orderpicking/test_images/1_rgb.png'))[:, :, ::-1]
    depth = cv2.imread(os.path.join(KPICK_DIR, 'apps/orderpicking/test_images/1_depth.png'), cv2.IMREAD_UNCHANGED)
    rgbd = RGBD(rgb=rgb, depth=depth, depth_min=depth_min, depth_max=depth_max)
    rgbd.set_workspace(pts=workspace)

    # load model
    detector = create_and_load_order_detector()

    # predict
    detector.args.flag.show_steps = False
    ret = detector.detect_and_show_suctions(rgbd=rgbd, net_args=detector.args.suction_net, rpn_args=detector.args.rpn,
                                            args=detector.args, remove_bg=False)

    # show
    cv2.imshow('reviewer', ret['im'][:, :, ::-1])
    cv2.waitKey()


if __name__ == '__main__':
    demo_orderpicking_gui(data_root='kpick/apps/orderpicking/test_images',
                          rgb_formats=['*rgb*'], depth_formats=['*depth*', ])
    # demo_orderpicking_single()
