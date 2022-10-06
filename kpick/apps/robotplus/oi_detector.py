import os.path

from kpick.pick.grip.grip_detection import GripDetector, demo_grip_gui, demo_grip_single, get_grip_detector_obj
class OIDetector(GripDetector):
    pass

def create_and_load_oi_detector(cfg_path=None):
    from kpick.pick.grip.grip_detection import get_grip_detector_obj
    if cfg_path is None:
        import kpick
        KPICK_DIR = os.path.split(kpick.__file__)[0]
        cfg_path = os.path.join(KPICK_DIR, 'apps/robotplus/configs/oi_grip.cfg')
        if not os.path.exists(cfg_path):
            print(f'{cfg_path} does not exist ...')

    return get_grip_detector_obj(Detector=OIDetector)(cfg_path=cfg_path)


def demo_detect_oi_single():
    import cv2
    import kpick
    import os
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    KPICK_DIR = os.path.split(kpick.__file__)[0]

    # configs
    depth_min = 400
    depth_max = 600
    workspace = [(344, 175), (942, 176), (940, 541), (343, 541)]

    # load model
    detector = create_and_load_oi_detector()

    # load image
    rgb = cv2.imread(os.path.join(KPICK_DIR, 'apps/robotplus/test_images/oi01_rgb.png'))[:, :, ::-1]
    depth = cv2.imread(os.path.join(KPICK_DIR, 'apps/robotplus/test_images/oi01_depth.png'), cv2.IMREAD_UNCHANGED)
    rgbd = RGBD(rgb=rgb, depth=depth, depth_min=depth_min, depth_max=depth_max,
                denoise_ksize=detector.args.sensor.denoise_ksize)
    rgbd.set_workspace(pts=workspace)

    # predict
    detector.args.flag.show_steps = True
    ret = detector.detect_and_show_grips(rgbd=rgbd, net_args=detector.args.grip_net, args=detector.args)

    # show
    cv2.imshow('reviewer', ret['im'][:, :, ::-1])
    cv2.waitKey()

if __name__=='__main__':
    demo_grip_gui(title='OI Detector GUI', Detector=OIDetector,
                  cfg_path='kpick/apps/robotplus/configs/oi_grip.cfg',
                  default_cfg_path='kpick/apps/robotplus/configs/default.cfg',
                  data_root='data/apps/robotplus/OI_object/220810',
                  rgb_formats=['*rgb*'], depth_formats=['*depth*'])

    # demo_detect_oi_single()




