from kpick.pick.grip.grip_detection import GripDetector, demo_grip_gui, get_grip_detector_obj
from kpick.pick.suction.suction_detection import SuctionDetector, demo_suction_gui, get_suction_detector_gui, \
    get_suction_detector_obj
from kpick.pick.multimode_grasp import MultiModeGraspDetector, get_multimode_grasp_detector_obj, \
    demo_multimode_grasp_gui, get_multimode_grasp_detector_gui


class OnePointDualGraspDetector(MultiModeGraspDetector):
    pass


class TwoPointsDualGraspDetector(MultiModeGraspDetector):
    pass


class RandomGripDetector(GripDetector):
    pass


class RandomSuctionDetector(SuctionDetector):
    pass


def create_and_load_random_picking_detector(Detector=None, cfg_path=None, detector_type='one_point_dual'):
    import os, kpick
    KPICK_DIR = os.path.split(kpick.__file__)[0]

    detector_types = ['grip', 'suction', 'one_point_dual', 'two_point_dual']
    funcs = [get_grip_detector_obj, get_suction_detector_obj, get_multimode_grasp_detector_obj, get_multimode_grasp_detector_obj]
    Detectors = [GripDetector, SuctionDetector, OnePointDualGraspDetector, TwoPointsDualGraspDetector]
    cfg_dir = os.path.join(KPICK_DIR, 'apps/randompicking/configs')
    cfg_paths = [os.path.join(cfg_dir, 'grip.cfg'), os.path.join(cfg_dir, 'suction.cfg'),
                 os.path.join(cfg_dir, 'one_point_dual.cfg'), os.path.join(cfg_dir, 'two_points_dual.cfg')]

    idx = detector_types.index(detector_type)
    cfg_path = cfg_paths[idx] if cfg_path is None else cfg_path
    Detector = Detectors[idx] if Detector is None else Detector

    return funcs[idx](Detector=Detector)(cfg_path=cfg_path)


def demo_random_picking_simple():
    import cv2
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD
    import os
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD
    import kpick
    KPICK_DIR = os.path.split(kpick.__file__)[0]

    # load model
    detector = create_and_load_random_picking_detector()

    # load image
    rgb = cv2.imread('data/test_images/multimode_grasp_rgb.png')[:, :, ::-1]
    depth = cv2.imread('data/test_images/multimode_grasp_depth.png', cv2.IMREAD_UNCHANGED)
    rgbd = RGBD(rgb=rgb, depth=depth, depth_min=600, depth_max=800,
                denoise_ksize=detector.args.sensor.denoise_ksize)
    rgbd.set_workspace(pts=[(360, 270), (889, 273), (890, 635), (359, 632)])

    # predict
    detector.args.flag.show_steps = True
    ret = detector.detect_and_show_multimode_grasp(rgbd=rgbd, grip_args=detector.args.grip_net,
                                                   suction_args=detector.args.suction_net,
                                                   inner_grip_args=detector.args.inner_grip_net,
                                                   args=detector.args)

    # show
    cv2.imshow('reviewer', ret['im'][:, :, ::-1])
    cv2.waitKey()


if __name__ == '__main__':
    # demo_random_picking_simple()
    # demo_grip_gui(Detector=RandomGripDetector, cfg_path='kpick/apps/randompicking/configs/grip.cfg',
    #               default_cfg_path='kpick/apps/randompicking/configs/default.cfg')
    # demo_suction_gui(DetectorGui=get_suction_detector_gui(get_suction_detector_obj(RandomSuctionDetector)),
                     # cfg_path='kpick/apps/randompicking/configs/grip.cfg',
                     # default_cfg_path='kpick/apps/randompicking/configs/default.cfg')
    demo_multimode_grasp_gui(DetectorGui=get_multimode_grasp_detector_gui(get_multimode_grasp_detector_obj(OnePointDualGraspDetector)),
                     cfg_path='kpick/apps/randompicking/configs/one_point_dual.cfg',
                     default_cfg_path='kpick/apps/randompicking/configs/default.cfg')
    # demo_multimode_grasp_gui(
    #     DetectorGui=get_multimode_grasp_detector_gui(get_multimode_grasp_detector_obj(OnePointDualGraspDetector)),
    #     cfg_path='kpick/apps/randompicking/configs/two_point_dual.cfg',
    #     default_cfg_path='kpick/apps/randompicking/configs/default.cfg')
