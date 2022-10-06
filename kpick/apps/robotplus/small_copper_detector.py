import os.path

from kpick.pick.grip.grip_detection import GripDetector


class SmallCopperDetector(GripDetector):
    pass


def create_and_load_small_copper_detector(cfg_path=None):
    if cfg_path is None:
        import kpick
        KPICK_DIR = os.path.split(kpick.__file__)[0]
        cfg_path = os.path.join(KPICK_DIR, 'apps/robotplus/configs/copper_grip.cfg')
        if os.path.exists(cfg_path):
            print(f'{cfg_path} does not exist ...')

    return SmallCopperDetector(cfg_path=cfg_path)


def demo_detect_copper_gui(cfg_path=None, sensor_types=['realsense', 'zivid']):
    from kpick.base.base import DetGui
    class CopperDetectorGui(SmallCopperDetector, DetGui):
        def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', detected=None, **kwargs):
            if method_ind == 0:
                ret = self.get_background_depth_map(rgbd=rgbd)
            if method_ind == 1:
                ret = self.detect_and_show_poses(rgbd=rgbd, remove_bg=self.args.net.remove_bg,
                                                 disp_mode=disp_mode, detected=detected)
            return ret

    if cfg_path is None:
        import kpick
        KPICK_DIR = os.path.split(kpick.__file__)[0]
        cfg_path = os.path.join(KPICK_DIR, 'apps/robotplus/configs/small_copper_grip.cfg')
        if os.path.exists(cfg_path):
            print(f'{cfg_path} does not exist ...')

    from ketisdk.gui.gui import GUI, GuiModule
    detect_module = GuiModule(CopperDetectorGui, type='grip_detector', name='SCopperDet',
                              category='detector', cfg_path=cfg_path, num_method=2,
                              key_args=['net.grip_w_ranges', 'net.depth_grad_thresh', 'net.ellipse_axes',
                                        'net.erode_h', 'net.dy'])

    sensor_modules = []
    for sensor_type in sensor_types:
        if sensor_type == 'realsense':
            from ketisdk.sensor.realsense_sensor import get_realsense_modules
            sensor_modules += get_realsense_modules()
        if sensor_type == 'zivid':
            from ketisdk.sensor.zivid_sensor import get_zivid_module
            sensor_modules += [get_zivid_module(), ]

    GUI(title='Small Copper Detection GUI',
        modules=[detect_module, ] + sensor_modules,
        )


def demo_detect_small_copper_single():
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
    detector = create_and_load_small_copper_detector()

    # predict
    detector.args.flag.show_steps = True
    ret = detector.detect_and_show_poses(rgbd=rgbd)

    # show
    cv2.imshow('reviewer', ret['im'][:, :, ::-1])
    cv2.waitKey()


if __name__ == '__main__':
    demo_detect_copper_gui()
    # demo_detect_copper_single()
