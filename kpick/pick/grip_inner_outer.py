from kpick.pick.grip.grip_detection import GripDetector, get_base_cfg
from kpick.pick.grip_inner.inner_grip_detector import InnerGripDetector, get_base_inner_grip_cfg
import cv2

def get_inner_outer_grip_cfg():
    args = get_base_cfg()
    args.merge_with(get_base_inner_grip_cfg())
    return args


class InnerOuterGripDetector(GripDetector, InnerGripDetector):
    def init(self, grip_net_args, inner_grip_net_args):
        GripDetector.init(self, net_args=grip_net_args)
        InnerGripDetector.get_model(self, net_args=inner_grip_net_args)

    def detect_inner_outer_grips(self, rgbd, grip_net_args, inner_grip_net_args, remove_bg=False,
                                 inner_grip_thresh=0.5):
        ret = GripDetector.detect_grips(self, rgbd=rgbd, net_args=grip_net_args, remove_bg=remove_bg)
        ret2 = InnerGripDetector.detect_inner_grips(self, rgbd=rgbd, net_args=inner_grip_net_args,
                                                    thresh=inner_grip_thresh)
        ret.update(ret2)
        best_grip = ret['grip'][ret['best_ind']]
        best_inner_grip = ret['inner_grip'][ret['best_inner_ind']]
        ret['target'] = best_grip if best_grip[-1] >= best_inner_grip[-1] else best_inner_grip
        ret['target_name'] = 'grip' if best_grip[-1] >= best_inner_grip[-1] else 'inner_grip'
        return ret

    def show_detected(self, rgbd, det, grip_net_args, inner_grip_net_args, args, disp_mode='rgb', out=None):
        out = GripDetector.show_grips(self, rgbd=rgbd, det=det, net_args=grip_net_args,
                                      args=args, disp_mode=disp_mode, out=out)
        out = InnerGripDetector.show_inner_grips(self, rgbd=rgbd, det=det, net_args=inner_grip_net_args,
                                                 args=args, disp_mode=disp_mode, out=out)

        left,top = rgbd.workspace.bbox[:2]
        str1 = f'Grip: {det["grip"][det["best_ind"]][-1]:>0.3f}'
        str2 =  f'Inner_grip: {det["inner_grip"][det["best_inner_ind"]][-1]:>0.3f}'
        cv2.putText(out, str1 if det['target_name']=='grip' else str2, (left, top-20), cv2.FONT_HERSHEY_COMPLEX,
                    args.disp.text_scale,(255, 0, 0), args.disp.text_thick)
        cv2.putText(out, str2 if det['target_name'] == 'grip' else str1, (left+300, top - 20), cv2.FONT_HERSHEY_COMPLEX,
                    args.disp.text_scale, (0, 0, 255), args.disp.text_thick)

        print(det['target_name'])
        return out


def get_inner_outer_grip_detector_obj(Detector=InnerOuterGripDetector):
    from ketisdk.vision.base.base_objects import BasObj
    class InnerOuterGripDetectionObj(Detector, BasObj):
        def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
            BasObj.__init__(self, args=args, cfg_path=cfg_path, name=name, default_args=get_inner_outer_grip_cfg())
            Detector.init(self, grip_net_args=self.args.grip_net, inner_grip_net_args=self.args.inner_grip_net)

        def load_params(self, args):
            BasObj.load_params(self, args=args)
            Detector.load_params(self, net_args=self.args.grip_net)

        def detect_and_show(self, rgbd, disp_mode='rgb'):
            ret = Detector.detect_inner_outer_grips(self, rgbd=rgbd, grip_net_args=self.args.grip_net,
                                                    remove_bg=self.args.grip_net.remove_bg,
                                                    inner_grip_net_args=self.args.inner_grip_net,
                                                    inner_grip_thresh=self.args.inner_grip_net.score_thresh)
            out = Detector.show_detected(self, rgbd=rgbd, det=ret, grip_net_args=self.args.grip_net,
                                         inner_grip_net_args=self.args.inner_grip_net, args=self.args,
                                         disp_mode=disp_mode)
            ret['im'] = out
            return ret

    return InnerOuterGripDetectionObj


def get_inner_outer_grip_detection_gui(DetectorObj=get_inner_outer_grip_detector_obj()):
    from kpick.base.base import DetGui
    class InnerOuterGripGuiDetector(DetectorObj, DetGui):

        def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', detected=None, **kwargs):
            if method_ind == 0:
                ret = self.detect_and_show(rgbd=rgbd, disp_mode=disp_mode)

            return ret

    return InnerOuterGripGuiDetector


def demo_inner_outer_grip_gui(title='Grip Detection GUI', Detector=InnerOuterGripDetector,
                              cfg_path='configs/inner_outer_grip.cfg', default_cfg_path='configs/inner_outer_grip_default.cfg',
                              sensors=['realsense', ], key_args=[],
                              data_root=None, rgb_formats=None, depth_formats=None):
    from ketisdk.gui.gui import GUI, GuiModule

    key_args += ['grip_net.erode_h', 'grip_net.grip_w_ranges', 'grip_net.depth_grad_thresh',
                 'grip_net.remove_bg', 'grip_net.top_n', 'grip_net.dy', 'grip_net.ellipse_axes']
    key_args += ['inner_grip_net.score_thresh']
    detect_module = GuiModule(
        get_inner_outer_grip_detection_gui(DetectorObj=get_inner_outer_grip_detector_obj(Detector=Detector)),
        type='grip_detector', name='Inner Outer Grip Detector',
        category='detector', cfg_path=cfg_path, num_method=1,
        key_args=key_args,
        )

    modules = [detect_module, ]
    if 'realsense' in sensors:
        from ketisdk.sensor.realsense_sensor import get_realsense_modules
        modules += get_realsense_modules()
    if 'zivid' in sensors:
        from ketisdk.sensor.zivid_sensor import get_zivid_module
        modules += [get_zivid_module(), ]

    GUI(title=title, modules=modules, data_root=data_root,
        rgb_formats=rgb_formats, depth_formats=depth_formats, default_cfg_path=default_cfg_path)

def demo_inner_grip_single():
    import cv2
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    # load model
    detector = get_inner_outer_grip_detector_obj()(cfg_path='configs/inner_outer_grip.cfg')

    # load image
    rgb = cv2.imread('data/test_images/inner_outer_grip_rgb.png')[:, :, ::-1]
    depth = cv2.imread('data/test_images/inner_outer_grip_depth.png', cv2.IMREAD_UNCHANGED)
    rgbd = RGBD(rgb=rgb, depth=depth, depth_min=600, depth_max=800,
                denoise_ksize=detector.args.sensor.denoise_ksize)
    rgbd.set_workspace(pts=[(360, 270), (889, 273), (890, 635), (359, 632)])

    # predict
    detector.args.flag.show_steps = True
    ret = detector.detect_and_show(rgbd=rgbd)

    # show
    cv2.imshow('reviewer', ret['im'][:, :, ::-1])
    cv2.waitKey()


if __name__ == '__main__':
    # demo_inner_outer_grip_gui()
    demo_inner_grip_single()
