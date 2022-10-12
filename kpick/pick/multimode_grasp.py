from kpick.pick.grip.grip_detection import GripDetector, get_base_cfg
from kpick.pick.grip_inner.inner_grip_detector import InnerGripDetector, get_base_inner_grip_cfg
from kpick.pick.suction.suction_points import SuctionPointsDetector, get_default_suction_points_args
from ketisdk.utils.proc_utils import CFG
import numpy as np
import cv2


def get_multimode_grasp_default_cfg():
    DEFAULT_ARGS = get_base_cfg()
    DEFAULT_ARGS.merge_with(get_default_suction_points_args())
    DEFAULT_ARGS.merge_with(get_base_inner_grip_cfg())
    DEFAULT_ARGS.mode = CFG()
    DEFAULT_ARGS.mode.active_modes = ['grip', 'suction', 'inner_grip', 'suction_2points']
    return DEFAULT_ARGS


class MultiModeGraspDetector(GripDetector, SuctionPointsDetector, InnerGripDetector):
    def init(self, grip_args=None, suction_args=None, inner_grip_args=None, rpn_args=None):
    # def init(self, **kwargs):
        if grip_args is not None: GripDetector.init(self, net_args=grip_args)
        if suction_args is not None: SuctionPointsDetector.init(self, net_args=suction_args, rpn_args=rpn_args)
        if inner_grip_args is not None: InnerGripDetector.get_model(self, net_args=inner_grip_args)

    def detect_and_show_multimode_grasp(self, rgbd, grip_args, suction_args, inner_grip_args, args, disp_mode='rgb'):
        out = rgbd.disp(mode=disp_mode)
        all_det, grasp_names = [], []

        if 'suction' in args.mode.active_modes:
            suction_det = self.detect_suctions(rgbd, net_args=suction_args, rpn_args=args.rpn,
                                               remove_bg=suction_args.remove_bg)
            out = self.show_suctions(rgbd, suction_det, net_args=suction_args, args=args, disp_mode=disp_mode, out=out)
            all_det.append(suction_det)
            grasp_names.append('suction')

        if 'grip' in args.mode.active_modes:
            grip_det = self.detect_grips(rgbd, net_args=grip_args, remove_bg=grip_args.remove_bg)
            out = self.show_grips(rgbd, grip_det, net_args=grip_args, args=args, disp_mode=disp_mode,out=out)
            all_det.append(grip_det)
            grasp_names.append('grip')

        if 'inner_grip' in args.mode.active_modes:
            inner_grip_det = self.detect_inner_grips(rgbd, thresh=inner_grip_args.score_thresh,
                                                     net_args=inner_grip_args)
            out = self.show_inner_grips(rgbd, inner_grip_det, net_args=inner_grip_args, args=args, disp_mode=disp_mode,
                                        out=out)
            all_det.append(inner_grip_det)
            grasp_names.append('inner_grip')

        if 'suction_2points' in args.mode.active_modes:
            suction_2points_det = self.detect_suctions_2points(rgbd, net_args=suction_args, rpn_args=args.rpn,
                                                               remove_bg=suction_args.remove_bg)
            out = self.show_suctions_2points(rgbd, det=suction_2points_det, net_args=suction_args, args=args,
                                             disp_mode=disp_mode, out=out)
            all_det.append(suction_2points_det)
            grasp_names.append('suction_2points')


        # select grasp
        grasp_scores = [float(det['best'].flatten()[-1]) if det is not None else -1 for det in all_det]
        sorted_inds = np.argsort(grasp_scores)[::-1]

        grasp_names = [grasp_names[i] for i in sorted_inds]
        grasp_scores = [grasp_scores[i] for i in sorted_inds]

        target_grasp, target_grasp_name = None, None
        if grasp_scores[0] >0:
            target_grasp = all_det[sorted_inds[0]]['best']
            target_grasp_name = grasp_names[0]

            ss = ''
            for n, s in zip(grasp_names, grasp_scores):
                ss += f'{n}:{s:.2f}_'
            left, top = rgbd.workspace.bbox[:2]
            cv2.putText(out, ss[:-1], (left, top), cv2.FONT_HERSHEY_COMPLEX, args.disp.text_scale, (0, 0, 255),
                        args.disp.text_thick)

        # print(f'{target_grasp_name}: {float(target_grasp.flatten()[-1]):>.3f}')

        return {'im': out, 'target_grasp': target_grasp, 'target_grasp_name': target_grasp_name}


def get_multimode_grasp_detector_obj(Detector=MultiModeGraspDetector):
    from ketisdk.vision.base.base_objects import BasObj
    class MultiModeGraspDetectorObj(MultiModeGraspDetector, BasObj):
        def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
            BasObj.__init__(self, args=args, cfg_path=cfg_path, name=name,
                            default_args=get_multimode_grasp_default_cfg())

            args_dict= {}
            if 'grip' in self.args.mode.active_modes: args_dict.update({'grip_args': self.args.grip_net})
            if 'suction' in self.args.mode.active_modes or 'suction_2points' in self.args.mode.active_modes:
                args_dict.update({'suction_args': self.args.suction_net, 'rpn_args': self.args.rpn})
            if 'inner_grip' in self.args.mode.active_modes: args_dict.update({'inner_grip_args': self.args.inner_grip_net})
            self.init(**args_dict)

        def load_params(self, args):
            BasObj.load_params(self, args=args)
            SuctionPointsDetector.load_params(self, net_args=self.args.suction_net)

    return MultiModeGraspDetectorObj


def get_multimode_grasp_detector_gui(DetectorObj=get_multimode_grasp_detector_obj()):
    from kpick.base.base import DetGui
    class MultiModeGraspDetectorGui(DetectorObj, DetGui):
        def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
            if method_ind == 0:
                ret = self.get_background_depth_map(rgbd=rgbd, net_args=self.args.suction_net)
            if method_ind == 1:
                ret = self.detect_and_show_grips(rgbd=rgbd, disp_mode=disp_mode, net_args=self.args.grip_net,
                                                 args=self.args, remove_bg=self.args.grip_net.remove_bg)
            if method_ind == 2:
                ret = self.detect_and_show_suctions(rgbd=rgbd, disp_mode=disp_mode, detected=kwargs['detected'],
                                                    remove_bg=self.args.suction_net.remove_bg,
                                                    net_args=self.args.suction_net,
                                                    rpn_args=self.args.rpn, args=self.args)

            if method_ind == 3:
                ret = self.detect_and_show_multimode_grasp(rgbd=rgbd, grip_args=self.args.grip_net,
                                                           suction_args=self.args.suction_net,
                                                           inner_grip_args=self.args.inner_grip_net,
                                                           args=self.args, disp_mode=disp_mode)

            return ret

    return MultiModeGraspDetectorGui


def demo_multimode_grasp_gui(cfg_path=None, default_cfg_path=None, DetectorGui=get_multimode_grasp_detector_gui(),
                             data_root=None, rgb_formats=None, depth_formats=None):
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.gui.gui import GUI, GuiModule

    get_multimode_grasp_detector_obj()()
    detect_module = GuiModule(DetectorGui, type='gripsuction_detector', name='GripSuction Detector',
                              category='detector', cfg_path=cfg_path, num_method=4,
                              default_cfg_path=default_cfg_path,
                              key_args=['grip_net.grip_w_ranges', 'grip_net.depth_grad_thresh', 'grip_net.ellipse_axes',
                                        'grip_net.erode_h', 'grip_net.remove_bg',
                                        'inner_grip_net.score_thresh', 'inner_grip_net.w_range',
                                        'suction_net.fc2conv', 'suction_net.remove_bg', 'suction_net.pad_sizes',
                                        'suction_net.stride', 'suction_net.score_thresh', 'suction_net.cups_dis',
                                        'rpn.enable', 'rpn.score_thresh', 'rpn.show_boxes']
                              )

    GUI(title='Grip + Suction Detection GUI',
        modules=[detect_module, ] + get_realsense_modules(),
        data_root=data_root, rgb_formats=rgb_formats, depth_formats=depth_formats
        )


if __name__ == '__main__':
    demo_multimode_grasp_gui(cfg_path='configs/multimode_grasp.cfg',
                          default_cfg_path='configs/default_multimode_grasp.cfg')
