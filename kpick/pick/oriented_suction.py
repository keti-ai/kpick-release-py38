from kpick.pick.grip.grip_detection import GripDetector, get_base_cfg
from kpick.pick.suction.suction_detection import SuctionDetector, get_default_suction_args
from ketisdk.utils.proc_utils import Timer
import numpy as np


def get_default_oriented_suction_cfg():
    args = get_base_cfg()
    args.merge_with(get_default_suction_args())
    return args


class OrientedSuctionDetector(GripDetector, SuctionDetector):
    def init(self, grip_args, suction_args):
        GripDetector.init(self, net_args=grip_args)
        SuctionDetector.init(self, net_args=suction_args)

    def load_params(self, grip_args, suction_args):
        GripDetector.load_params(self, net_args=grip_args)
        SuctionDetector.load_params(self, net_args=suction_args)

    def detect_oriented_suctions(self, rgbd, grip_net_args, suction_net_args):
        timer = Timer()
        self.gidx = grip_net_args.classes.index('grip')

        ret = self.find_grip_candidates(rgbd, grip_net_args)
        if ret is None:
            print('Best grip: Null')
            return None
        Grip, all_boxes, inds, array_rots = ret['grip'], ret['boxes'], ret['inds'], ret['arrays']

        Grip, all_boxes, inds = self.get_grip_in_workspace(rgbd, Grip, all_boxes, inds)
        if len(Grip) == 0:
            print('Best grip: Null')
            return None
        print(f'{len(Grip)} candidates in workspace')

        if grip_net_args.remove_bg:
            Grip, all_boxes, inds = self.remove_grasp_on_background(rgbd, Grip, all_boxes, inds, net_args=grip_net_args)
            if len(Grip) == 0:
                print('Best grip: Null')
                return None
            timer.pin_time('Remove_bg')
        timer.pin_time('Find_grip_candidates')

        # scores = self.predict_tensor_rois(im_tensors=array_rots, boxes=all_boxes, net_args=grip_net_args, inds=inds,
        #                                   model=self.grip_model, scaling=self.grip_scaling)
        # scores = scores[:, self.gidx].flatten()
        # Grip[:, -1] = scores
        #
        # timer.pin_time('Grip_Scoring')

        # cleft, ctop, cright, cbottom = rgbd.workspace.bbox
        #
        # Xc, Yc = (all_boxes[:, [0]] + all_boxes[:,[2]])//2, (all_boxes[:, [1]] + all_boxes[:,[3]])//2
        # suction_boxes, all_width, all_height = [], [], []
        # for p in suction_net_args.pad_sizes:
        #     ww, hh = p
        #     Left, Top = np.maximum(cleft, Xc - ww // 2), np.maximum(ctop, Yc - hh // 2)
        #     Right, Bottom = np.minimum(Left + ww, cright), np.minimum(Top + hh, cbottom)
        #     all_width.append(Right - Left)
        #     all_height.append(Bottom - Top)
        #     suction_boxes.append(np.concatenate((Left, Top, Right, Bottom), axis=1))
        # all_width = np.concatenate(all_width, axis=1)
        # all_height = np.concatenate(all_height, axis=1)
        # valid_locs = np.where((np.min(all_width, axis=1) > 5) & (np.min(all_height, axis=1) > 5))[0].flatten().tolist()
        # if len(valid_locs) == 0:
        #     print('Best grip: Null')
        #     return None
        # suction_boxes = [bxs[valid_locs, :] for bxs in suction_boxes]
        # inds = inds[valid_locs, :]
        # all_scores = [self.predict_tensor_rois(im_tensors=array_rots, boxes=bxs, model=self.suction_model, inds=inds,
        #                                        net_args=grip_net_args, scaling=self.suction_scaling)[:, [self.gidx]]
        #               for bxs in suction_boxes]
        # all_scores = np.concatenate(all_scores, axis=1)
        # suction_scores = np.mean(all_scores, axis=1, keepdims=True)

        scores = self.predict_tensor_rois(im_tensors=array_rots, boxes=all_boxes, net_args=suction_net_args, inds=inds,
                                          model=self.suction_model, scaling=self.suction_scaling)[:,self.gidx]
        Grip[:, -1] = scores
        timer.pin_time('Suction_Scoring')

        # locs = np.where(suction_scores<suction_net_args.score_thresh)[0].tolist()
        # if len(locs)==0:
        #     print('Best hybrid grasp: Null')
        #     return None
        # Grip = Grip[locs, :]
        print(f'Detected {len(Grip)} detected suctions')
        timer.pin_time('Get_oriented_suctions')

        # select best
        best_n_inds = self.select_best_n_grips(Grip, net_args=grip_net_args)
        best_ind = best_n_inds[0]
        best_grip = Grip[[best_ind], :]
        ret = {'grip': Grip, 'best_ind': best_ind, 'best_n_inds': best_n_inds, 'best': best_grip}
        print(f'Best oriented suction: {np.round(best_grip, decimals=3).tolist()}')
        print(timer.pin_times_str())

        return ret

def get_oriented_suction_detector_obj(Detector=OrientedSuctionDetector):
    from ketisdk.vision.base.base_objects import BasObj
    class OrinetedSuctionDetectorObj(Detector, BasObj):
        def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
            BasObj.__init__(self, args=args, cfg_path=cfg_path, name=name, default_args=get_default_oriented_suction_cfg())
            Detector.init(self, grip_args=self.args.grip_net, suction_args=self.args.suction_net)

        def load_params(self, args):
            BasObj.load_params(self, args=args)
            Detector.load_params(self, grip_args=self.args.grip_net, suction_args=self.args.suction_net)

        def detect_and_show_oriented_suction(self, rgbd, disp_mode='rgb'):
            ret = Detector.detect_oriented_suctions(self, rgbd=rgbd, grip_net_args=self.args.grip_net,
                                               suction_net_args=self.args.suction_net)
            out = self.show_grips(rgbd, ret, self.args.grip_net, self.args, disp_mode=disp_mode)
            if ret is not None:
                ret.update({'im': out})
            else:
                ret = {'im': rgbd.disp(mode=disp_mode), 'grip': None}

            return ret

    return OrinetedSuctionDetectorObj


def get_oriented_suction_detector_gui(DetectorObj=get_oriented_suction_detector_obj()):
    from kpick.base.base import DetGui
    class OrientedSuctionDetectorGuiObj(DetectorObj, DetGui):

        def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', detected=None, **kwargs):
            if method_ind == 0:
                ret = DetectorObj.detect_and_show_oriented_suction(self, rgbd=rgbd, disp_mode=disp_mode)
            if method_ind == 1:
                ret = DetectorObj.detect_and_show_grips(self, rgbd=rgbd, net_args=self.args.grip_net, args=self.args,
                                                        remove_bg=self.args.grip_net.remove_bg, disp_mode=disp_mode)
            if method_ind==2:
                ret = DetectorObj.detect_and_show_suctions(self, rgbd=rgbd, net_args=self.args.suction_net,
                                                           rpn_args=self.args.rpn, args=self.args,
                                                           remove_bg=self.args.suction_net.remove_bg, disp_mode=disp_mode)

            return ret

    return OrientedSuctionDetectorGuiObj


def demo_oriented_suction_gui(title='Hyubrid Grasp Detection GUI', Detector=OrientedSuctionDetector,
                          cfg_path='configs/ori_suction.cfg', default_cfg_path='configs/ori_suction_default.cfg',
                          sensors=['realsense', ], key_args=[],
                          data_root=None, rgb_formats=None, depth_formats=None):
    from ketisdk.gui.gui import GUI, GuiModule

    key_args += ['grip_net.erode_h', 'grip_net.grip_w_ranges', 'grip_net.depth_grad_thresh',
                 'grip_net.remove_bg', 'grip_net.top_n', 'grip_net.dy', 'grip_net.ellipse_axes']
    key_args += ['suction_net.fc2conv', 'suction_net.remove_bg', 'suction_net.pad_sizes',
                 'suction_net.stride', 'rpn.enable', 'rpn.score_thresh']

    detect_module = GuiModule(
        get_oriented_suction_detector_gui(DetectorObj=get_oriented_suction_detector_obj(Detector=Detector)),
        type='oriented_suction_detector', name='Oriented Suction Detector',
        category='detector', cfg_path=cfg_path, num_method=3,
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

def demo_hybrid_grasp_single():
    import cv2
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    # load model
    detector = get_hybrid_grasp_detector_obj()(cfg_path='configs/hybrid_grasp.cfg')

    # load image
    rgb = cv2.imread('data/test_images/hybrid_grasp_rgb.png')[:, :, ::-1]
    depth = cv2.imread('data/test_images/hybrid_grasp_depth.png', cv2.IMREAD_UNCHANGED)
    rgbd = RGBD(rgb=rgb, depth=depth, depth_min=400, depth_max=600,
                denoise_ksize=detector.args.sensor.denoise_ksize)
    rgbd.set_workspace(pts=[(360, 270), (889, 273), (890, 635), (359, 632)])

    # predict
    detector.args.flag.show_steps = True
    ret = detector.detect_and_show_hybrid(rgbd=rgbd)

    # show
    cv2.imshow('reviewer', ret['im'][:, :, ::-1])
    cv2.waitKey()


if __name__ == '__main__':
    # demo_hybrid_grasp_single()
    demo_oriented_suction_gui()
