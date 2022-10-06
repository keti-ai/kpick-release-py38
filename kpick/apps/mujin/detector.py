from kpick.pick.suction.suction_detection import SuctionDetector
from ketisdk.utils.proc_utils import Timer, ProcUtils, ArrayUtils
import logging
import os
import kpick
KPICK_MODULE_DIR = os.path.split(kpick.__file__)[0]

# log_dir = 'data/logs'
# os.makedirs(log_dir, exist_ok=True)
# logging.basicConfig(filename=os.path.join(log_dir, f'log_{ProcUtils().get_current_time_str()}'),
#                     filemode='a',
#                     level=logging.DEBUG,
#                     format='%(message)s')
#
# logging.info(f'Filename \t\tRuntime')

# def getEdgeBlur(im, kernel_sizes=(9,21,21,21)):
#     edge = cv2.Canny(im, 100, 200)
#
#     texture = cv2.dilate(edge, np.ones((kernel_sizes[0],kernel_sizes[0]), 'uint8'))
#     texture = cv2.erode(texture, np.ones((kernel_sizes[1],kernel_sizes[1]), 'uint8'))
#     texture = cv2.dilate(texture, np.ones((kernel_sizes[2], kernel_sizes[2]), 'uint8'))
#
#     edge_filtered = 255*np.multiply(edge/255, 1-texture/255).astype('uint8')
#
#     edge_blur = cv2.blur(edge_filtered, (kernel_sizes[3],kernel_sizes[3]))
#
#     # cv2.imshow('edge', edge)
#     # cv2.imshow('texture', texture)
#     # cv2.imshow('edge_filtered', edge_filtered)
#     # cv2.waitKey()
#
#     return edge_blur/np.amax(edge_blur)

class MujinDetector(SuctionDetector):
    pass
    # def scoreFarEdge(self, rgbd, Suction):
    #     EdgeScore = np.zeros((rgbd.height, rgbd.width), 'float')
    #     left, top, right, bottom = rgbd.workspace.bbox
    #     EdgeScore[top:bottom, left:right] = getEdgeBlur(rgbd.crop_rgb(), kernel_sizes=self.args.tune.far_edge_ker_sizes)
    #
    #     Loc = (Suction[:, 1].astype('int'), Suction[:, 0].astype('int'))
    #     EdgeScore = EdgeScore[Loc].reshape((-1, 1))
    #
    #     return  1- EdgeScore
    #
    # def scoreDepth(self, rgbd, Suction):
    #     Score = Suction[:,-1].reshape((-1,1))
    #
    #     # Ellipse_score = (self.scoreInsideEllipse(Center=Suction[:, :2], axes=self.args.net.ellipse_axes,
    #     #                         Score=Suction[:, -1].reshape((-1, 1))) > .3).astype('float32')
    #     Ellipse_score = self.scoreInsideEllipse(Center=Suction[:, :2], axes=self.args.net.ellipse_axes,
    #                                             Score=Suction[:, -1].reshape((-1, 1)))
    #
    #     high_score_inds = np.where((np.multiply(Score,Ellipse_score) > self.args.net.score_thresh) & (Suction[:,2].reshape((-1,1)) > 100))[0].tolist()
    #     High_depth = Suction[high_score_inds, 2]
    #     depth_min, depth_max = np.amin(High_depth), np.amax(High_depth)
    #     #
    #     # out = rgbd.disp()
    #     # for suc in Suction[high_score_inds, :]:
    #     #     x, y = suc[:2].astype('int')
    #     #     cv2.drawMarker(out, (x,y), (255,0,0), cv2.MARKER_TILTED_CROSS, self.args.disp.marker_size)
    #     #
    #     # cv2.imshow('out', out)
    #     # cv2.waitKey()
    #
    #     # select_inds = np.array(high_score_inds)[np.where(High_depth< (depth_min + 0.3* (depth_max-depth_min)))[0].tolist()]
    #     select_inds = np.where(High_depth< (depth_min + 30))[0].tolist()
    #     select_inds_ = [high_score_inds[i] for i in select_inds]
    #
    #     out = np.zeros(Score.shape, 'float32')
    #     out[select_inds_] = 1.
    #
    #     return out.reshape((-1,1))
    #
    # def get_exp_scores(self, rgbd, Suction_):
    #     Suction = np.copy(Suction_)
    #     # ret *= (Suction[:,-1].reshape((-1,1))>self.args.net.score_thresh).astype('float')
    #     ret  = Suction[:,-1].reshape((-1,1))
    #     # ret *= self.scoreValidDepth(Suction[:,2])
    #     ret  = np.multiply(ret, self.scoreInsideEllipse(Center=Suction[:, :2], axes=self.args.net.ellipse_axes,
    #                                    Score=Suction[:, -1].reshape((-1, 1))))  # , Score=Suction[:,-1])
    #     ret  = np.multiply(ret, self.scoreFarEdge(rgbd, Suction))
    #     # # ret *= self.scoreDepth(Suction[:, 2])
    #     ret  = np.multiply(ret, self.scoreDepth(rgbd, Suction))
    #
    #     return ret

def create_and_load_mujin_detector(cfg_path=None):
    from kpick.pick.suction.suction_detection import get_suction_detector_obj
    if cfg_path is None:
        import kpick
        KPICK_DIR = os.path.split(kpick.__file__)[0]
        cfg_path = os.path.join(KPICK_DIR, 'apps/mujin/configs/suction.cfg')
        if os.path.exists(cfg_path):
            print(f'{cfg_path} does not exist ...')
    Module = get_suction_detector_obj(Detector=MujinDetector)

    return Module(cfg_path=cfg_path)

def demo_mujin_gui(cfg_path=None, default_cfg_path=None, data_root=None, rgb_formats=None, depth_formats=None):

    from kpick.pick.suction.suction_detection import demo_suction_gui

    if cfg_path is None: cfg_path = os.path.join(KPICK_MODULE_DIR, 'apps/mujin/configs/suction.cfg')
    if not os.path.exists(cfg_path):
        print(f'{cfg_path} does not exist ...')

    if default_cfg_path is None: default_cfg_path = os.path.join(KPICK_MODULE_DIR, 'apps/mujin/configs/default.cfg')
    if not os.path.exists(default_cfg_path):
        print(f'{default_cfg_path} does not exist ...')

    demo_suction_gui(cfg_path=cfg_path,  Detector=MujinDetector, default_cfg_path=default_cfg_path,
                     data_root=data_root, rgb_formats=rgb_formats, depth_formats=depth_formats,
                     )


def demo_mujin_single():
    import cv2
    import kpick
    import os
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    KPICK_DIR = os.path.split(kpick.__file__)[0]

    # configs
    depth_min = 700
    depth_max = 900
    workspace = [(300, 80), (1000, 80), (1000, 530), (300, 530)]

    # load image
    rgb = cv2.imread(os.path.join(KPICK_DIR, 'apps/mujin/test_images/2_rgb.png'))[:, :, ::-1]
    depth = cv2.imread(os.path.join(KPICK_DIR, 'apps/mujin/test_images/2_depth.png'), cv2.IMREAD_UNCHANGED)
    rgbd = RGBD(rgb=rgb, depth=depth, depth_min=depth_min, depth_max=depth_max)
    rgbd.set_workspace(pts=workspace)

    # load model
    detector = create_and_load_mujin_detector()

    # predict
    detector.args.flag.show_steps = False
    ret = detector.detect_and_show_suctions(rgbd=rgbd, net_args=detector.args.suction_net, rpn_args=detector.args.rpn,
                                            args=detector.args, remove_bg=False)

    # show
    cv2.imshow('reviewer', ret['im'][:, :, ::-1])
    cv2.waitKey()

if __name__=='__main__':
    demo_mujin_gui(data_root='data/apps/mujin/220712',
                   rgb_formats=['*rgb*'], depth_formats=['*depth*',])

    # demo_mujin_single()