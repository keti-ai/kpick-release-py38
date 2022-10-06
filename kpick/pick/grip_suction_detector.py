from kpick.pick.grip.grip_detection import GripDetector, get_base_cfg
from kpick.pick.suction.suction_detection import SuctionDetector, get_default_suction_args
from kpick.pick.grip_inner.inner_grip_detector import InnerGripDetector, get_base_inner_grip_cfg

def get_grip_suction_default_cfg():
    DEFAULT_ARGS = get_base_cfg()
    DEFAULT_ARGS.merge_with(get_default_suction_args())
    DEFAULT_ARGS.merge_with(get_base_inner_grip_cfg())
    return DEFAULT_ARGS

class GripSuctionDetector(GripDetector, SuctionDetector, InnerGripDetector):
    def init(self,  grip_args, suction_args, inner_grip_args):
        GripDetector.init(self, net_args=grip_args)
        SuctionDetector.init(self, net_args=suction_args)
        InnerGripDetector.get_model(self, net_args=inner_grip_args)



    def detect_and_show_grips_suctions(self, rgbd, grip_args, suction_args, inner_grip_args, args, disp_mode='rgb'):
        grip_det = self.detect_grips(rgbd, net_args=grip_args, remove_bg=grip_args.remove_bg)
        suction_det = self.detect_suctions(rgbd,net_args=suction_args, rpn_args=args.rpn, remove_bg=suction_args.remove_bg)
        inner_grip_det = self.detect_inner_grips(rgbd, thresh=inner_grip_args.score_thresh)

        out = self.show_grips(rgbd,grip_det, net_args=grip_args,args=args, disp_mode=disp_mode)
        out = self.show_suctions(rgbd,suction_det,net_args=suction_args, args=args,disp_mode=disp_mode, out=out)
        out = self.show_inner_grips(rgbd, inner_grip_det, net_args=inner_grip_args, args=args, disp_mode=disp_mode, out=out)

        return {'im':out}



from ketisdk.vision.base.base_objects import BasObj
class GripSuctionDetectorObj(GripSuctionDetector, BasObj):
    def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
        BasObj.__init__(self, args=args, cfg_path=cfg_path, name=name, default_args=get_grip_suction_default_cfg())
        self.init(grip_args=self.args.grip_net, suction_args=self.args.suction_net, inner_grip_args=self.args.inner_grip_net)

    def load_params(self, args):
        BasObj.load_params(self,args=args)
        SuctionDetector.load_params(self, net_args=self.args.suction_net)





from kpick.base.base import DetGui
class GripSuctionGuiDetector(GripSuctionDetectorObj, DetGui):
    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        if method_ind == 0:
            ret = self.get_background_depth_map(rgbd=rgbd, net_args=self.args.suction_net)
        if method_ind == 1:
            ret = self.detect_and_show_grips(rgbd=rgbd, disp_mode=disp_mode, net_args=self.args.grip_net,
                                             args=self.args, remove_bg=self.args.grip_net.remove_bg)
        if method_ind == 2:
            ret = self.detect_and_show_suctions(rgbd=rgbd, disp_mode=disp_mode, detected=kwargs['detected'],
                                             remove_bg=self.args.suction_net.remove_bg, net_args=self.args.suction_net,
                                             rpn_args=self.args.rpn, args=self.args)

        if method_ind==3:
            ret = self.detect_and_show_grips_suctions(rgbd=rgbd,grip_args=self.args.grip_net,
                                                      suction_args = self.args.suction_net,
                                                      inner_grip_args=self.args.inner_grip_net,
                                                      args=self.args, disp_mode=disp_mode)

        return ret

def demo_grip_suction_gui(cfg_path=None, default_cfg_path=None):
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.gui.gui import GUI, GuiModule

    GripSuctionGuiDetector()
    detect_module = GuiModule(GripSuctionGuiDetector, type='gripsuction_detector', name='GripSuction Detector',
                              category='detector', cfg_path=cfg_path, num_method=4,
                              default_cfg_path=default_cfg_path,
                              key_args=['grip_net.grip_w_ranges', 'grip_net.depth_grad_thresh', 'grip_net.ellipse_axes',
                                        'grip_net.erode_h', 'grip_net.remove_bg',
                                        'suction_net.remove_bg', 'suction_net.pad_sizes', 'suction_net.stride' ]
                              )

    GUI(title='Grip + Suction Detection GUI',
        modules=[detect_module, ] + get_realsense_modules(),
        )

if __name__=='__main__':
    demo_grip_suction_gui(cfg_path='configs/grip_suction.cfg',
                          default_cfg_path='configs/default_grip_suction.cfg')