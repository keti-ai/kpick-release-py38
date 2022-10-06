from kpick.pick.grip.grip_detection import GripDetector
class CylindGraspDetector(GripDetector):
    def get_exp_scores(self,rgbd, Grip):
        ret = 1
        # ret  *= self.scoreValidDepth(Grip[:,2])
        ret *= self.scoreInsideEllipse(Center=Grip[:, :2], Angle=Grip[:, 5].reshape((-1, 1)),
                                       axes=self.args.net.ellipse_axes)
        return ret

from kpick.base.base import DetGui
class CylindGraspDetectorGui(CylindGraspDetector, DetGui):
    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', detected=None):
        if method_ind == 0:
            ret = self.detect_and_show_poses(rgbd=rgbd, disp_mode=disp_mode, detected=detected)
        if method_ind==1:
            ret = self.detect_and_show_poses(rgbd=rgbd, remove_bg=True, disp_mode=disp_mode, detected=detected)
        if method_ind==2:
            ret = self.get_background_depth_map(rgbd=rgbd)
        if method_ind==3:
            ret = self.getShowMomentScore(rgbd=rgbd, disp_mode=disp_mode)
        if method_ind==4:
            ret = self.getShowSpatialScore(rgbd=rgbd, disp_mode=disp_mode)
        if method_ind==5:
            ret = self.getShowDepthScore(rgbd=rgbd, disp_mode=disp_mode)
        if method_ind==6:
            ret = self.getShowWidthScore(rgbd=rgbd, disp_mode=disp_mode)
        if method_ind==7:
            ret = self.getShowExpScore(rgbd=rgbd, disp_mode=disp_mode)
        return  ret

def demo_detect_grasp_gui():

    # from kpick.pick.grip_inner.grip_inner import GripInnerDetectorGui
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.gui.gui import GUI, GuiModule

    cfg_path = 'apps/robotplus/configs/grip.cfg'
    detect_module0 = GuiModule(CylindGraspDetectorGui, type='grip_detector', name='Grip Detector',
                              category='detector', cfg_path=cfg_path, num_method=9)
    # cfg_path = 'configs/grip_inner.cfg'
    # detect_module1 = GuiModule(GripInnerDetectorGui, type='grip_inner_detector', name='Grip Inner Detector',
    #                            category='detector', cfg_path=cfg_path, num_method=5)

    GUI(title='RobotPlus Detection GUI',
           modules=[detect_module0] + get_realsense_modules(),
           )

class SmallCylindGraspDetector(GripDetector):
    def get_exp_scores(self,rgbd, Grip):
        ret = 1
        # ret  *= self.scoreValidDepth(Grip[:,2])
        ret *= self.scoreInsideEllipse(Center=Grip[:, :2], Angle=Grip[:, 5].reshape((-1, 1)),
                                       axes=self.args.net.ellipse_axes)
        return ret

class SmallCylindGraspDetectorGui(SmallCylindGraspDetector, DetGui):
    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        if method_ind == 0:
            ret = self.detect_and_show_poses(rgbd=rgbd, disp_mode=disp_mode, remove_bg=self.args.net.remove_bg)
        if method_ind==1:
            ret = self.get_background_depth_map(rgbd=rgbd)
        return  ret

def demo_small_cylind_gui(cfg_path='apps/robotplus/configs/small_cylind.cfg', default_cfg_path='configs/default_grip.cfg'):
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.gui.gui import GUI, GuiModule

    detect_module0 = GuiModule(SmallCylindGraspDetectorGui, type='grip_detector', name='Grip Detector',
                               category='detector', cfg_path=cfg_path, num_method=6,
                               key_args=['net.grip_w_ranges', 'net.depth_grad_thresh', 'net.ellipse_axes',
                                         'net.erode_h', 'net.dy', 'net.remove_bg']
                               )

    GUI(title='Small Cylind Detector', default_cfg_path=default_cfg_path,
        modules=[detect_module0] + get_realsense_modules(),
        )



if __name__=='__main__':
    # demo_detect_grasp_gui()
    demo_small_cylind_gui()
