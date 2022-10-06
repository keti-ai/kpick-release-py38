from kpick.pick.grip.grip_detection import GripDetector
class SmartGripDetector(GripDetector):
    def get_exp_scores(self,rgbd, Grip):
        ret = 1
        # ret  *= self.scoreValidDepth(Grip[:,2])
        ret *= self.scoreInsideEllipse(Center=Grip[:, :2], Angle=Grip[:, 5].reshape((-1, 1)),
                                       axes=self.args.net.ellipse_axes)
        return ret

from kpick.base.base import DetGui
class SmartGripDetectorGui(SmartGripDetector, DetGui):
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

    cfg_path = 'apps/smart_gripper/configs/grip.cfg'
    detect_module0 = GuiModule(SmartGripDetectorGui, type='grip_detector', name='Grip Detector',
                              category='detector', cfg_path=cfg_path, num_method=9)
    # cfg_path = 'configs/grip_inner.cfg'
    # detect_module1 = GuiModule(GripInnerDetectorGui, type='grip_inner_detector', name='Grip Inner Detector',
    #                            category='detector', cfg_path=cfg_path, num_method=5)

    GUI(title='RobotPlus Detection GUI',
           modules=[detect_module0] + get_realsense_modules(),
           )


if __name__=='__main__':
    demo_detect_grasp_gui()