from ketisdk.gui.default_config import default_args
from ketisdk.utils.proc_utils import CFG
from kpick.base.base import DetGuiObj

def get_default_args():
    args = default_args()

    args.pose = CFG()
    args.pose.intr_params = None
    args.pose.x_range = (-500, 500)
    args.pose.y_range = (-500, 500)
    args.pose.denoise_ksize = 25

    return args


class PoseEstimator():
    def get_xyz(self, rgbd, intr_params, denoise_ksize=25, xRange=(-500, 500), yRange=(-500, 500), zRange=(400, 800)):
        XYZ = rgbd.xyz(intr_params=intr_params, denoiseKSize=denoise_ksize)
        X, Y, Z = XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2]

        XIm = ((X - xRange[0]) / (xRange[1] - xRange[0]) * 255).astype('uint8')
        XIm[X<xRange[0]]=0
        XIm[X>xRange[1]]=255

        return {'im': XIm, 'array_real': X}




class PoseEstimatorGui(PoseEstimator, DetGuiObj):
    def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
        DetGuiObj.__init__(self, args=args, cfg_path=cfg_path, name=name, default_args=get_default_args())

    def array2Image(self, X, xRange=(-500, 500)):
        XIm = (X - xRange[0]) / (xRange[1] - xRange[0]) * 255
        XIm[X < xRange[0]] = 0
        XIm[X > xRange[1]] = 255
        return XIm.astype('uint8')

    def show_X(self, rgbd):
        XYZ = rgbd.xyz(intr_params=self.args.pose.intr_params, denoiseKSize=self.args.pose.denoise_ksize)
        return {'im': self.array2Image(XYZ[:,:,0], xRange=self.args.pose.x_range)}

    def show_Y(self, rgbd):
        XYZ = rgbd.xyz(intr_params=self.args.pose.intr_params, denoiseKSize=self.args.pose.denoise_ksize)
        return {'im': self.array2Image(XYZ[:,:,1], xRange=self.args.pose.y_range)}

    def show_Z(self, rgbd):
        XYZ = rgbd.xyz(intr_params=self.args.pose.intr_params, denoiseKSize=self.args.pose.denoise_ksize)
        return {'im': self.array2Image(XYZ[:,:,2], xRange=(self.args.sensor.depth_min, self.args.sensor.depth_max))}



    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        if method_ind==0:
            ret = self.show_X(rgbd=rgbd)
        if method_ind==1:
            ret = self.show_Y(rgbd=rgbd)
        if method_ind==2:
            ret = self.show_Z(rgbd=rgbd)
        return ret


def demo_pose_estimation(cfg_path='configs/pose.cfg', default_cfg_path='config/pose_default.cfg', use_sensors=['realsense']):
    from ketisdk.gui.gui import GUI, GuiModule
    sensor_modules = []
    if 'realsense' in use_sensors:
        from ketisdk.sensor.realsense_sensor import get_realsense_modules
        sensor_modules +=  get_realsense_modules()

    module = GuiModule(PoseEstimatorGui, name='pose estimator',cfg_path=cfg_path, default_cfg_path=default_cfg_path,
                       num_method=3, key_args=['pose.intr_params', 'pose.x_range', 'pose.y_range', 'pose.denoise_ksize'])

    GUI(title='Pose Estimation Demo', modules= [module,] + sensor_modules)

if __name__=='__main__':
    demo_pose_estimation()
    # PoseEstimatorGui()



