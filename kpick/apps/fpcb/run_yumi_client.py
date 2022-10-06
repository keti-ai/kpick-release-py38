import os.path

from ketisdk.robot.yumi.yumi_client import run_yumi_client_gui, YumiClientGui
from kpick.apps.fpcb.detect_cable import CableDetector, CableDectorGUI
import kpick
import cv2
import numpy as np
from ketisdk.utils.proc_utils import ProcUtils
from shutil import copyfile
KPICK_DIR = os.path.split(kpick.__file__)[0]

from kpick.base.base import DetGuiObj
import json
import time


class CableCheckerGui(DetGuiObj):
    def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
        super().__init__(args=args, cfg_path=cfg_path, name=name, default_args=default_args)
        HOME_DIR = os.path.expanduser("~")
        self.req_file = os.path.join(HOME_DIR, '000_yumi_shared', 'req.json')
        self.res_file = os.path.join(HOME_DIR, '000_yumi_shared', 'res.json')

    def send_req(self, req):
        with open(self.req_file, 'w') as f: json.dump(req, f)
        print('REQUEST sent ...')

    def get_res(self, timeout=10):
        print('Scaning RES file ... ')
        timeout_ = time.time() + timeout*60
        while True:
            timeout_remain = (timeout_ - time.time())
            if timeout_remain % 60 == 0:
                print(f'[{timeout_remain//60}/{timeout}]s')
            if time.time()>timeout_:
                print(f'Timeout {timeout}s reached: Head existing checker existed ...')
                return
            time.sleep(1)
            if not os.path.exists(self.res_file): continue
            try:
                with open(self.res_file, 'r') as f:
                    res = json.load(f)
                    os.remove(self.res_file)
                print(f'RESPOND received ...')
                return res
            except:
                pass
            return

    def check_cable_head_loop(self, sensor):
        tag = ProcUtils().get_current_time_str()
        head_left = self.check_cable_head_1st(sensor=sensor, tag=f'{tag}/_1')
        # if head_left is not None:
        #     print('Stop cable head checking')
        #     return
        count=0
        while True:
            print('Waiting to check cable head while Stretching')
            res = self.get_res()
            if 'NEED_CHECK' not in res:
                continue
            if not res['NEED_CHECK']:
                break
            head_existed = self.check_cable_head_while_stretch(sensor=sensor, tag=f'{tag}/_2_{count}')
            if head_existed:
                break
            count+=1
        print('Stop cable head checking')

    def check_cable_head_while_stretch(self, sensor, tag='tag2'):
        ## check configs
        check_plane = {'z': 600, 'y': 450, 'dz': 60, 'dy': 200}
        workspace = [(630, check_plane['y'] - check_plane['dy']), (830, check_plane['y'] - check_plane['dy']),
                     (830, check_plane['y'] + check_plane['dy']), (630, check_plane['y'] + check_plane['dy'])]
        cable_line_width = 8
        cable_head_width = 28

        # get rgbd
        rgbd = sensor.get_rgbd()
        rgbd.set_workspace(pts=workspace)
        rgbd_crop = rgbd.crop()
        depth_crop = rgbd_crop.depth

        #
        mask_crop = np.logical_and(check_plane['z'] - check_plane['dz'] < depth_crop,
                                   depth_crop < check_plane['z'] + check_plane['dz']).astype('uint8')
        ksize = int(0.8*cable_head_width + 0.2*cable_line_width)
        mask_head_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN,
                                          kernel=np.ones((ksize, ksize), 'uint8'))
        Y, X = np.where(mask_head_crop > 0)
        head_size = len(Y)
        head_existed = head_size > (0.8*ksize * ksize)

        self.send_req(req={'HEAD_EXISTED': head_existed})

        rgb_crop_path = 'kpick/apps/fpcb/data/check/2_rgb_crop.png'
        depth_crop_path = 'kpick/apps/fpcb/data/check/2_depth_crop.png'
        depth_crop_show_path = 'kpick/apps/fpcb/data/check/2_depth_crop_show.png'
        mask_crop_path = 'kpick/apps/fpcb/data/check/2_mask_crop.png'
        mask_head_crop_path = 'kpick/apps/fpcb/data/check/2_mask_head_crop.png'

        cv2.imwrite(rgb_crop_path, rgbd_crop.bgr())
        cv2.imwrite(depth_crop_path, rgbd_crop.depth)
        cv2.imwrite(depth_crop_show_path, rgbd_crop.disp(mode='depth'))
        cv2.imwrite(mask_crop_path, 255 * mask_crop)
        cv2.imwrite(mask_head_crop_path, 255 * mask_head_crop)

        rgb_path = f'kpick/apps/fpcb/data/store/{tag}_rgb.png'
        depth_path = f'kpick/apps/fpcb/data/store/{tag}_depth.png'
        rgbd_show_path = f'kpick/apps/fpcb/data/store/{tag}_rgbd_show.png'

        save_dir = os.path.split(rgb_path)[0]
        os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(rgb_path, rgbd.bgr())
        cv2.imwrite(depth_path, rgbd.depth)
        cv2.imwrite(rgbd_show_path, rgbd.disp()[:, :, ::-1])
        copyfile(depth_crop_show_path, depth_crop_show_path.replace('check/2_', f'store/{tag}_'))
        copyfile(mask_crop_path, mask_crop_path.replace('check/2_', f'store/{tag}_'))
        copyfile(mask_head_crop_path, mask_head_crop_path.replace('check/2_', f'store/{tag}_'))

        return head_existed


    def check_cable_head_1st(self, sensor, detected=None, tag='tag1'):
        ## check configs
        check_plane = {'z': 550, 'y': 450, 'dz': 60, 'dy': 200}
        workspace = [(590, check_plane['y'] - check_plane['dy']), (840, check_plane['y'] - check_plane['dy']),
                     (840, check_plane['y'] + check_plane['dy']), (590, check_plane['y'] + check_plane['dy'])]
        cable_line_width = 10
        cable_head_width = 30

        # get rgbd
        rgbd = sensor.get_rgbd()
        rgbd.set_workspace(pts=workspace)
        rgbd_crop = rgbd.crop()
        depth_crop = rgbd_crop.depth

        #
        mask_crop = np.logical_and(check_plane['z'] - check_plane['dz'] < depth_crop,
                                  depth_crop < check_plane['z'] + check_plane['dz']).astype('uint8')
        ksize = int(0.7*cable_head_width + 0.3*cable_line_width)
        mask_head_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN,
                                          kernel=np.ones((ksize,ksize), 'uint8'))
        Y,X = np.where(mask_head_crop>0)
        head_size = len(Y)
        head_existed = head_size>(0.5*ksize*ksize)
        yc = int(np.mean(Y)) if head_existed else None
        xc = int(np.mean(X)) if head_existed else None
        zc = depth_crop[(yc, xc)] if head_existed else None

        head_left = yc > check_plane['dy'] if head_existed else None

        print(f'Head at ({xc},{yc},{zc}), head_left: {head_left}')

        self.send_req(req={'HEAD_LEFT': head_left})

        rgb_crop_path = 'kpick/apps/fpcb/data/check/1_rgb_crop.png'
        depth_crop_path = 'kpick/apps/fpcb/data/check/1_depth_crop.png'
        depth_crop_show_path = 'kpick/apps/fpcb/data/check/1_depth_crop_show.png'
        mask_crop_path = 'kpick/apps/fpcb/data/check/1_mask_crop.png'
        mask_head_crop_path = 'kpick/apps/fpcb/data/check/1_mask_head_crop.png'

        cv2.imwrite(rgb_crop_path, rgbd_crop.bgr())
        cv2.imwrite(depth_crop_path, rgbd_crop.depth)
        cv2.imwrite(depth_crop_show_path, rgbd_crop.disp(mode='depth'))
        cv2.imwrite(mask_crop_path, 255*mask_crop)
        cv2.imwrite(mask_head_crop_path, 255*mask_head_crop)

        rgb_path = f'kpick/apps/fpcb/data/store/{tag}_rgb.png'
        depth_path = f'kpick/apps/fpcb/data/store/{tag}_depth.png'
        rgbd_show_path = f'kpick/apps/fpcb/data/store/{tag}_rgbd_show.png'

        save_dir = os.path.split(rgb_path)[0]
        os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(rgb_path, rgbd.bgr())
        cv2.imwrite(depth_path, rgbd.depth)
        cv2.imwrite(rgbd_show_path, rgbd.disp()[:,:,::-1])
        copyfile(depth_crop_show_path, depth_crop_show_path.replace('check/1_',f'store/{tag}_'))
        copyfile(mask_crop_path, mask_crop_path.replace('check/1_',f'store/{tag}_'))
        copyfile(mask_head_crop_path, mask_head_crop_path.replace('check/1_',f'store/{tag}_'))


        #
        # cv2.waitKey()
        return head_left

    def gui_run(self, sensor=None, method_ind=0, **kwargs):
        if method_ind == 0: self.check_cable_head_loop(sensor=sensor)


def run_yumi_cable_gui(sensor_modules=[], cfg_path=None, data_dir=None, rgb_formats=None, depth_formats=None):
    from ketisdk.gui.gui import GUI, GuiModule

    # detect_module = GuiModule(CableYumi, type='ABB_robot', name='YuMi', category='robot_arm',
    #                           num_method=12, cfg_path=cfg_path)
    # GUI(title='Yumi Cable', modules=[detect_module, ] + sensor_modules)

    modules = sensor_modules
    modules += [GuiModule(CableDectorGUI, type='cable_detector', name='CabDet', category='detector',
                                num_method=6)]

    modules += [GuiModule(YumiClientGui, type='ABB_robot', name='YuMi', category='robot_arm',
                             num_method=12, cfg_path=cfg_path)]

    modules += [GuiModule(CableCheckerGui, type='cable_checker', name='Checker',
                               category='acquisition', num_method=3)]

    GUI(title='Yumi Cable', modules=modules, data_root=data_dir, rgb_formats=rgb_formats, depth_formats=depth_formats)


if __name__ == '__main__':
    from ketisdk.sensor.zivid_sensor import get_zivid_module
    from ketisdk.sensor.realsense_sensor import get_realsense_modules

    run_yumi_cable_gui(sensor_modules=[get_zivid_module(), ] + get_realsense_modules(),
                       cfg_path=os.path.join(KPICK_DIR, 'apps/fpcb/configs/yumi_zivid_client.cfg'),
                       data_dir='data/zivid/0303', rgb_formats='image/*', depth_formats='depth/*')
