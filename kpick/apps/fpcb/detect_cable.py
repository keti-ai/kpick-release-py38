import os

from kpick.segmentation.segmentation_base import Segmentator
from kpick.base.base import DetGuiObj
import numpy as np
import cv2
from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

class CableDetector(Segmentator):

    def smooth(self, im):
        # return cv2.boxFilter(im, -1, ksize=(3,3))
        return im

    def mask2grasp(self, mask=None, locs=None):
        if locs is None:
            locs = np.where(mask > 0)

        # pose
        Y, X = locs
        if len(Y) < 0: return None
        meanX, meanY = np.mean(X), np.mean(Y)
        meanXX, meanXY = np.mean(np.multiply(X, X)), np.mean(np.multiply(X, Y))
        varX = meanXX - meanX * meanX
        covXY = meanXY - meanX * meanY

        # tan = covXY / (varX + 0.00001)
        kk = - varX / (covXY + 0.0001)
        kk_ = - covXY / (varX + 0.00001)
        left, top, right, bottom = np.amin(X), np.amin(Y), np.amax(X), np.amax(Y)
        hh, ww = bottom - top, right - left
        angle = int(np.arctan(kk)*180/np.pi)

        rr = 10
        if hh > ww:
            y0, y1 = meanY - rr, meanY + rr
            x0, x1 = kk_ * (y0 - meanY) + meanX, kk_ * (y1 - meanY) + meanX
        else:
            x0, x1 = meanX - rr, meanX + rr
            y0, y1 = kk * (x0 - meanX) + meanY, kk * (x1 - meanX) + meanY

        return (int(meanX), int(meanY), angle, int(x0), int(y0), int(x1), int(y1))

    def mask2graspHead(self, mask, depth_invalid_mask, cab_thick=11, rgb=None):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), 'uint8'))
        mask = cv2.erode(mask, np.ones((cab_thick, cab_thick), 'uint8'))

        cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('viewer', 1080, 720)
        retval, labels = cv2.connectedComponents(mask)
        for val in range(retval):
            if val==0: continue
            maskv = labels==val
            Y, X = np.where(maskv)
            yc,xc = int(np.mean(Y)), int(np.mean(X))

            r = 2*cab_thick
            maski = np.zeros(mask.shape,'uint8')
            maski[yc-r:yc+r, xc-r: xc+r] =  np.copy(depth_invalid_mask[yc-r:yc+r, xc-r: xc+r])

            maskvi = np.logical_or(maskv, maski)
            Y,X = np.where(maskvi)

            maskg = np.zeros(mask.shape, 'uint8')
            maskg[yc - r:yc + r, xc - r: xc + r] = np.copy(mask[yc - r:yc + r, xc - r: xc + r])
            grasp = self.mask2grasp(mask=maskg)


            out = np.copy(rgb)
            locs = np.where(maskv > 0)
            out[locs] = 0.3 * out[locs] + (150, 0, 0)

            locs = np.where(maski > 0)
            out[locs] = 0.3 * out[locs] + (0, 150, 0)

            x0, y0, x1, y1 = grasp[-4:]
            cv2.line(out, (x0, y0), (x1, y1), (255,255,0), 2)

            cv2.putText(out, f'{len(Y)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,0), 2)

            cv2.imshow('viewer', out[:,:,::-1])

            cv2.waitKey()

        return mask

    def getGraspHead(self, depth, bg_depth, cab_thick=11):
        mask = self.getFgMaskByDepth(depth, bg_depth)
        return self.mask2graspHead(mask, cab_thick=cab_thick)

    def getGrasp(self, depth, bg_depth):
        mask = self.getTopMask(depth, bg_depth, percent=0.99)
        return self.mask2grasp(mask)

    def mask2lineOnly(self, mask, depth_valid_mask, cab_thick=11, head_thick=30, rgb=None):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), 'uint8'))
        ss = (cab_thick+head_thick)//2
        head_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ss, ss), 'uint8'))
        head_mask = cv2.dilate(head_mask, np.ones((41,41), 'uint8'))

        line_mask = np.logical_and(mask,np.logical_not(head_mask))
        line_mask = np.logical_and(line_mask, depth_valid_mask)

        line_mask = cv2.morphologyEx(line_mask.astype('uint8'), cv2.MORPH_OPEN,
                                     np.ones((5, 5), 'uint8'))

        return line_mask

    def getTopHighMask(self, depth, bg_depth, line_mask, thresh=5, diff_min=0, diff_max=100, percent=1.):
        ddepth = np.abs(depth.astype('float') - bg_depth.astype('float'))
        invalid_locs = np.where((depth == 0) | (bg_depth == 0))
        ddepth[invalid_locs] = 0
        ddepth[np.where(line_mask==0)] = 0

        ddepth = np.clip(ddepth, a_min=diff_min, a_max=diff_max)
        aa = np.histogram(ddepth, bins=np.unique(ddepth))
        aa[0][:3] = 0  # diff is small --> remove

        hist = aa[0] / np.sum(aa[0])
        for i in range(1, len(hist)):
            hist[i] += hist[i - 1]
        ddepth_max = aa[1][np.where(hist <= 0.999)][-1]
        # low_locs = np.where(ddepth<percent*np.amax(ddepth))
        # ddepth[low_locs]=0
        ddepth[np.where(ddepth < percent * ddepth_max)] = 0
        return 255 * (ddepth > thresh).astype('uint8')

class CableDectorGUI(CableDetector, DetGuiObj):
    def load_params(self, args):
        super().load_params(args=args)
        HOME_DIR = os.path.expanduser("~")
        self.data_dir = os.path.join(HOME_DIR,'000_yumi_shared')
        os.makedirs(self.data_dir, exist_ok=True)
        self.bg_crop_path = os.path.join(self.data_dir, 'bg_crop.png')
        if os.path.exists(self.bg_crop_path):
            self.bg_depth_crop = cv2.imread(self.bg_crop_path, cv2.IMREAD_UNCHANGED)
            print(f'Background  {self.bg_crop_path} loaded ...')
        else:
            print(f'Background  {self.bg_crop_path} does not exist ...')

    def getBgDepth(self, rgbd):
        self.bg_depth_crop =  self.smooth(rgbd.crop_depth())
        cv2.imwrite(self.bg_crop_path, self.bg_depth_crop)

        print('Cropped depth background stored ...')

    def getShowFgMask(self, rgbd, disp_mode='rgb'):
        out = rgbd.disp(mode=disp_mode)
        if hasattr(self, 'bg_depth_crop'):
            mask_crop = self.getFgMaskByDepth(self.smooth(rgbd.crop_depth()), self.bg_depth_crop)
            mask_crop = self.mask2lineOnly(mask_crop, rgbd.crop_depth()>0)

            mask = np.zeros((rgbd.height, rgbd.width), 'uint8')
            left, top, right, bottom = rgbd.workspace.bbox
            mask[top:bottom, left:right] = mask_crop

            locs = np.where(mask>0)
            out[locs] = 0.7*out[locs] +(0,75,0)

        return {'im': out}

    def getShowNormalizeDepth(self, rgbd, disp_mode='rgb'):
        out = rgbd.disp(mode=disp_mode)
        if hasattr(self, 'bg_depth_crop'):
            depth_norm_crop = self.normalizeDepth(self.smooth(rgbd.crop_depth()), self.bg_depth_crop)

            depth_norm = np.zeros((rgbd.height, rgbd.width), 'uint16') + depth_norm_crop[(0,0)]
            left, top, right, bottom = rgbd.workspace.bbox
            depth_norm[top:bottom, left:right] = depth_norm_crop

            rgbd.depth=depth_norm
            out = rgbd.disp(mode='depth_jet')
            # out = RGBD(rgb=rgbd.rgb, depth=depth_norm, workspace=rgbd.workspace).disp(mode='depth_jet')

        return {'im': out}

    def getShowGrasp(self, rgbd, disp_mode='rgb'):
        out = rgbd.disp(mode=disp_mode)
        depth_crop = rgbd.crop_depth()
        left, top, right, bottom = rgbd.workspace.bbox
        # cv2.imwrite('data/depth.png', rgbd.crop().disp(mode='depth')[:, :, ::-1])
        # cv2.imwrite('data/depth_bg.png', RGBD(depth=self.bg_depth_crop).crop().disp(mode='depth')[:, :, ::-1])
        grasp = None
        if hasattr(self, 'bg_depth_crop'):
            # cv2.imwrite('data/rgb.png', rgbd.crop_rgb()[:,:,::-1])
            mask_crop = self.getFgMaskByDepth(self.smooth(depth_crop), self.bg_depth_crop)
            # cv2.imwrite('data/mask_fg.png', (mask_crop).astype('uint8'))
            mask_crop = self.mask2lineOnly(mask_crop, depth_crop > 0)
            # cv2.imwrite('data/mask_line.png', (255 * mask_crop).astype('uint8'))
            mask_crop = self.getTopHighMask(self.smooth(depth_crop), self.bg_depth_crop, line_mask=mask_crop)
            # cv2.imwrite('data/mask_top.png', (mask_crop).astype('uint8'))

            mask = np.zeros((rgbd.height, rgbd.width), 'uint8')
            left, top, right, bottom = rgbd.workspace.bbox
            mask[top:bottom, left:right] = mask_crop

            # cv2.imshow('mask', mask)
            # cv2.waitKey()

            locs = np.where(mask > 0)
            out[locs] = 0.3*out[locs] +(150,0, 0)

            grasp = self.mask2grasp(mask)
            if grasp is not None:
                x0,y0, x1,y1 = grasp[-4:]
                cv2.line(out,(int(x0), int(y0)), (int(x1), int(y1)), (0,255,255), self.args.disp.line_thick)
            # cv2.imwrite('data/out.png', out[top:bottom, left:right][:,:,::-1])


        return {'im': out, 'grasp': grasp}

    def getShowGraspHead(self, rgbd, disp_mode='rgb', cab_thick=11):
        out = rgbd.disp(mode=disp_mode)
        if hasattr(self, 'bg_depth_crop'):
            mask_crop = self.getFgMaskByDepth(self.smooth(rgbd.crop_depth()), self.bg_depth_crop)
            # mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, np.ones((3,3), 'uint8'))

            mask = np.zeros((rgbd.height, rgbd.width), 'uint8')
            left, top, right, bottom = rgbd.workspace.bbox
            mask[top:bottom, left:right] = mask_crop

            # cv2.imshow('mask', mask)
            # cv2.waitKey()

            locs = np.where(mask > 0)
            out[locs] = 0.3*out[locs] +(150,0, 0)

            mask = self.mask2graspHead(mask, cab_thick=cab_thick, depth_invalid_mask=rgbd.depth==0, rgb=rgbd.rgb)

            locs = np.where(mask > 0)
            out[locs] = 0.3 * out[locs] + (0, 150, 0)

        return {'im': out}

    def getShowGraspLineOnly(self, rgbd, disp_mode='rgb', cab_thick=11):
        out = rgbd.disp(mode=disp_mode)
        if hasattr(self, 'bg_depth_crop'):
            mask_crop = self.getFgMaskByDepth(self.smooth(rgbd.crop_depth()), self.bg_depth_crop)
            mask_line_crop = self.mask2lineOnly(mask_crop, depth_valid_mask=rgbd.crop_depth()>0)

            mask = np.zeros((rgbd.height, rgbd.width), 'uint8')
            left, top, right, bottom = rgbd.workspace.bbox
            mask[top:bottom, left:right] = mask_line_crop

            locs = np.where(mask > 0)
            out[locs] = 0.3 * out[locs] + (0, 150, 0)

        return {'im': out}


    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', detected=None, **kwargs):
        if method_ind==3: ret = self.getShowFgMask(rgbd=rgbd, disp_mode=disp_mode)
        if method_ind==1: ret = self.getBgDepth(rgbd=rgbd)
        if method_ind == 2: ret = self.getShowNormalizeDepth(rgbd=rgbd, disp_mode=disp_mode)
        if method_ind == 0: ret = self.getShowGrasp(rgbd=rgbd, disp_mode=disp_mode)
        if method_ind == 4: ret = self.getShowGraspHead(rgbd=rgbd, disp_mode=disp_mode, cab_thick=11)
        if method_ind == 5: ret = self.getShowGraspLineOnly(rgbd=rgbd, disp_mode=disp_mode)
        return ret

def detect_cable(sensor_modules=[]):
    from ketisdk.gui.gui import GUI, GuiModule

    detect_module = GuiModule(CableDectorGUI, name='Cable Detector', num_method=6)
    GUI(title='Detect Cable', modules=[detect_module,] + sensor_modules,
        default_cfg_path='kpick/apps/fpcb/configs/cable_workspace.cfg',
        data_root='data/zivid/0216/',
        rgb_formats=['image/*',], depth_formats=['depth/*',]
        )

if __name__=='__main__':
    from ketisdk.sensor.zivid_sensor import get_zivid_module
    detect_cable(sensor_modules=[get_zivid_module(),])