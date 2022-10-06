from kpick.base.base import DetGuiObj
from ketisdk.gui.default_config import default_args
from ketisdk.utils.proc_utils import CFG, Timer
import cv2
import numpy as np
import sparse


def get_default_args():
    args = default_args()
    args.fa = CFG()
    args.fa.blur_ksize = (7, 7)
    args.fa.scale = 10
    args.fa.blur_thresh = 10
    args.fa.trans_range = (-20, 20, 2)
    args.fa.rot_range = (0, 360, 10)
    args.fa.test_rot_angle=90
    args.fa.contour_min_len=4

    return args


class FocusAreaDetectorGui(DetGuiObj):
    def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
        DetGuiObj.__init__(self, args=args, cfg_path=cfg_path, name=name, default_args=get_default_args())

    def get_noise_map(self, I, blur_ksize):
        base = cv2.blur(I, blur_ksize)
        return I.astype('float32') - base.astype('float32')

    def get_noise_mask(self, I, blur_ksize, blur_thresh):
        noise = self.get_noise_map(I, blur_ksize)
        noise_mask = (np.abs(noise) > self.args.fa.blur_thresh).astype('uint8')
        noise_mask = cv2.morphologyEx(noise_mask, cv2.MORPH_CLOSE, np.ones((3, 3), 'uint8'))
        return noise_mask

    def get_show_noise_layer(self, rgbd, disp_mode='rgb'):
        # out = rgbd.disp(mode=disp_mode)
        left, top, right, bottom = rgbd.workspace.bbox
        h, w = rgbd.rgb.shape[:2]
        I = rgbd.crop_gray()

        noise_mask = self.get_noise_mask(I, self.args.fa.blur_ksize, self.args.fa.blur_thresh)

        out = rgbd.disp()
        Y, X = np.where(noise_mask>0)
        out[(Y+top, X+left)] = (0,255,0)
        return {'im': out}

    def get_contours_from_noise(self, noise_mask, contour_min_len=4):
        contours, hierarchy = cv2.findContours(image=noise_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if c.shape[0] >= contour_min_len]
        return contours

    def get_contours_and_show(self, rgbd):
        left, top, right, bottom = rgbd.workspace.bbox
        h, w = rgbd.rgb.shape[:2]
        I = rgbd.crop_gray()
        noise_mask = self.get_noise_mask(I, self.args.fa.blur_ksize, self.args.fa.blur_thresh)

        contours = self.get_contours_from_noise(noise_mask)

        rgb_crop = rgbd.crop_rgb()
        cv2.drawContours(image=rgb_crop, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,lineType=cv2.LINE_AA)
        # out_im = np.zeros((rgbd.rgb.shape[:2]), 'uint8')
        out_im = rgbd.rgb.copy()
        out_im[top:bottom, left:right, :] = rgb_crop
        return {'im': out_im}

    def get_feature_map_(self, I, contour_min_len=4):
        noise_mask = self.get_noise_mask(I, self.args.fa.blur_ksize, self.args.fa.blur_thresh)
        contours = self.get_contours_from_noise(noise_mask, contour_min_len=contour_min_len)

        feature_mask = np.zeros_like(I)
        h,w = feature_mask.shape[:2]
        cv2.drawContours(image=feature_mask, contours=contours, contourIdx=-1, color=255, thickness=1)
        feature_mask = cv2.morphologyEx(feature_mask, cv2.MORPH_CLOSE, np.ones((7, 7), 'uint8'))

        # hole filling
        flood = feature_mask.copy()
        cv2.floodFill(flood,np.zeros((h+2, w+2), 'uint8'),(0,0), 255)
        inv_flood = cv2.bitwise_not(flood)
        feature_mask = feature_mask | inv_flood
        feature_mask = feature_mask//255

        # feature_mask = cv2.blur(feature_mask, (21,21))

        # cut small part
        kernel = np.ones((7,7), 'uint8')
        feature_mask = cv2.erode(feature_mask, kernel)

        # get largest connected component
        ret_val, label_mask, stats, centers= cv2.connectedComponentsWithStats(feature_mask)
        if ret_val>2:
            areas = [s[-1] for s in stats]
            areas[0]=0
            arg_max = np.argmax(areas)
            feature_mask=(label_mask==arg_max).astype('uint8')

        feature_mask = cv2.dilate(feature_mask, kernel)
        # feature_mask = cv2.blur(255*feature_mask, (21, 21))

        feature_mask = feature_mask*255

        return feature_mask

    def get_feature_map(self, I, contour_min_len=4):
        n=11
        Sml= 0
        I = I.astype('float32')
        for r in range(1,n+1):
            Sml+= np.abs(2*I - np.roll(I, r, axis=0) - np.roll(I, r, axis=0)) +\
                  np.abs(2*I - np.roll(I, r, axis=1) - np.roll(I, r, axis=1))
        Sml *= 255/np.amax(Sml)
        return Sml.astype('uint8')



    def get_feature_map_and_show(self, rgbd):
        left, top, right, bottom = rgbd.workspace.bbox
        h, w = rgbd.rgb.shape[:2]
        I = rgbd.crop_gray()

        feature_map = self.get_feature_map(I, self.args.fa.contour_min_len)

        mask = np.zeros((h,w), 'uint8')
        mask[top:bottom, left:right] = feature_map
        out = cv2.bitwise_and(rgbd.rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))

        return {'im': feature_map}

    def locs1d2boundlocs(self, locs1d, h, w):
        Yr = locs1d // w
        Xr = locs1d - Yr * w
        mask = np.zeros((h, w), 'uint8')
        mask[(Yr.astype('int'), Xr.astype('int'))] = 255
        bound = np.bitwise_xor(mask, cv2.dilate(mask, np.ones((3, 3), 'uint8')))
        return np.where(bound > 0)

    def get_rotated_features(self, feature_map, rot_range, out=None):
        hi, wi = feature_map.shape[:2]
        xic, yic = float(wi) / 2, float(hi) / 2

        Y, X = np.where(feature_map > 0)
        X, Y = X.reshape(1, -1).astype('float32') - xic, Y.reshape(1, -1).astype('float32') - yic
        L = np.concatenate((X, Y), axis=0)

        Angle = np.arange(*rot_range).reshape((1, 1, -1))
        # self.Angle = Angle.flatten()
        CosA, SinA = np.cos(Angle * np.pi / 180), np.sin(Angle * np.pi / 180)
        M = np.concatenate((
            np.concatenate((CosA, -SinA), axis=1),
            np.concatenate((SinA, CosA), axis=1),
        ), axis=0).transpose((2, 0, 1))

        L1 = np.einsum('mij,jn->min', M, L)
        L = (L1[:, 0, :] + xic).astype('int') + wi * (L1[:, 1, :] + yic).astype('int')

        R = []
        for l in L:
            ll = np.zeros((hi * wi), 'uint8')
            l = l[l < hi * wi]
            ll[l] = 1
            ll_sparse = sparse.COO(ll)
            R.append(ll_sparse.reshape((1, -1)))
            # self.ref_R.append(ll.reshape((1,-1)))
        R = np.concatenate(R, axis=0)

        if out is not None:         # for  debug
            locs1d = np.where(R[0,:]>0)[0]
            matched_locs = self.locs1d2boundlocs(locs1d,hi,wi)
            out[matched_locs] = (255, 0, 0)
            cv2.imshow('matched', out[:, :, ::-1])
            cv2.waitKey()
        return R, Angle.flatten()

    def get_rotated_features_v2(self, feature_map, rot_range, out=None):
        hi, wi = feature_map.shape[:2]
        xic, yic = wi//2, hi//2

        Angle = np.arange(*rot_range)
        R = []
        for  angle in Angle:
            if angle==0:
                rotated_feat = feature_map
            else:
                M = cv2.getRotationMatrix2D((xic, yic), -angle, 1.0)
                rotated_feat = cv2.warpAffine(feature_map, M, (wi, hi))
            R.append(sparse.COO(rotated_feat).reshape((1,-1)))
        R = np.concatenate(R, axis=0)

        if out is not None:         # for  debug
            # locs1d = np.where(R[6,:]>0)[0]
            # matched_locs = self.locs1d2boundlocs(locs1d,hi,wi)
            # out[matched_locs] = (255, 0, 0)
            mask = R[6,:].reshape((hi, wi))
            out = cv2.bitwise_and(out, cv2.cvtColor(mask.todense(), cv2.COLOR_GRAY2RGB))
            cv2.imshow('matched', out[:, :, ::-1])
            cv2.waitKey()
        return R, Angle.flatten()



    def get_translated_features(self,feature_map, trans_range, out=None):
        hi, wi = feature_map.shape[:2]
        xic, yic = float(wi) / 2, float(hi) / 2

        Y, X = np.where(feature_map > 0)
        Y, X = Y.reshape((1, -1)), X.reshape((1, -1))
        L = np.concatenate((X, Y, np.ones_like(X)), axis=0)
        Tx, Ty = np.meshgrid(range(*trans_range), range(*trans_range))
        Tx, Ty = Tx.reshape((1, 1, -1)), Ty.reshape((1, 1, -1))
        Ones, Zeros = np.ones_like(Tx), np.zeros_like(Tx)
        T = np.concatenate((
            np.concatenate((Ones, Zeros, Tx), axis=1),
            np.concatenate((Zeros, Ones, Ty), axis=1),
        ), axis=0).transpose((2, 0, 1))

        T1 = np.einsum('nij,jl->nil', T, L)
        T1 = T1[:, 0, :] + wi * T1[:, 1, :]

        T = []
        for t in T1:
            ll = np.zeros((hi * wi), 'uint8')
            t = t[t < hi * wi]
            ll[t] = 1
            ll_sparse = sparse.COO(ll)
            T.append(ll_sparse.reshape((-1, 1)))
            # target_T.append(ll.reshape((-1, 1)))
        T = np.concatenate(T, axis=1)

        if out is not None:
            locs1d = np.where(T[:,10]>0)[0]
            matched_locs = self.locs1d2boundlocs(locs1d,hi,wi)
            out[matched_locs] = (255, 0, 0)
            cv2.imshow('matched', out[:, :, ::-1])
            cv2.waitKey()

        return T, Tx.flatten(), Ty.flatten()

    def get_translated_features_v2(self,feature_map, trans_range, out=None):
        hi, wi = feature_map.shape[:2]
        xic, yic = float(wi) / 2, float(hi) / 2

        Tx, Ty = np.meshgrid(range(*trans_range), range(*trans_range))
        Tx, Ty = Tx.flatten(), Ty.flatten()

        T = []
        for tx, ty in zip(Tx, Ty):
            translated_feat = np.roll(np.roll(feature_map,ty ,axis=0), tx, axis=1)
            T.append(sparse.COO(translated_feat).reshape((-1, 1)))
            # target_T.append(ll.reshape((-1, 1)))
        T = np.concatenate(T, axis=1)

        if out is not None:
            # locs1d = np.where(T[:,10]>0)[0]
            # matched_locs = self.locs1d2boundlocs(locs1d,hi,wi)
            # out[matched_locs] = (255, 0, 0)
            mask = T[:, 10].todense().reshape((hi, wi))
            out = cv2.bitwise_and(out, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
            cv2.imshow('matched', out[:, :, ::-1])
            cv2.waitKey()

        return T, Tx, Ty

    def get_reference_feature_maps(self, rgbd):
        timer = Timer()
        I = rgbd.crop_gray()


        feature_map = self.get_feature_map(I, contour_min_len=self.args.fa.contour_min_len)
        timer.pin_time('get_feature')

        # self.ref_R, self.Angle = self.get_rotated_features(feature_map, self.args.fa.rot_range)
        self.ref_R, self.Angle = self.get_rotated_features_v2(feature_map, self.args.fa.rot_range)
        # self.ref_T, self.Tx, self.Ty = self.get_translated_features_v2(feature_map, self.args.fa.trans_range)
        timer.pin_time('rotate')

        print(timer.pin_times_str())

        print('Reference feature maps saved')

    def match_ref_feature_to_im(self,rgbd):
        timer = Timer()
        h, w = rgbd.rgb.shape[:2]
        left, top, right, bottom = rgbd.workspace.bbox
        I = rgbd.crop_gray()
        hi, wi = I.shape[:2]
        xic, yic = float(wi) / 2, float(hi) / 2

        feature_map = self.get_feature_map(I, contour_min_len=self.args.fa.contour_min_len)
        timer.pin_time('get_feature')

        # target_T, Tx, Ty = self.get_translated_features(feature_map, self.args.fa.trans_range)
        target_T, Tx, Ty = self.get_translated_features_v2(feature_map, self.args.fa.trans_range)
        # target_R, Angle = self.get_rotated_features_v2(feature_map, self.args.fa.rot_range)
        timer.pin_time('translate')

        Intersect = np.dot(self.ref_R.astype('int'), target_T.astype('int'))
        # Intersect = np.dot(target_R.astype('int'), self.ref_T.astype('int'))
        arg_max = np.argmax(Intersect.todense())
        ww =Intersect.shape[1]
        arg_y = arg_max//ww
        arg_x = arg_max - ww*arg_y

        angle, tx, ty = self.Angle[arg_y], Tx[arg_x], Ty[arg_x]
        # angle, tx, ty = Angle[arg_y], self.Tx[arg_x], self.Ty[arg_x]

        print(f'Max intersect at Angle: {angle}, tx: {tx}, ty: {ty}')
        timer.pin_time('find_max_intersect')

        # display
        out = rgbd.disp()

        feature_map = (feature_map>127).astype('uint8')
        bound= np.bitwise_xor(feature_map, cv2.dilate(feature_map, np.ones((3,3), 'uint8')))
        Y,X = np.where(bound>0)
        out[(Y+top, X+left)] = (0,255,0)

        locs1d = np.where(self.ref_R[arg_y, :]>127)[0] - (tx + wi*ty)
        # locs1d = np.where(target_R[arg_y, :]>127)[0] - (tx + wi*ty)
        Y,X = self.locs1d2boundlocs(locs1d,hi,wi)
        out[(Y+top, X+left)] = (255, 0, 0)

        timer.pin_time('disp')

        # cv2.imshow('matched', out[:, :, ::-1])
        # cv2.waitKey()
        print(timer.pin_times_str())


        return {'im': out}


    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        if method_ind == 0:
            ret = self.get_show_noise_layer(rgbd=rgbd, disp_mode=disp_mode)
        if method_ind==1:
            # ret = self.get_reference_noise_mask(rgbd=rgbd)
            ret = self.get_contours_and_show(rgbd=rgbd)
        if method_ind==2:
            ret = self.get_feature_map_and_show(rgbd=rgbd)
        if method_ind==3:
            ret = self.get_reference_feature_maps(rgbd=rgbd)
        if method_ind==4:
            ret = self.match_ref_feature_to_im(rgbd=rgbd)


        return ret


def demo_gui():
    from ketisdk.gui.gui import GUI, GuiModule

    module = GuiModule(FocusAreaDetectorGui, cfg_path='kpick/apps/fpcb/configs/fpcb.cfg', num_method=6,
                       key_args=['fa.blur_ksize', 'fa.scale', 'fa.blur_thresh', 'fa.trans_range', 'fa.rot_range',
                                 'fa.test_rot_angle', 'fa.contour_min_len'])

    GUI(data_root='data/apps/fpcb/testset1', rgb_formats=['*', ], modules=[module, ],
        default_cfg_path='kpick/apps/fpcb/configs/default_fpcb.cfg')


if __name__ == '__main__':
    demo_gui()
