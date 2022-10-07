import numpy as np
from ketisdk.import_basic_utils import *

from torchvision.transforms import transforms
from kpick.classifier.roi_classifier import RoiCifarClassfier
import cv2
import os
from sklearn.cluster import KMeans
from ketisdk.utils.proc_utils import ArrayUtils

class BaseGraspDetector(RoiCifarClassfier):

    def load_params(self, net_args):
        # super().load_params(args=args)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(net_args.db_mean, net_args.db_std), ])
        if hasattr(net_args,'bg_depth_file'):

            if os.path.exists(net_args.bg_depth_file):
                self.bg_depth = cv2.blur(cv2.imread(net_args.bg_depth_file, cv2.IMREAD_UNCHANGED),
                                         ksize=net_args.depth_blur_ksize).astype('float')
                print(f'Background: {net_args.bg_depth_file} loaded...')
            else:
                self.bg_depth=None
                print(f'Background: {net_args.bg_depth_file} does not exist...')

    def get_background_depth_map(self, rgbd, net_args):
        self.bg_depth = cv2.blur(rgbd.depth, ksize=net_args.depth_blur_ksize)
        cv2.imwrite(net_args.bg_depth_file, self.bg_depth)
        self.bg_depth = self.bg_depth.astype('float')
        print('Background depth stored ...')

    def get_workspace_corner_depth(self, rgbd, net_args):
        # pts_array = np.array(rgbd.workspace.pts).reshape((-1,2))
        rx, ry = net_args.depth_blur_ksize[0]//2, net_args.depth_blur_ksize[1]//2
        dX, dY = np.arange(-rx,rx+1), np.arange(-ry, ry+1)

        corner_depth_map = []
        for pt in rgbd.workspace.pts:
            X, Y = pt[0]+dX, pt[1]+dY
            X, Y = X[(0<=X)&(X<rgbd.width)], Y[(0<=Y)&(Y<rgbd.height)]
            Y,X = np.meshgrid(Y,X)
            X,Y = X.flatten(), Y.flatten()
            corner_depth_map.append(pt + (np.mean(rgbd.depth[(Y,X)]),))
        corner_depth_map = np.array(corner_depth_map)
        print('Corner depth map stored ...')
        return corner_depth_map

    def get_fg_depth_inds(self, rgbd, locs, net_args):
        # zlocs = list(range(len(locs[0])))
        try:
            depth_blur = cv2.blur(rgbd.depth, ksize=net_args.depth_blur_ksize)

            Zb = depth_blur[locs].reshape((-1, 1)).astype('float')
            Zbg = self.bg_depth[locs].reshape((-1, 1))

            zlocs = np.where((np.abs(Zb - Zbg) > net_args.bg_depth_diff) & (Zb>0) & (Zbg>0))[0].tolist()
            num_grasp = len(zlocs)
            print(f'{num_grasp} candidates: depth different > {net_args.bg_depth_diff}')
        except:
            zlocs = list(range(len(locs[0])))

            # # if not hasattr(self, 'corner_depth_map'): self.get_workspace_corner_depth(rgbd)
            # reference_depth_map = self.get_workspace_corner_depth(rgbd)
            # # reference_depth_map = np.array(self.args.reference_depth_map)
            # num_grasp, num_corner = len(locs[0]), len(reference_depth_map)
            # corner_loc_map = ArrayUtils().repmat(reference_depth_map[:, :2], (1, 1, num_grasp))
            # suc_loc_map = ArrayUtils().repmat(
            #     np.concatenate((locs[1].reshape(1, 1, -1), locs[0].reshape(1, 1, -1)), axis=1),
            #     (num_corner, 1, 1)).astype('float')
            # dmap = corner_loc_map - suc_loc_map
            # dmap = np.linalg.norm(dmap, axis=1)
            # inv_dmap = np.max(dmap) - dmap
            # dsum = ArrayUtils().repmat(np.sum(inv_dmap, axis=0).reshape((1, -1)), (num_corner, 1)) + 0.00001
            # W = np.divide(inv_dmap, dsum)
            # ZZ = np.multiply(W, ArrayUtils().repmat(reference_depth_map[:, -1].reshape((-1, 1)), (1, num_grasp)))
            # ZZ = np.sum(ZZ, axis=0).reshape((-1, 1))
            #
            # depth_blur = cv2.blur(rgbd.depth, ksize=net_args.depth_blur_ksize)
            # Zb = depth_blur[locs].reshape((-1, 1)).astype('float')
            # zlocs = np.where(np.abs(Zb - ZZ) > net_args.bg_depth_diff)[0].tolist()
            # num_grasp = len(zlocs)
            # print(f'{num_grasp} candidates: depth different > {net_args.bg_depth_diff}')
        return zlocs

    def get_high_score_inds(self, Score, thresh):
        return np.where(Score>thresh)[0].flatten().tolist()

    def get_low_score_inds(self, Score, thresh):
        return np.where(Score<=thresh)[0].flatten().tolist()

    def scoreSpatial(self, workspace, Center):
        return ArrayUtils().normalize(1 - np.clip(workspace.center_bias(pts=Center), 0,1).reshape((-1,1)), denoise=True)

    def scoreDepth(self, Depth, depth_max):
        # valid_depth = (Depth > 100).astype('float').reshape((-1,1))
        Score = 1 - np.clip(Depth / depth_max, 0, 1)
        return Score.reshape((-1,1))

    def scoreValidDepth(self, Depth):
        return (Depth> 100).astype('float').reshape((-1,1))

    def show_suctions(self, rgbd, Suction, net_args, args, disp_mode='rgb', out=None, best_ind=None, best_n_inds=None,
                      high_score_inds=None, low_score_inds=None):
        if out is None: out = rgbd.disp(mode=disp_mode)
        if Suction is not None:
            if args.flag.show_steps:
                if low_score_inds is None: low_score_inds= self.get_low_score_inds(Suction[:, -1], net_args.score_thresh)
                for suc in Suction[low_score_inds, :]:
                    xc, yc = suc[:2].astype('int')
                    cv2.drawMarker(out, (xc, yc), (0, 0, 255), cv2.MARKER_DIAMOND, args.disp.marker_size // 2, 1)

                if high_score_inds is None: high_score_inds = self.get_high_score_inds(Suction[:, -1], net_args.score_thresh)
                for suc in Suction[high_score_inds, :]:
                    xc, yc = suc[:2].astype('int')
                    cv2.drawMarker(out, (xc, yc), (0, 255, 0), cv2.MARKER_DIAMOND, args.disp.marker_size // 2, 1)

            if best_n_inds is None: best_n_inds = self.select_best_n_inds(rgbd, Suction, net_args)
            for suc in Suction[best_n_inds, :]:
                xc, yc = suc[:2].astype('int')
                cv2.drawMarker(out, (xc, yc), (255, 0, 100), cv2.MARKER_DIAMOND, args.disp.marker_size // 2, 1)

            if best_ind is None:
                best_ind = self.select_best_ind(rgbd, Suction, net_args) if best_n_inds is None else best_n_inds[0]

            xc, yc = Suction[best_ind, :2].flatten().astype('int')
            cv2.drawMarker(out, (xc, yc), (255, 0, 0), cv2.MARKER_TILTED_CROSS, args.disp.marker_size,
                           args.disp.marker_thick)
        return out

    def show_poses(self, rgbd, net_args, args, disp_mode='rgb', detected=None):
        if 'im' not in detected:out = rgbd.disp(mode=disp_mode)
        else:out = detected['im']
        best_poses, pose_types = [], []
        if 'suction' in detected:
            Suction = detected['suction']
            if Suction is not None:
                self.show_suctions(rgbd=rgbd, Suction=Suction, disp_mode=disp_mode, out=out, net_args=net_args, args=args)
        if 'grip' in detected:
            Grip = detected['grip']
            if Grip is not None:
                self.show_grips(rgbd=rgbd, Grip=Grip, disp_mode=disp_mode, out=out, net_args=net_args, args=args)

        # if 'im' not in detected: detected.update({'im': out})
        # else: detected['im'] = out
        return out

    def getEllipsMask(self, Center, net_args,Score=None, Angle=0, axes=(20, 50)):
        Nc = len(Center)

        # Y,X = np.where(mask)
        # Y, X = Y.reshape((1,-1)), X.reshape((1,-1))
        if Score is None:
            X, Y = Center[:, 0].reshape((1, -1)), Center[:, 1].reshape((1, -1))
        else:
            inds = self.get_high_score_inds(Score, net_args.score_thresh)
            X, Y = Center[inds, 0].reshape((1, -1)), Center[inds, 1].reshape((1, -1))

        Theta = Angle / 180 * np.pi
        EllipseArea = np.pi * axes[0] * axes[1]

        Sint, Cost = np.sin(Theta), np.cos(Theta)
        if Score is None:
            Xc, Yc = Center[:, 0].reshape((-1, 1)), Center[:, 1].reshape((-1, 1))
        else:
            Xc, Yc = Center[inds, 0].reshape((-1, 1)), Center[inds, 1].reshape((-1, 1))

        Y_, X_ = Y - Yc, X - Xc
        a2, b2 = axes[0]*axes[0], axes[1]*axes[1]

        ElVal1 = np.square(np.multiply(X_, Cost) + np.multiply(Y_, Sint))/ a2
        ElVal2 = np.square(np.multiply(X_, Sint) - np.multiply(Y_, Cost)) / b2
        ElVal = ElVal1 + ElVal2

        InSideElMask = np.zeros((Nc,), 'uint8')
        InSideElMask[inds]  = ElVal <= 1



        return InSideElMask

    def scoreInsideEllipse(self, Center, net_args, Score=None, Angle=0, axes=(20, 50)):
        Nc = len(Center)

        # Y,X = np.where(mask)
        # Y, X = Y.reshape((1,-1)), X.reshape((1,-1))
        if Score is None:
            X, Y = Center[:, 0].reshape((1, -1)), Center[:, 1].reshape((1, -1))
        else:
            inds = self.get_high_score_inds(Score, net_args.score_thresh)
            X, Y = Center[inds, 0].reshape((1, -1)), Center[inds, 1].reshape((1, -1))

        Theta = Angle / 180 * np.pi
        # EllipseArea = np.pi * axes[0] * axes[1]

        Sint, Cost = np.sin(Theta), np.cos(Theta)
        if Score is None:
            Xc, Yc = Center[:, 0].reshape((-1, 1)), Center[:, 1].reshape((-1, 1))
        else:
            Xc, Yc = Center[inds, 0].reshape((-1, 1)), Center[inds, 1].reshape((-1, 1))

        Y_, X_ = Y - Yc, X - Xc
        a2, b2 = axes[0] * axes[0], axes[1] * axes[1]

        ElVal1 = np.square(np.multiply(X_, Cost) + np.multiply(Y_, Sint)) / a2
        ElVal2 = np.square(np.multiply(X_, Sint) - np.multiply(Y_, Cost)) / b2
        ElVal = ElVal1 + ElVal2

        InSideElMask= ElVal <= 1
        InSideRate = np.sum(InSideElMask, axis=1, keepdims=True)
        InSideRate = InSideRate / np.amax(InSideRate)

        out = np.zeros((Nc,1), 'uint8')
        if Score is not None: out[inds, :] = InSideRate
        else: out = InSideRate
        return out

    def scoreCenterMoment(self, InSideElMask, EllipseArea):
        SumInSideEl = np.sum(InSideElMask, axis=1).reshape((-1,1))
        InSideRate = SumInSideEl/EllipseArea
        return InSideRate/np.amax(InSideRate)

    def getGroupPose(self, Center, Angle=None, nClusters=10):
        kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(Center)
        gPose = []
        for i in range(nClusters):
            inds = np.where(kmeans.labels_==i)[0].tolist()
            if len(inds)==0: continue
            X,Y = Center[inds,0], Center[inds, 1]
            left, top = np.amin(X), np.amin(Y)
            right, bottom = np.amax(X), np.amax(Y)
            xc, yc = np.mean(X), np.mean(Y)
            n = len(X)

            if Angle is None:
                a, b = ArrayUtils().linear_regression(X,Y)
                angle = np.arctan(a)*180/np.pi
            else:
                angle = np.mean(Angle[inds,:]) if isinstance(Angle, np.ndarray) else Angle
                a = np.tan(angle/180*np.pi)
                b = yc - a*xc

            if abs(a)<1000:
                x0, x1 = xc-10, xc+10
                y0, y1 = a*x0+b, a*x1+b
            else:
                x0, x1 = xc, xc
                y0, y1 = yc-10, yc+10


            gPose.append((xc,yc,angle, left, top, right, bottom, x0, y0, x1, y1, n))

        gPose = np.array(gPose)
        sortIds = np.argsort(gPose[:,-1])[::-1].tolist()
        return gPose[sortIds, :]







































