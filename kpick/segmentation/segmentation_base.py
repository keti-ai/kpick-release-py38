import numpy as np
import cv2


class Segmentator():
    def getFgMaskByDepth(self, depth, bg_depth, thresh=5, percent=1, diff_min=0, diff_max=100):
        ddepth = np.abs(depth.astype('float') - bg_depth.astype('float'))
        invalid_locs = np.where((depth == 0) | (bg_depth == 0))
        ddepth[invalid_locs] = 0

        if percent!=1:
            ddepth = np.clip(ddepth, a_min=diff_min, a_max=diff_max)
            aa = np.histogram(ddepth, bins=np.unique(ddepth))
            aa[0][:3] = 0  # diff is small --> remove

            hist = aa[0] / np.sum(aa[0])
            for i in range(1, len(hist)):
                hist[i] += hist[i - 1]
            ddepth_max = aa[1][np.where(hist <= 0.999)][-1]
            # low_locs = np.where(ddepth<percent*np.amax(ddepth))
            # ddepth[low_locs]=0
            ddepth[np.where(ddepth<0.999*ddepth_max)]=0


        return 255*(ddepth>thresh).astype('uint8')

    def normalizeDepth(self, depth, bg_depth, thresh=5):
        ddepth = depth.astype('float') - bg_depth.astype('float')
        invalid_locs = np.where((depth==0) | (bg_depth==0))
        ddepth[invalid_locs] = 0

        udepth = np.mean(bg_depth)
        h,w = depth.shape[:2]

        out = udepth + np.zeros((h,w), 'float')
        out = np.maximum(out + ddepth, 0)
        return out.astype('uint16')
