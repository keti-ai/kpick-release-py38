import numpy

from kpick.dataset.via_dataset import ViaAnnotation, get_via_annotation_default_args
from kpick.dataset.coco import CocoUtils
from kpick.base.base import DetGuiObj
from ketisdk.utils.proc_utils import ProcUtils, CFG
import cv2
import math
import numpy as np
import os
from PIL import Image, ImageDraw


def get_kimchi_via_ann_default_args():
    args = get_via_annotation_default_args()

    args.dataset = CFG()
    args.dataset.ann_json = 'data/apps/kimchi/20220629/kimchi_project_20220629_json.json'

    args.dataset.im_dir = 'data/apps/kimchi/20220629/imgs'
    args.dataset.classes = ['kimchi', 'angle']
    args.dataset.angle_step = 10

    args.path.save_dirs = ['data/apps/kimchi/20220629/aug2', ]
    args.disp.class_colors = [(0, 255, 0), (0, 0, 255)]

    # args.aug = None
    args.aug = CFG()
    args.aug.brightnesss = [0.3, ]
    args.aug.contrasts = [0.3, ]
    args.aug.saturations = [0.1, ]
    args.aug.hues = [0.1, ]
    args.aug.angle_range = (0, 360, 5)
    args.aug.flip_axes = [-1,]

    return args

def orientedBox2Points(box, angle):
    left, top, right, bottom = box
    xc, yc = (left+right)/2, (top+bottom)/2
    w, h = right-left, bottom-top
    r = math.sqrt(w*w + h*h)/2
    dx, dy = ProcUtils().rotateXY(-r, 0, angle)
    x0, y0 = int(xc + dx), int(yc + dy)
    dx, dy = ProcUtils().rotateXY(r, 0, angle)
    x1, y1 = int(xc + dx), int(yc + dy)
    return (x0, y0), (x1, y1)


class KimchiViaDataset(ViaAnnotation, CocoUtils):
    def points2Mask(self, X, Y, im_size):
        poly = [(x, y) for x, y in zip(X, Y)]
        height, width = im_size
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
        return numpy.array(img)

    def draw_region(self, im, X, Y, disp_args, color=(0, 255, 0)):
        num_el = len(X)
        if num_el > 2:
            # return ViaAnnotation.draw_region(self, im, X,Y, disp_args, color)
            mask = self.points2Mask(X, Y, im.shape[:2])
            out = im.copy()
            locs = np.where(mask > 0)
            out[locs] = 0.7 * out[locs] + (0, 75, 0)
            return out

        out = im.copy()
        angle = self.get_angle((X[0], Y[0]), (X[1], Y[1]))
        label = self.angle2LabelInd(angle)
        cv2.putText(out, f'{angle:0.1f}->{label}', (X[0], Y[0]), cv2.FONT_HERSHEY_COMPLEX, disp_args.text_scale,
                    disp_args.text_color, disp_args.text_thick)
        cv2.line(out, (X[0], Y[0]), (X[1], Y[1]), (0,255,0), 2)

        #
        angle1 = self.labelInd2Angle(label)
        left, right = min(X[0], X[1]), max(X[0], X[1])
        top, bottom = min(Y[0], Y[1]), max(Y[0], Y[1])
        pt0, pt1 = orientedBox2Points((left, top, right, bottom), angle1)
        cv2.line(out, pt0, pt1, (255, 0, 0), 2)

        print(f'Angle Encoding Error: {(angle1-angle):>.2f}')

        return out

    def get_angle(self, pt0, pt1):
        dx, dy = pt1[0] - pt0[0], pt1[1] - pt0[1]
        angle = math.atan(dy / (dx + 0.000001)) * 180 / math.pi
        if (dy * angle) < 0:
            angle += 180 if dy > 0 else -180
        if angle < 0: angle += 360
        return angle

    def angle2LabelInd(self, angle, step=10):
        return np.clip(int(round(angle / step)) + 1, a_min=1, a_max=360 // step)

    def labelInd2Angle(self, ind, step=10):
        return ind * step - step // 2


class KimchiViaDatasetGuiObj(KimchiViaDataset, DetGuiObj):
    def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
        DetGuiObj.__init__(self, args=args, cfg_path=cfg_path, name=name,
                           default_args=get_kimchi_via_ann_default_args())
        self.classes = [f'{l}' for l in range(0, 360, self.args.dataset.angle_step)]
        KimchiViaDataset.init(self, ann_json=self.args.dataset.ann_json, im_dir=self.args.dataset.im_dir,
                              classes=self.classes)

    def init_acc(self):
        KimchiViaDataset.init_acc(self, save_dirs=self.args.path.save_dirs, classes=self.classes,
                                  aug_args=self.args.aug)

    def aug_category(self, cat_angle, angle, flip_axis):
        # cat_angle = self.labelInd2Angle(category_id, step=self.args.dataset.angle_step)
        cat_angle -= angle
        if cat_angle > 360: cat_angle -= 360
        if cat_angle < 0: cat_angle += 360

        if flip_axis == 0: cat_angle = -cat_angle
        if flip_axis == 1: cat_angle = 180 - cat_angle
        if cat_angle > 360: cat_angle -= 360
        if cat_angle < 0: cat_angle += 360

        return self.angle2LabelInd(cat_angle, step=self.args.dataset.angle_step)


    def make_bd(self, rgbd, filename):
        # out = rgbd.disp(mode=disp_mode)
        ret = self.get_regions(filename=filename)
        masks, category_ids, angles = [], [], []
        for reg, ind in zip(ret['regions'], ret['indexes']):
            if len(reg[0]) > 2:
                masks.append(self.points2Mask(reg[0], reg[1], rgbd.rgb.shape[:2]))
            else:
                category_ids.append(self.angle2LabelInd(self.get_angle((reg[0][0], reg[1][0]), (reg[0][1], reg[1][1])),
                                                        step=self.args.dataset.angle_step))
                angles.append(self.get_angle((reg[0][0], reg[1][0]), (reg[0][1], reg[1][1])))

        left, top, right, bottom = rgbd.workspace.bbox
        if self.args.aug is not None:
            KimchiViaDataset.acc_augmentation_db(self, rgb=rgbd.crop_rgb(), category_ids=angles,#category_ids=category_ids,
                                                 masks=[m[top:bottom, left:right] for m in masks])
        else:
            KimchiViaDataset.acc_single_db(self, rgb=rgbd.crop_rgb(), category_ids=category_ids,
                                           masks=[m[top:bottom, left:right] for m in masks])

    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
        if method_ind == 0:
            ret = self.show_single_ann(rgbd=rgbd, filename=filename, disp_args=self.args.disp, disp_mode=disp_mode)
        if method_ind == 1:
            ret = self.make_bd(rgbd=rgbd, filename=filename)

        return ret


def demo_grip_via_annocation_gui(cfg_path='kpick/apps/kimchi/configs/kimchi_mask_angle.cfg',
                                 default_cfg_path='kpick/apps/kimchi/configs/kimchi_default.cfg',
                                 data_root=None, rgb_formats=None, depth_formats=None):
    from ketisdk.gui.gui import GUI, GuiModule

    module = GuiModule(KimchiViaDatasetGuiObj, name='KimchiAnn',
                       cfg_path=cfg_path, num_method=3,
                       key_args=['dataset.ann_json', 'dataset.im_dir']
                       )

    GUI(title='Kimchi Via Ann GUI', default_cfg_path=default_cfg_path,
        modules=[module], data_root=data_root, rgb_formats=rgb_formats, depth_formats=depth_formats,
        )


class OrientedCoCo(CocoUtils):
    def visualize_instance(self, rgb, instance, categories=None):
        out = CocoUtils.visualize_instance(self, rgb, instance, categories)
        # bbox
        if 'bbox' in instance:
            bbox = instance['bbox']
            left, top, w, h = np.array(bbox).astype('int')
            r = math.sqrt(w * w + h * h) / 2
            xc, yc = left + w/2, top + h/2

            category_id = instance['category_id']
            if category_id is not None:
                for cat in self.categories:
                    if cat['id'] == category_id:
                        # cv2.putText(out, cat['name'], (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        angle = (int(category_id)-0.5) * 10
                        print(f'{category_id}->{angle}')
                        dx, dy = ProcUtils().rotateXY(-r, 0, angle)
                        x0, y0 = int(xc + dx), int(yc + dy)
                        dx, dy = ProcUtils().rotateXY(r, 0, angle)
                        x1, y1 = int(xc + dx), int(yc + dy)
                        cv2.line(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.drawMarker(out, (x0, y0), (0, 2550, 0), cv2.MARKER_TILTED_CROSS, 15, 2)
        return out

if __name__ == '__main__':
    # KimchiViaDatasetGuiObj()
    # demo_grip_via_annocation_gui(data_root='data/apps/kimchi/20220629', rgb_formats=['imgs/*'])
    coco = OrientedCoCo(ann_path='data/apps/kimchi/20220629/aug2/ann_aug_train.json')
    coco.show_ims(im_dir='data/apps/kimchi/20220629/aug2/imgs')
