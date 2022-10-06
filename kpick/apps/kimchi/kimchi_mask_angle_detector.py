from kpick.detectron2_detector import demo_detectron_gui, DetectronDetector,get_detectron_gui_obj
from ketisdk.utils.proc_utils import ProcUtils
import math
import cv2

class KimChiDetector(DetectronDetector):
    def show_predict(self, im, predict, kpt_skeleton=None, colors=None, line_thick=2, text_scale=1, text_thick=2, marker_size=5):
        out =DetectronDetector.show_predict(self,im, predict, kpt_skeleton, colors, line_thick, text_scale, text_thick, marker_size)
        boxes = predict['boxes'] if 'boxes' in predict else None
        classes = predict['classes'] if 'classes' in predict else None
        for box, cl in zip(boxes, classes):
            left, top, right, bottom = box
            xc, yc = (left+right)/2, (top+bottom)/2
            w, h = right-left, bottom-top
            r = math.sqrt(w*w + h*h)/2
            angle = int(cl)*10
            dx, dy = ProcUtils().rotateXY(-r,0, angle)
            x0, y0 = int(xc+dx), int(yc+dy)
            dx, dy = ProcUtils().rotateXY(r, 0, angle)
            x1, y1 = int(xc + dx), int(yc + dy)

            cv2.line(out, (x0,y0), (x1, y1), (0,255,0),2)
            cv2.drawMarker(out,(x0,y0),(0,2550,0), cv2.MARKER_TILTED_CROSS, 15,2)

            print(f'{angle}: {float(predict["scores"][0]):>0.3f}')
        return out






# model_cfg_path='configs/mask_rcnn_R_50_FPN_3x.yaml'
# model_path='data/apps/kimchi/model/mask_angle_aug1/model_final.pth'

model_cfg_path='configs/mask_rcnn_R_101_FPN_3x.yaml'
model_path='data/apps/kimchi/model/mask_angle_aug1_R101/model_final.pth'

# model_cfg_path='configs/faster_rcnn_R_50_FPN_3x.yaml'
# model_path='data/apps/kimchi/model/angle/model_final.pth'


# detector = BaseDetectron()
# detector.get_model(model_cfg_path=model_cfg_path, model_path=model_path, score_thresh=0.5)

demo_detectron_gui(model_cfg_path=model_cfg_path, model_path=model_path,
                   cfg_path='kpick/apps/kimchi/configs/det.cfg',
                   default_cfg_path='kpick/apps/kimchi/configs/default.cfg',
                   sensors=['realsense',],
                   data_root='data/apps/kimchi/20220629',
                   rgb_formats=['imgs/*'],
                   Detector=KimChiDetector,
                   )