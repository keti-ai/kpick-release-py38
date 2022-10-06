import os.path
import torch
from kpick.object_detection.test import parse_args, Predictor
import numpy as np
import cv2
try:
    import nanodet
except:
    from kpick.object_detection.install import install_nanodet
    install_nanodet()

from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.data.collate import naive_collate
from nanodet.data.batch_process import stack_batch_img


__CHECKPOINT_LINKS__ = {'nanodet_m_box.ckpt':
                       'https://docs.google.com/uc?export=download&id=1UgTV95PZRErOFIu-wG0S5ubDxsDqrIQS'}

class BaseNanoDet():
    def get_model(self, cfg_file, ckpt, thresh=0.5, device="cuda:0"):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.thresh = thresh

        local_rank = 0
        self.cfg=cfg
        self.device=device
        load_config(self.cfg, cfg_file)
        logger = Logger(local_rank, use_tensorboard=False)

        if os.path.exists(ckpt):
            self.model = Predictor(cfg=self.cfg, model_path=ckpt, logger=logger, device=device)
        else:
            print(f'{ckpt} does not exist ...')
            save_dir, filename = os.path.split(ckpt)
            os.makedirs(save_dir, exist_ok=True)
            link = __CHECKPOINT_LINKS__[filename]
            print(f'Downloading checkpoint {filename} from {link}')
            os.system(f'wget --no-check-certificate "{link}" -O {ckpt}')
            print(f'Checkpoint saved at {ckpt} ...')
        print(f'{ckpt} loaded ... ')

    def run(self, im, thresh=0.5, filename='unnamed', classes=None, lefttop=(0, 0)):
        h, w = im.shape[:2]
        img_info = {
            'id': 0,
            'file_name': [filename],
            'height': [h],
            'width': [w],
        }

        # wrap_matrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        meta = dict(img_info=img_info, raw_img=im, img=im)

        meta = self.model.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        with torch.no_grad():
            ret = self.model.model.inference(meta)[0]

        out = dict()
        for name in ret:
            boxes = np.array(ret[name])
            if len(boxes) == 0: continue
            inds = np.where(boxes[:, -1] > thresh)[0].tolist()
            if len(inds) == 0: continue
            name_ = classes[name] if classes is not None else str(name)
            boxes = boxes[inds, :]
            if lefttop != (0, 0):
                left, top = lefttop
                boxes[:, [0, 2]] += left
                boxes[:, [1, 3]] += top
            out.update({name_: boxes})

        return out

from kpick.detector_base import BaseDetector


class NanodetDetector(BaseDetector, BaseNanoDet):
    def get_model(self, cfg_file, ckpt, thresh=0.5, device="cuda:0"):
        BaseNanoDet.get_model(self, cfg_file=cfg_file, ckpt=ckpt, thresh=thresh, device=device)

    def run(self, im, thresh=0.5, filename='unnamed', classes=None,lefttop=(0, 0)):
        return BaseNanoDet.run(self, im, thresh=thresh, filename=filename, classes=classes,
                               lefttop=lefttop)


from kpick.base.base import DetGuiObj
from ketisdk.utils.proc_utils import Timer


class NanoDetectorGui(NanodetDetector, DetGuiObj):
    def get_model(self):
        device = self.args.net.device if hasattr(self.args.net, 'device') else 'cuda:0'
        super().get_model(cfg_file=self.args.net.cfg_file, ckpt=self.args.net.ckpt,
                          thresh=self.args.net.score_thresh, device=device)

    def show_det(self, im, ret):
        out  =np.copy(im)
        for name in ret:
            boxes = ret[name]
            for box in boxes:
                left,top, right,bottom = box[:4].astype('int')
                lb = f'{name}:{round(box[-1],2)}'
                cv2.rectangle(out, (left,top), (right, bottom), self.args.disp.line_color, self.args.disp.line_thick)
                cv2.putText(out, lb, (left, top-5), cv2.FONT_HERSHEY_COMPLEX, self.args.disp.text_scale,
                            self.args.disp.text_color)
        return out


    def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', detected=None):
        if method_ind == 0:
            timer = Timer()
            # detect from cropped image
            # ret = self.run(rgbd.crop().bgr(), thresh=self.args.net.score_thresh, filename=filename,
            #                      lefttop=rgbd.workspace.bbox[:2], classes=self.args.net.class_names)
            ret = self.run(rgbd.crop_rgb(pad_val=(0,0,0))[:,:,::-1], thresh=self.args.net.score_thresh,
                           filename=filename, classes=self.args.net.class_names)
            print(f'Runtime: {timer.run_time_str()}')

            out = self.show_det(rgbd.disp(mode=disp_mode), ret)

        return {'im': out}


from ketisdk.utils.proc_utils import CFG


def test_nanodet_gui(cfg_file, ckpt, score_thresh=0.5, class_names=['unnamed',], run_realsense=False, device='cuda:0'):
    # detection module
    args = CFG()
    args.net = CFG()
    args.net.cfg_file = cfg_file
    args.net.score_thresh = score_thresh
    args.net.ckpt = ckpt
    args.net.device = device
    args.net.class_names = class_names

    from ketisdk.gui.gui import GUI, GuiModule
    module = GuiModule(NanoDetectorGui, type='Nanodet detector', category='detector', args=args)

    # realsense module
    realsense_modules = []
    if run_realsense:
        from ketisdk.sensor.realsense_sensor import get_realsense_modules
        realsense_modules += get_realsense_modules()

    #
    GUI(title='Detectron GUI', modules=[module, ] + realsense_modules)


if __name__ == '__main__':
    cfg_file = 'data/nanodet/configs/nanodet-m-box.yml'
    ckpt = 'data/nanodet/model/nanodet_m_box.ckpt'
    score_thresh = 0.5

    # NanodetDetector().get_model(cfg_file=cfg_file, ckpt=ckpt, thresh=score_thresh)
    test_nanodet_gui(cfg_file=cfg_file, ckpt=ckpt, score_thresh=score_thresh, class_names =['box',],  run_realsense=True)

# image --config data/nanodet/configs/nanodet-m-box.yml --model data/nanodet/model/nanodet_m_box.ckpt --path data/nanodet/coco_box/imgs/20211109143905856440.png
