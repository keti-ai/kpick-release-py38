from ketisdk.import_basic_utils import *
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

def get_detectron2_default_args():
    from ketisdk.gui.default_config import default_args
    from ketisdk.utils.proc_utils import CFG
    args = default_args()

    args.net = CFG()
    args.net.model_cfg_path = 'configs/mask_rcnn_R_50_FPN_3x.yaml'
    args.net.model_path = 'data/apps/kimchi/model/mask_angle/model_final.pth'
    args.net.num_classes = 36
    args.net.score_thresh = 0.5
    args.net.kpt_skeleton = None

    return args


class BaseDetectron():
    def get_model(self, model_cfg_path, model_path, num_classes=1, score_thresh=0.5):
        cfg = get_cfg()
        cfg.merge_from_file(model_cfg_path)
        # cfg.DATASETS.TEST = self.args.test_dataset
        # cfg.DATALOADER.NUM_WORKERS = self.args.num_workers
        # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.args.batch_size_per_image
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.args.num_classes  # only has one class (ballon)
        # cfg.MODEL.ANCHOR_GENERATOR.SIZES = self.args.anchor_sizes
        # cfg.MODEL.WEIGHTS = self.args.checkpoint
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.args.score_thresh_test

        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = score_thresh
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.freeze()

        model = DefaultPredictor(cfg)
        return model

    def run(self, im, model, thresh=None, classes=None):
        pred = model(im)
        out = dict()
        if 'instances' in pred:
            inst = pred["instances"].to("cpu")
            if inst.has("pred_boxes"): out.update({'boxes': inst.pred_boxes.tensor.numpy()})
            if inst.has("scores"): out.update({'scores': inst.scores.numpy().reshape((-1, 1))})
            if inst.has("pred_classes"): out.update({'classes': inst.pred_classes.numpy()})
            if inst.has("pred_keypoints"): out.update({'keypoints': inst.pred_keypoints.numpy()})
            if inst.has("pred_masks"): out.update({'masks': inst.pred_masks.numpy()})
        if 'sem_seg' in pred:
            sem_seg = pred["sem_seg"].argmax(dim=0).to("cpu").numpy()
            out.update({'sem_seg': sem_seg})
        return out

    def run_extract(self, predictions):
        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores.numpy() if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints.numpy() if predictions.has("pred_keypoints") else None


from kpick.detector_base import BaseDetector
class DetectronDetector(BaseDetector, BaseDetectron):
    def get_model(self, model_cfg_path, model_path, num_classes=1, score_thresh=0.5):
        return BaseDetectron.get_model(self, model_cfg_path=model_cfg_path,
                                       model_path=model_path, num_classes=num_classes,
                                       score_thresh=score_thresh)

    def run(self, im, model, thresh=None, classes=None):
        return BaseDetectron.run(self, im, model, thresh=thresh, classes=classes)

    def predict_single(self, rgbd, model):
        timer = Timer()
        # detect from cropped image
        ret = self.predict_array(rgbd.crop().bgr(), model=model)
        print(f'Runtime: {timer.run_time_str()}')

        # invert cropping
        ret = self.uncrop_predict( predict=ret, im_size=(rgbd.height, rgbd.width),
                                      bbox=rgbd.workspace.bbox)

        return ret

        # # show detection in cropped image
        # out = self.show_predict(self, rgbd.disp(mode=disp_mode), ret, self.args.net.kpt_skeleton)
        # return {'im': out}



def get_detectron_obj(Detector=DetectronDetector):
    from ketisdk.vision.base.base_objects import BasObj
    class DetectronDetectorObj(Detector, BasObj):
        def __init__(self, args=None, cfg_path=None, name='unnamed', default_args=None):
            BasObj.__init__(self, args=args, cfg_path=cfg_path, name=name, default_args=get_detectron2_default_args())
            self.model = Detector.get_model(self, model_cfg_path=self.args.net.model_cfg_path,
                                            model_path=self.args.net.model_path,
                                            num_classes=self.args.net.num_classes,
                                            score_thresh=self.args.net.score_thresh)

        def load_params(self, args):
            BasObj.load_params(self, args=args)


        def predict_show_single(self, rgbd, disp_mode='rgb'):
            ret = Detector.predict_single(self,rgbd, model=self.model)
            out = self.show_predict(rgbd.disp(mode=disp_mode), ret, self.args.net.kpt_skeleton)
            return {'im':out}

    return DetectronDetectorObj

def get_detectron_gui_obj(DetectorObj):
    from kpick.base.base import DetGui
    class DetectronDetectorGuiObj(DetectorObj, DetGui):

        def gui_process_single(self, rgbd, method_ind=0, filename='unnamed', disp_mode='rgb', **kwargs):
            if method_ind == 0:
                ret = self.predict_show_single(rgbd, disp_mode=disp_mode)

            return ret

    return DetectronDetectorGuiObj


def demo_detectron_gui(model_cfg_path=None, model_path=None, num_classes=None,
                       cfg_path=None, default_cfg_path=None,
                       Detector=DetectronDetector, score_thresh=0.5,
                       kpt_skeleton=None, key_args=None, sensors = ['realsense',],
                       data_root=None, rgb_formats=None, depth_formats=None):
    # detection module
    args = get_detectron2_default_args()
    if model_cfg_path is not None: args.net.model_cfg_path = model_cfg_path
    if model_path is not None: args.net.model_path = model_path
    if num_classes is not None: args.net.num_classes = num_classes
    if kpt_skeleton is not None: args.net.kpt_skeleton = kpt_skeleton
    args.net.score_thresh = score_thresh


    from ketisdk.gui.gui import GUI, GuiModule
    module = GuiModule(get_detectron_gui_obj(DetectorObj=get_detectron_obj(Detector)), cfg_path=cfg_path,
                       type='detectron detector', category='detector', args=args,key_args=key_args)
    modules = [module,]

    # realsense module
    if 'realsense' in sensors:
        from ketisdk.sensor.realsense_sensor import get_realsense_modules
        modules += get_realsense_modules()

    #
    GUI(title='Detectron GUI', modules=modules, default_cfg_path=default_cfg_path,
        data_root=data_root, rgb_formats=rgb_formats, depth_formats=depth_formats)

if __name__=='__main_':
    # get_detectron_gui_obj(get_detectron_obj())
    demo_detectron_gui()


