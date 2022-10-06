
# register data set
from detectron2.data.datasets import register_coco_instances
register_coco_instances('InnerGripDataset_train', {}, 'data/apps/robotplus/train_data/set1/aug0/ann_aug_train.json',
                        'data/apps/robotplus/train_data/set1/aug0/imgs_aug')
register_coco_instances('InnerGripDataset_val', {}, 'data/apps/robotplus/train_data/set1/aug0/ann_aug_val.json',
                        'data/apps/robotplus/train_data/set1/aug0/imgs_aug')


# # visualize dataset
# from detectron2.data import MetadataCatalog, DatasetCatalog
# fruits_nuts_metadata = MetadataCatalog.get("SsamjangDataset_train")
# dataset_dicts = DatasetCatalog.get("SsamjangDataset_train")
# import cv2
# import random
# from detectron2.utils.visualizer import Visualizer
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow('im',vis.get_image()[:, :, ::-1])
#     cv2.waitKey()


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file(
    "configs/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("InnerGripDataset_train",)
cfg.DATASETS.TEST = ('InnerGripDataset_val',)  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = (
    10000
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (inner)
cfg.INPUT.MASK_FORMAT='bitmask'
cfg.OUTPUT_DIR = 'data/apps/robotplus/model/inner_grip'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
