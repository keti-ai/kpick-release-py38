import os

from kpick.dataset.coco import run_coco_gui, CocoGui, CocoUtils
import kpick
KPICK_DIR = os.path.split(kpick.__file__)[0]

def make_db_gui():
    cfg_path = os.path.join(KPICK_DIR, 'apps/dataset/common_object_coco.cfg')
    default_cfg_path = os.path.join(KPICK_DIR, 'apps/dataset/common_object_coco_default.cfg')
    run_coco_gui(cfg_path=cfg_path, data_dir='data/maskrcnn', rgb_formats=['imgs/*'],
                 default_cfg_path=default_cfg_path)

def show_db():
    coco = CocoUtils(ann_path='data/object_coco_aug/ann_aug_val.json')
    coco.show_ims('data/object_coco_aug/imgs_aug')

if __name__=='__main__':
    show_db()
