import os

from kpick.dataset.coco import run_coco_gui
import kpick
KPICK_DIR = os.path.split(kpick.__file__)[0]

if __name__=='__main__':
    cfg_path = os.path.join(KPICK_DIR, 'apps/mujin/dataset/box_coco.cfg')
    default_cfg_path = os.path.join(KPICK_DIR, 'apps/mujin/dataset/box_coco_default.cfg')

    run_coco_gui(cfg_path=cfg_path, default_cfg_path=default_cfg_path,
                 data_dir='data/apps/mujin/box_coco/imgs', rgb_formats=['*'])