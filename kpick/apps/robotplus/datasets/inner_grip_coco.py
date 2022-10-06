
from kpick.dataset.coco import run_coco_gui

if __name__=='__main__':
    run_coco_gui(cfg_path='apps/robotplus/configs/inner_grip.cfg',
                 default_cfg_path='apps/robotplus/configs/default_inner_grip.cfg',
                 data_dir='data/apps/robotplus/train_data/set1', rgb_formats=['imgs/*'],
                 key_args=['path.ann_files', 'path.im_dirs', 'save.aug_dirs'])