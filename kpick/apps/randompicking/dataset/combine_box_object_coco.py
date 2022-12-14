from kpick.dataset.coco import run_coco_gui, concat_cocos, CocoUtils, replace_background

# run_coco_gui(
#     cfg_path='kpick/apps/randompicking/configs/combine_box_object_coco.cfg',
#     default_cfg_path='kpick/apps/randompicking/configs/box_object_data_default.cfg',
#     data_dir='data/object_box_coco_aug', rgb_formats=['imgs_aug/*']
# )
# concat_cocos(ann_files=['data/object_box_coco_aug/object_ann_aug.json', 'data/object_box_coco_aug/box_ann_aug.json'],
#              im_dirs=['data/object_box_coco_aug/imgs_aug', 'data/object_box_coco_aug/imgs_aug'],
#              out_file='data/object_box_coco_aug/concat_ann.json')

# CocoUtils('data/object_box_coco_aug/concat_ann.json').split_trainval()

# coco = CocoUtils('data/object_box_coco_aug/concat_ann_val.json')
# coco.show_ims(im_dir='data/object_box_coco_aug/imgs_aug')

# replace_background('data/object_box_coco_aug/concat_ann.json', im_dir='data/object_box_coco_aug/imgs_aug',
#                    background_path='data/object_box_coco_aug/bg00.png', select_stride=20)

CocoUtils('data/object_box_coco_aug/concat_ann_bg_train.json').show_ims(im_dir='data/object_box_coco_aug/imgs_aug')
# CocoUtils('data/object_box_coco_aug/concat_ann_bg_val.json').show_ims(im_dir='data/object_box_coco_aug/imgs_aug')
