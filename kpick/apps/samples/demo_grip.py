def demo_grip_gui(cfg_path=None, default_cfg_path='configs/default.cfg'):
    from kpick.pick.grip.grip_detection import GripGuiDetector
    # from kpick.pick.grip_inner.grip_inner import GripInnerDetectorGui
    from ketisdk.sensor.realsense_sensor import get_realsense_modules
    from ketisdk.gui.gui import GUI, GuiModule

    detect_module0 = GuiModule(GripGuiDetector, type='grip_detector', name='Grip Detector',
                               category='detector', cfg_path=cfg_path, num_method=6,
                               key_args=['grip_net.grip_w_ranges', 'grip_net.depth_grad_thresh', 'grip_net.ellipse_axes',
                                         'grip_net.erode_h', 'grip_net.dy', 'grip_net.remove_bg']
                               )
    # cfg_path = 'configs/grip_inner.cfg'
    # detect_module1 = GuiModule(GripInnerDetectorGui, type='grip_inner_detector', name='Grip Inner Detector',
    #                            category='detector', cfg_path=cfg_path, num_method=5)

    # GripGuiDetector(cfg_path=cfg_path)

    GUI(title='Grip Detection GUI', default_cfg_path=default_cfg_path,
        modules=[detect_module0] + get_realsense_modules(),
        )


def demo_grip_without_gui(cfg_path, rgb_path, depth_path, ws_pts=None):
    from kpick.pick.grip.grip_detection import GripDetector
    import cv2
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    # Set viewer
    cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('viewer', 1080, 720)

    # Load image
    rgb = cv2.imread(rgb_path)[:, :, ::-1]
    depth = cv2.imread(depth_path)[:, :, 0]
    rgbd = RGBD(rgb=rgb, depth=depth)

    # workspace_pts = [(550, 100), (840, 100), (840, 600), (550, 600)]
    if ws_pts is not None: rgbd.set_workspace(pts=ws_pts)

    # Initialize network
    detector = GripDetector(cfg_path=cfg_path)

    # Run
    ret = detector.detect_and_show_poses(rgbd=rgbd)

    # Show
    cv2.imshow('viewer', ret['im'][:, :, ::-1])
    if cv2.waitKey() == 27: exit()


def demo_grip_and_inner_grip_without_gui(rgb_path, depth_path, ws_pts=None):
    from kpick.pick.grip.grip_detection import GripDetector
    from kpick.pick.grip_inner.grip_inner import GripInnerDetector
    import cv2
    from ketisdk.vision.utils.rgbd_utils_v2 import RGBD

    # Set viewer
    cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('viewer', 1080, 720)

    # Load image
    rgb = cv2.imread(rgb_path)[:, :, ::-1]
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgbd = RGBD(rgb=rgb, depth=depth)

    # workspace_pts = [(550, 100), (840, 100), (840, 600), (550, 600)]
    if ws_pts is not None: rgbd.set_workspace(pts=ws_pts)

    # Initialize network
    gripDetector = GripDetector(cfg_path='configs/pick/grip_net.cfg')
    gripInnerDetector = GripInnerDetector(cfg_path='configs/pick/grip_inner.cfg')

    # Run
    gripRet = gripDetector.detect_and_show_poses(rgbd=rgbd)
    gripInnerRet = gripInnerDetector.get_grip_and_show(rgbd=rgbd, out=gripRet['im'])

    # grips = gripRet['grips']
    # grips_inner = gripInnerRet['grips']

    # Show
    cv2.imshow('viewer', gripInnerRet['im'][:, :, ::-1])
    if cv2.waitKey() == 27: exit()


def test_tensorrt_grip():
    from kpick.classifier import cifar_classification_models as models
    from time import time
    import torch
    from torch2trt import trt, torch2trt
    from torchvision.models import resnet18

    # Load check point

    t = time()
    torch.device("cuda:0")
    model = models.__dict__['resnet'](
        num_classes=2,
        depth=20,
        input_shape=(32, 128, 6)
    )
    model = torch.nn.DataParallel(model)
    checkpoint_path = 'data/model/pick/grip_evaluator/resnet20_32x128x6_20200624/GripCifar10-resnet-model_best.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module

    input_shape = (6, 32, 128)
    # model = resnet18(pretrained=True)
    # input_shape = (3, 224, 224)

    print(f'Torch model loaded in {time() - t}')

    batch_size = 64
    x = torch.randn((batch_size,) + input_shape).cuda()

    model.eval().cuda()

    # convert  to tensorRT
    t = time()
    model_trt = torch2trt(model, [x], max_batch_size=batch_size,
                          fp16_mode=True, default_device_type=trt.DeviceType.DLA, dla_core=0)
    print(f'TensorRT model converted in {time() - t}')

    # ================================= Execute
    t = time()
    y = model(x)
    print(f'Torch executed in {time() - t}')

    t = time()
    y_trt = model_trt(x)
    print(f'TensorRT executed in {time() - t}')  # check the output against PyTorch

    print(f'output max: {torch.max(torch.abs(y))}')
    print(f'output trt max: {torch.max(torch.abs(y_trt))}')
    print(f'diff max: {torch.max(torch.abs(y - y_trt))}')


def test_ellipse():
    import numpy as np
    import math
    import cv2
    xmin, xmax = 100, 500
    ymin, ymax = 100, 300
    h, w = 400, 600

    a, b = (xmax - xmin) / 8, (ymax - ymin) / 8
    xc, yc = (xmin + xmax) // 2, (ymin + ymax) // 2
    X, Y = np.meshgrid(range(xmin, xmax), range(ymin, ymax))
    Y, X = Y.reshape((-1, 1)), X.reshape((-1, 1))
    Yc, Xc = Y - yc, X - xc
    a2, b2 = a * a, b * b

    for angle in range(0, 90, 5):
        theta = angle / 180 * math.pi
        sint, cost = np.sin(theta), np.cos(theta)

        elVal = np.square(Xc * cost + Yc * sint) / a2 + np.square(Xc * sint - Yc * cost) / b2
        inLocs = np.where(elVal <= 1)[0].flatten().tolist()

        locs = (Y[inLocs, :].flatten(), X[inLocs, :].flatten())
        mask = np.zeros((h, w), 'uint8')
        mask[locs] = 255

        print(f'area: {math.pi * a * b}, sum mask: {np.sum(mask > 0)}')

        cv2.imshow('mask', mask)
        cv2.waitKey()


if __name__ == '__main__':
    # demo_grip_gui(cfg_path='kpick/apps/samples/configs/grip_cornel.cfg',
    #               default_cfg_path='configs/cornell_dataset.cfg')
    demo_grip_gui(cfg_path='kpick/apps/samples/configs/small_cylind.cfg',
                  default_cfg_path='configs/grip_default.cfg')

    # from  kpick.pick.grip.grip_detection import GripDetector
    # aa = 1

    # ws_pts = [(396, 201), (945, 193), (952, 618), (417, 618)]
    # rgb_path = 'data/realsense/0707/image/20210707102818545822.png'
    # depth_path = 'data/realsense/0707/depth/20210707102818545822.png'
    # demo_grip_and_inner_grip_without_gui(rgb_path=rgb_path, depth_path=depth_path, ws_pts=ws_pts)

    # test_tensorrt_grip()
    # test_ellipse()
