

def test():
    from kpick.matching.qatm.QATMer import test_qatmer, test_qatmgui
    # test_qatmer()
    # test_qatmgui(run_realsense=True, viz_tmp='d90', hard_thresh=0.02,
    #              tmp_path='data/template/cylind.png')
    # test_qatmgui(run_realsense=True, viz_tmp='d0',
    #              tmp_path='data/template/copper_pipe2.png')

    test_qatmgui(run_realsense=True, viz_tmp='d90', hard_thresh=0.02,
                 tmp_path='data/template/box.png')


if __name__=='__main__':
    test()
    # matcher = Matching2d(gt_dir='data/matching/samples/gt_placement')
    #
    # im = cv2.imread('data/matching/samples/test_placement/test_input.png')
    # masks = np.load('data/matching/samples/test_placement/target_masks.npy')
    #
    # ret = matcher.match(masks, im)
    #
    #
    # aa = 1