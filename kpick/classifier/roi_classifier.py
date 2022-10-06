from .classifier_base import BasCifarClassfier
import torch
import numpy as np
import torch.nn as nn
from ketisdk.utils.proc_utils import Timer
import torch.nn.functional as F

class RoiCifarClassfier(BasCifarClassfier):
    # def __init__(self, args=None, cfg_path=None, name='unnamed', train=False,default_args=None):
    #     super().__init__(args=args, cfg_path=cfg_path, name=name, train=train, default_args=default_args)
    def init(self, net_args, net_train_args=None, train=False):
        model = BasCifarClassfier.init(self,net_args=net_args,  net_train_args=net_train_args, train=train)
        scaling = nn.Upsample(size=net_args.input_shape[:2], mode='bilinear')
        return model, scaling

    def crop_and_concate_roi_tensor(self, im_tensors, box_tensors, ind_tensors, scaling, isSameSize=False):
        roi_concat = []
        # if not isSameSize: scaling = nn.Upsample(size=self.args.net.input_shape[:2], mode='bilinear')
        for i, box in zip(ind_tensors, box_tensors):
            left, top, right, bottom = box

            roi = im_tensors[i][:, top:bottom, left:right]
            roi_scale = scaling(roi.view(1, *roi.size()))[0, :, :, :] if not isSameSize else roi
            # roi_scale = F.resized_crop(roi_scale,top,left, bottom-top, right-left, self.smaller_size)
            roi_concat.append(roi_scale)
        roi_concat = torch.stack(roi_concat)

        if isSameSize:
            roi_concat = scaling(roi_concat)

        return roi_concat

    def test_conv(self):
        tensors = torch.rand(1, 6, 32,40).cuda()
        hi, wi = 32, 32
        stride = 4

        hi2, wi2 = hi//2, wi//2

        pred_map = self.model(tensors)
        for score_loc in [(0,0), (0,1), (0,2)]:
            y,x = score_loc
            xc, yc = 4*x + wi2, 4*y + hi2
            x_start, y_start = xc-wi2, yc - hi2
            roi = tensors[:,:,y_start:y_start+hi, x_start:x_start+wi]

            pred_roi = self.model(roi)
            pred_r = pred_map[:,:,y,x]

            aa = 1

    def forward(self, tensors, model):
        # self.test_conv()

        predicted = model(tensors)
        scores = F.softmax(predicted, dim=1).cpu().detach().numpy()
        return scores


    def predict_tensor_rois(self, im_tensors, boxes, model, net_args, scaling, inds=None, isSameSize=False):
        if inds is None: inds = np.zeros((len(boxes),1), 'uint8')

        box_tensors = torch.from_numpy(boxes)
        ind_tensors = torch.from_numpy(inds)
        if self.use_cuda:
            im_tensors = [el.cuda() for el in im_tensors]
            ind_tensors, box_tensors = ind_tensors.cuda(), box_tensors.cuda()

        timer = Timer()
        roi_tensors = self.crop_and_concate_roi_tensor(im_tensors, box_tensors, ind_tensors, scaling, isSameSize=isSameSize)
        num_box = len(boxes)
        timer.pin_time('crop')
        if num_box < net_args.test_batch: return self.forward(roi_tensors, model=model)

        # num_batch = int(np.ceil(num_box / self.args.test_batch))
        probs = []
        split_range = list(range(0, num_box,net_args.test_batch)) + [num_box,]
        for j, end in enumerate(split_range[1:]):
            start = 0 if j==0 else split_range[j]
            probs.append(self.forward(roi_tensors[start:end, :], model=model))
        # probs = torch.cat(probs, dim=0)
        ret = np.vstack(probs)
        timer.pin_time('model_run')
        print(timer.pin_times_str())
        # ret = probs.cpu().detach().numpy()
        return  ret

    # def predict_rois_m(self,im_tensors, boxes, inds):
    #     box_tensors = torch.from_numpy(boxes)
    #     ind_tensors = torch.from_numpy(inds)
    #     if self.use_cuda:
    #         im_tensors = [el.cuda() for el in im_tensors]
    #         ind_tensors, box_tensors = ind_tensors.cuda(), box_tensors.cuda()
    #     # im_tensors = im_tensors.view(1, *im_tensors.size())
    #     out =self.model((im_tensors, box_tensors, ind_tensors))
    #
    #     # # visualize model
    #     # if not hasattr(self, 'do_torchviz'):
    #     #     from torchviz import make_dot
    #     #     make_dot(out, params=dict(list(self.model.named_parameters()))).render("rnn_torchviz", format="png")
    #     #     print('model saved...')
    #     #     self.do_torchviz = True
    #     return out.cpu().detach().numpy()
    #
    #
    # def predict_tensor_rois(self, im_tensors, boxes, inds=None):
    #     if inds is None: inds = np.zeros((len(boxes),1), 'uint8')
    #     num_box = len(boxes)
    #     if num_box < self.args.net.test_batch: return self.predict_rois_m(im_tensors=im_tensors, boxes=boxes,
    #                                                                    inds=inds)
    #
    #     # num_batch = int(np.ceil(num_box / self.args.test_batch))
    #     probs = []
    #     split_range = list(range(0, num_box,self.args.net.test_batch)) + [num_box,]
    #     for j, end in enumerate(split_range[1:]):
    #         start = 0 if j==0 else split_range[j]
    #         probs.append(self.predict_rois_m(im_tensors=im_tensors,
    #                                          boxes=boxes[start:end, :],
    #                                          inds=inds[start:end, :]))
    #     return np.vstack(probs)
    #
