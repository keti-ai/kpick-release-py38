from ketisdk.vision.base.base_objects import BasObj
import torchvision.transforms as transforms
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
# from ..dataset.datasets import Dataset, get_mean_std
from . import cifar_classification_models as models
import torch.backends.cudnn as cudnn
import shutil
from time import time

from .classification_utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from .classification_utils.progress.progress.bar import Bar
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from ketisdk.import_basic_utils import *
import kpick
MODULE_DIR  = os.path.split(kpick.__file__)[0]

__CHECPOINT_LINKS__ = {'SuctionCifar10-resnet20-32x32x6.pth':
                           'https://docs.google.com/uc?export=download&id=1xzLrHpYHMq6MELE9LY9lKsrF9k5NusOK',
                       'GripCifar10-resnet20-32x128x6.pth':
                           'https://docs.google.com/uc?export=download&id=1UDyhSpkOxcSWJC7c-jfvy3NMX9vaG-f5',
                       }
__DEFAULT_CHECKPOINTS_PATH__ = {
'SuctionCifar10-resnet20-32x32x6.pth': os.path.join(MODULE_DIR, 'checkpoint/SuctionCifar10-resnet20-32x32x6.pth'),
'GripCifar10-resnet20-32x128x6.pth': os.path.join(MODULE_DIR, 'checkpoint/GripCifar10-resnet20-32x128x6.pth')
}

# class BasCifarClassfier(BasObj):
class BasCifarClassfier():
    # def __init__(self,args=None, cfg_path=None, name='unnamed', train=False, default_args=None):
    #     super().__init__(args=args, cfg_path=cfg_path, name=name, default_args=default_args)
    def init(self, net_args=None, net_train_args=None,train=False):
        self.train_mode = train
        self.net_train_args = net_train_args
        self.net_args = net_args
        if self.train_mode: self.lr = net_train_args.lr
        self.get_data(net_args)
        model = self.get_model(net_args, net_train_args)


        return model

    def predict_array(self,im, transform):
        # t = time()
        # img = Image.fromarray(im)
        img = self.transform(im)
        if self.use_cuda: img = img.cuda()
        img = img.view(1, *img.size())
        img = Variable(img)
        predicted = self.model(img)
        probs = F.softmax(predicted, dim=1)
        if self.use_cuda: probs = probs.cpu()
        # print('classification elapsed: %.3fs' %(time()-t))
        return probs.detach().numpy().flatten()

    def predict_rgbd(self, rgbd):
        im = rgbd.resize(self.args.net.input_shape[:2]).array(get_rgb=self.args.net.get_rgb,
                                                          get_depth=self.args.net.get_depth,
                                                          depth2norm=self.args.net.depth2norm)
        return self.predict_array(im)

    def predict_arrays_m(self, ims, transform, print_info=True):
        timer = Timer()
        img_concat = self.arrays2tensors(ims=ims, transform=transform)
        timer.pin_time(label='concat')
        predicted = self.model(img_concat)
        probs = F.softmax(predicted, dim=1)
        if self.use_cuda: probs = probs.cpu()
        probs = probs.detach().numpy()
        timer.pin_time(label='predict')
        if print_info: print(timer.pin_times_str())
        return probs

    def arrays2tensors(self, ims, transform):
        img_concat = []
        for im in ims:
            img = transform(im)
            if self.use_cuda: img = img.cuda()
            img_concat.append(img)
        img_concat = torch.stack(img_concat)
        # img_concat= torch.autograd.Variable(img_concat, volatile=True)
        return img_concat

    # def predict_arrays(self, ims, print_info=True):
    #     timer = Timer()
    #     num_im = len(ims)
    #     if num_im <= self.args.test_batch: return self.predict_arrays_m(ims)
    #
    #     part = list(np.arange(0, num_im, self.args.test_batch))
    #     if part[-1] != num_im: part.append(num_im)
    #
    #     probs = []
    #     for i in range(len(part)-1):
    #         select_ims = ims[part[i]: part[i+1]]
    #         partial_probs = self.predict_arrays_m(select_ims, print_info=False)
    #         probs.append(partial_probs)
    #     probs = np.vstack(probs)
    #
    #     if print_info:
    #         print('test batch size: %d' %self.args.test_batch)
    #         print(timer.pin_times_str())
    #
    #
    #     return probs

    def predict_arrays(self, ims, Dataset, transform, print_info=True):
        num_im = len(ims)
        if num_im<self.args.net.test_batch: return self.predict_arrays_m(ims=ims, transform=transform, print_info=print_info)
        timer1 = Timer()
        testset = Dataset(name=self.args.net.dber , transform=transform, train=False, data=ims, im_shape=self.args.net.input_shape, indexing=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.net.test_batch,
                                                      shuffle=True, num_workers=self.args.net.num_workers)
        timer1.pin_time('load_data')

        num_batch = int(np.ceil(num_im/self.args.net.test_batch))
        probs = np.zeros((num_im, self.num_classes))

        # probs_tensor = torch.zeros((num_im, self.num_classes), dtype=torch.float,device=self.device)
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            timer = Timer()
            if self.use_cuda:
                inputs = inputs.cuda()
            # inputs = torch.autograd.Variable(inputs, volatile=True)
            outputs = self.model(inputs)
            outputs = F.softmax(outputs, dim=1)
            timer.pin_time('score')

            if self.use_cuda: outputs = outputs.cpu()
            outputs = outputs.detach().numpy()
            timer.pin_time('cvt_cpu')

            probs[(indexes,)] = outputs
            timer.pin_time('save_array')
            # print(timer.pin_times_str())

        timer1.pin_time('scoring_all')
        if print_info:
            print('batch_szie:%d-num_batch:%d-msPerBatch:%0.1f' %(self.args.net.test_batch, num_batch,
                                                                  1000*timer1.get_pinned_time('scoring_all')/num_batch))
            print(timer1.pin_times_str())

        return probs


    def predict_rgbds(self, rgbds):
        ims = []
        for rgbd in rgbds:
            ims.append(rgbd.resize(self.args.net.input_shape[:2]).array(get_rgb=self.args.net.get_rgb,
                                                                    get_depth=self.args.net.get_depth,
                                                                    depth2norm=self.args.net.depth2norm))
        probs = self.predict_arrays(ims)
        # print('classification elapsed: %.3fs' % (time() - t))
        return probs


    def trainval(self):
        # Train and val
        for epoch in range(self.start_epoch, self.net_train_args.epochs):
            self.adjust_learning_rate(epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, self.net_train_args.epochs, self.lr))

            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()

            # append logger file
            self.logger.append([self.lr, train_loss, test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc > self.best_acc
            self.best_acc = max(test_acc, self.best_acc)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'acc': test_acc,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

        self.logger.close()
        self.logger.plot()
        savefig(os.path.join(self.net_args.checkpoint_dir, 'log.eps'))

        print('Best acc:')
        print(self.best_acc)


    def train(self):
        # switch to train mode
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time()

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            data_time.update(time() - end)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            # print(f'input shape: {inputs.shape}')
            # print(f'input type: {inputs.dtype}')
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1,2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.trainloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg)

    def test(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time()
        bar = Bar('Processing', max=len(self.testloader))
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            # measure data loading time
            data_time.update(time() - end)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 2))
            # print(f'loss: {loss.size(0)}, input_size: {inputs.size(0)}')
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg)



    def get_data(self, net_args):
        # # Transform
        # db_mean, db_std = get_mean_std(self.args.dber)
        #
        # self.transform_train = transforms.Compose([
        #     # transforms.RandomCrop(32, padding=4),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(db_mean, db_std),
        # ])
        # self.transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(db_mean, db_std),
        # ])
        if self.train_mode:
            # Transform
            db_mean, db_std = net_args.db_mean, net_args.db_std

            transform_train = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(db_mean, db_std),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(db_mean, db_std),
            ])


            # Data
            data_root = self.net_train_args.root_dir
            from kpick.pick.dataset.dataset import Dataset
            testset = Dataset(name=self.net_args.dber, root=data_root, train=False, transform=transform_test,
                              download=False, im_shape=self.net_args.input_shape)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.net_args.test_batch,
                                                          shuffle=True, num_workers=self.net_args.num_workers)

            trainset = Dataset(name=self.net_args.dber, root=data_root, train=True, transform=transform_train,
                               download=False, im_shape=self.net_args.input_shape)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.net_train_args.train_batch,
                                                           shuffle=True, num_workers=self.net_args.num_workers)
            self.classes = trainset.classes
        else: self.classes = net_args.classes

        self.num_classes = len(self.classes)

    def get_model(self, net_args, net_train_args=None):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        # Model
        print("==> creating model '{}'".format(net_args.arch))
        if net_args.arch.startswith('resnext'):
            model = models.__dict__[net_args.arch](
                cardinality=net_args.cardinality,
                num_classes=self.num_classes,
                depth=net_args.model_depth,
                widen_factor=net_args.widen_factor,
                dropRate=net_args.drop,
            )
        elif net_args.arch.startswith('densenet'):
            model = models.__dict__[net_args.arch](
                num_classes=self.num_classes,
                depth=net_args.depth,
                growthRate=net_args.growthRate,
                compressionRate=net_args.compressionRate,
                dropRate=net_args.drop,
            )
        elif net_args.arch.startswith('wrn'):
            model = models.__dict__[net_args.arch](
                num_classes=self.num_classes,
                depth=net_args.model_depth,
                widen_factor=net_args.widen_factor,
                dropRate=net_args.drop,
            )
        elif net_args.arch.endswith('resnet'):
            model = models.__dict__[net_args.arch](
                num_classes=self.num_classes,
                depth=net_args.model_depth,
                input_shape=net_args.input_shape,
                fc2conv= net_args.fc2conv
                # block_name=self.args.block_name,
            )
        elif net_args.arch.endswith('resnet_roi'):
            arch = 'resnet' if self.train_mode else net_args.arch
            model = models.__dict__[arch](
                num_classes=self.num_classes,
                depth=net_args.model_depth,
                input_shape=net_args.input_shape,
                fc2conv= net_args.fc2conv
                # block_name=self.args.block_name,
            )
        elif net_args.arch=='torch_resnet18':
            from torchvision.models import resnet18
            model = resnet18(pretrained=False)
        else:
            model = models.__dict__[net_args.arch](num_classes=self.num_classes)
        model = torch.nn.DataParallel(model).cuda()

        cudnn.benchmark = True
        self.title = '%s-%s-' % (net_args.dber, net_args.arch)
        # self.checkpoint_prefix = os.path.join(self.args.net.checkpoint_dir, self.title)
        self.checkpoint_prefix = self.net_args.checkpoint_dir
        if self.train_mode:
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(model.parameters(), lr=net_train_args.lr, momentum=net_train_args.momentum,
                                       weight_decay=net_train_args.weight_decay)
            self.resume()
            self.model = model
        else:
            if not hasattr(net_args, 'run_tensorrt'):
                model=self.load_model(model=model, checkpoint_path=net_args.checkpoint_path, fc2conv=net_args.fc2conv)
            else: model = self.load_model(model=model, checkpoint_path=net_args.checkpoint_path,run_tensorRT=net_args.run_tensorrt,
                                  input_shape=net_args.input_shape, fc2conv=net_args.fc2conv)
        return model

    def load_model_gpu(self, model, checkpoint_path, fc2conv=False):
        print(f'loading {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        if fc2conv:
            checkpoint['state_dict']['module.fc.weight'] = \
                torch.unsqueeze(torch.unsqueeze(checkpoint['state_dict']['module.fc.weight'], -1), -1)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval().cuda()
        print('model loaded ...')
        return model

    def model2trt(self, input_shape):
        print('converting to TensorRT...')
        from torch2trt import trt, torch2trt
        x = torch.ones((1, input_shape[2], input_shape[0], input_shape[1])).cuda()
        self.model = torch2trt(self.model.module, [x], max_batch_size=self.args.net.test_batch,
                               fp16_mode=True, default_device_type=trt.DeviceType.DLA, dla_core=0)
        print('converting complete')


    def load_model(self, model,checkpoint_path, input_shape=None, run_tensorRT=False, fc2conv=False):
        # checkpoint_path = self.checkpoint_prefix + 'model_best.pth'
        # checkpoint_path = checkpoint_path.replace('_roi', '')
        if not os.path.exists(checkpoint_path):
            print(f'checkpoint {checkpoint_path} does not exist ...')
            checkpoint_dir, checkpoint_name = os.path.split(checkpoint_path)
            # os.makedirs(checkpoint_dir, exist_ok=True)
            # link = __CHECPOINT_LINKS__[checkpoint_name]
            # os.system(f'wget --no-check-certificate "{link}" -O {checkpoint_path}')
            # print(f'checkpoint donwloaded and saved at {checkpoint_path}')
            if checkpoint_name not in __DEFAULT_CHECKPOINTS_PATH__:
                print(f'Cannot find {checkpoint_name} in {os.path.join(MODULE_DIR, "checkpoint")}')
                return
            checkpoint_path = __DEFAULT_CHECKPOINTS_PATH__[checkpoint_name]
            print(f'Use Default Checkpoint {checkpoint_path} instead ...')


        if not run_tensorRT:
            model = self.load_model_gpu(model, checkpoint_path, fc2conv=fc2conv)
        else:
            trt_suf = '_trt_conv.pth' if fc2conv else '_trt.pth'
            trt_checkpoint_path = checkpoint_path.replace('.pth', trt_suf)

            if not os.path.exists(trt_checkpoint_path):
                print(f'checkpoint {trt_checkpoint_path} does not exist ...')
                model = self.load_model_gpu(model, checkpoint_path, fc2conv=fc2conv)
                self.model2trt(input_shape=input_shape)
                torch.save(model.state_dict(), trt_checkpoint_path)
                print(f'{trt_checkpoint_path} model saved ...')
                return

            print(f'loading {trt_checkpoint_path} model')
            from torch2trt import TRTModule
            model = TRTModule()
            model.load_state_dict(torch.load(checkpoint_path))
            print('trt model loaded ...')

        return model


    def resume(self):
        # save
        if not os.path.exists(self.net_args.checkpoint_dir): os.makedirs(self.net_args.checkpoint_dir)

        # Resume
        self.resume_checkpoint = self.checkpoint_prefix + 'model_best.pth'
        has_resume = os.path.exists(self.resume_checkpoint)

        if has_resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(self.resume_checkpoint)
            self.best_acc = checkpoint['best_acc']
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger = Logger(self.checkpoint_prefix + 'log.txt', title=self.title, resume=True)
        else:
            self.logger = Logger(self.checkpoint_prefix + 'log.txt', title=self.title)
            self.logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
            self.best_acc = 0
            self.start_epoch = 0

    def save_checkpoint(self, state, is_best):
        filepath = self.checkpoint_prefix + str(state['epoch']).rjust(4, '0') + '.pth'
        temp = self.checkpoint_prefix +  'temp.pth'
        torch.save(state, temp)
        if is_best:
            shutil.copyfile(temp, self.resume_checkpoint)
        if state['epoch'] % self.net_train_args.save_every == 0:
            shutil.copyfile(temp, filepath)

    def adjust_learning_rate(self, epoch):
        if epoch in self.net_train_args.schedule:
            self.lr *= self.net_train_args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr


class RGBDLoader():

    def __init__(self, args, rgbds, transform):
        self.transform = transform
        self.ims = []
        for rgbd in rgbds:
            self.ims.append(rgbd.resize(args.input_shape[:2]).array(get_rgb=args.get_rgb,
                                                                    get_depth=args.get_depth))
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: image.
        """
        img = self.ims[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.ims)







