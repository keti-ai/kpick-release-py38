from kpick.classifier.classifier_base import BasCifarClassfier
from  ketisdk.utils.proc_utils import CFG

def train_grip_20220715():
    net_args = CFG()
    net_args.arch = 'resnet'
    net_args.model_depth = 20
    net_args.fc2conv = False
    net_args.use_cuda = True
    net_args.test_batch = 256
    net_args.num_workers = 8
    net_args.get_rgb = True
    net_args.get_depth = True
    net_args.depth2norm = True
    net_args.input_shape = (32, 128, 6)
    net_args.db_mean = (0.41828456, 0.40237707, 0.41067797, 0.24735339, 0.1663269, 0.85723261)
    net_args.db_std = (0.14510322, 0.14341601, 0.14637822, 0.25664706, 0.21761108, 0.16026509)
    net_args.dber = 'GripCifar10'
    net_args.classes = ['grip', 'ungrip']
    net_args.checkpoint_dir = 'data/grip_dataset/20220616/resnet20_32x128x6_20220715'

    net_train_args = CFG()
    net_train_args.root_dir = 'data/grip_dataset/combine_grip_db_0624'
    net_train_args.schedule = (81, 122)
    net_train_args.gamma = 0.1
    net_train_args.wd = 0.0001
    net_train_args.cardinality = 8
    net_train_args.widen_factor = 4
    net_train_args.growthRate = 12
    net_train_args.compressionRate = 2
    net_train_args.train_batch = 256
    net_train_args.lr = 0.01
    net_train_args.momentum = 0.9
    net_train_args.weight_decay = 0.0005
    net_train_args.epochs = 100
    net_train_args.save_every = 10
    net_train_args.traindb_dir = 'cifar10'

    trainer = BasCifarClassfier()
    trainer.init(net_args=net_args,net_train_args=net_train_args, train=True)
    trainer.trainval()



if __name__=='__main__':
    train_grip_20220715()
