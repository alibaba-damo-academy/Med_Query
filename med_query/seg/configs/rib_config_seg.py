import os
from easydict import EasyDict as edict

work_dir = os.path.expandvars("$WORK_DIR")
exp_dir = os.path.join(work_dir, "rib_experiment")

cfg = edict()
cfg.task = "single-rib-seg"
cfg.dataset = {}
cfg.dataset.type = "SingleRibDataset"
# data should be converted to RAI， crop_size in order x, y, z
cfg.dataset.default_value = -1024
# linear or nearest
cfg.dataset.interpolation = "linear"

cfg.dataset.trainset_csv = os.path.join(exp_dir, "trainset.csv")
cfg.dataset.validset_csv = os.path.join(exp_dir, "validset.csv")
cfg.dataset.testset_csv = os.path.join(exp_dir, "testset.csv")

cfg.dataset.im_dir = os.path.join(exp_dir, "images")
cfg.dataset.mask_dir = os.path.join(exp_dir, "masks_labelled")
cfg.dataset.gt_csv = os.path.join(exp_dir, "cases_info.csv")
cfg.dataset.debug_dir = os.path.join(exp_dir, "debug")
cfg.dataset.is_debugging = False

cfg.dataset.cluster = "0"
cfg.dataset.crop_size = [288, 128, 64]
cfg.dataset.expand = [1.4, 1.4, 1.6]
cfg.dataset.patches_per_volume = 16
cfg.dataset.num_classes = 2

cfg.dataset.normalization_params = {}
cfg.dataset.normalization_params.min_value = -200
cfg.dataset.normalization_params.max_value = 1000
cfg.dataset.normalization_params.clip = True

cfg.dataset.aug_params = {}
cfg.dataset.aug_params.on = True
cfg.dataset.aug_params.prob = 0.6
cfg.dataset.aug_params.translation = [-5, 5]  # mm
cfg.dataset.aug_params.scale = [0.9, 1.1]  # ratio
# rotation axis defined in rai frame, set it to [0,0,0] for random axis
cfg.dataset.aug_params.rotation_axis = [0, 0, 0]
cfg.dataset.aug_params.rotation_angle = [-10, 10]  # degree

cfg.training_params = {}
# 0 stands for not to resume, while positive numbers stands for the epoch number to resume
cfg.training_params.resume_epoch = 0
cfg.training_params.batch_size = 3
cfg.training_params.max_epochs = 2001
cfg.training_params.save_epochs = 50
cfg.training_params.valid_epoch_freq = 20
# num_threads is for each gpu, not in total
cfg.training_params.num_threads = 2
cfg.training_params.random_seed = 42
cfg.training_params.lr = 1e-3
cfg.training_params.clip_grad = 5
cfg.training_params.use_amp = False
cfg.training_params.save_dir = os.path.join(exp_dir, "models", "single_rib_seg")

cfg.lr_policy = "multi_step"
cfg.lr_milestones = [1000]

cfg.network = {}
cfg.network.net_name = "unet"
cfg.network.in_channels = 1
cfg.network.out_channels = cfg.dataset.num_classes
cfg.network.bias_value = 0
# should be one of "batchnorm", "groupnorm", "instancenorm"
cfg.network.normalization = "instancenorm"
cfg.network.dimensions = 3
cfg.network.base = 8
cfg.network.downsample = 5

cfg.loss = {}
# focal loss, dice loss and mix is also allowed
cfg.loss.loss_name = "mix"
cfg.loss.focal_alpha = [1, 2]
cfg.loss.focal_gamma = 2
cfg.loss.balance = [0.2, 0.8]
