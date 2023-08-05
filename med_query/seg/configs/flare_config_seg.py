import os
from easydict import EasyDict as edict

work_dir = os.path.expandvars("$WORK_DIR")
exp_dir = os.path.join(work_dir, "flare_experiment")

cfg = edict()
cfg.task = "single-organ-seg"
cfg.dataset = {}
cfg.dataset.type = "SingleOrganDataset"
# data should be converted to RAI， crop_size in order x, y, z
cfg.dataset.default_value = -1024
# linear or nearest
cfg.dataset.interpolation = "linear"

cfg.dataset.trainset_csv = os.path.join(exp_dir, "trainset.csv")
cfg.dataset.validset_csv = os.path.join(exp_dir, "validset.csv")
cfg.dataset.testset_csv = os.path.join(exp_dir, "testset.csv")

cfg.dataset.im_dir = os.path.join(exp_dir, "images")
cfg.dataset.mask_dir = os.path.join(exp_dir, "masks")
cfg.dataset.gt_csv = os.path.join(exp_dir, "cases_info.csv")
cfg.dataset.debug_dir = os.path.join(exp_dir, "debug")
cfg.dataset.is_debugging = False

cfg.dataset.organ_dict = {
    "1": {"cluster": "0", "crop_size": [288, 288, 160], "organ": "liver"},
    "2": {"cluster": "1", "crop_size": [128, 128, 128], "organ": "right kidney"},
    "3": {"cluster": "10", "crop_size": [128, 128, 128], "organ": "spleen"},
    "4": {"cluster": "2", "crop_size": [256, 128, 160], "organ": "pancreas"},  # updated
    "5": {"cluster": "3", "crop_size": [144, 144, 288], "organ": "aorta"},  # updated
    "6": {
        "cluster": "3",
        "crop_size": [144, 144, 288],
        "organ": "inferior vena cava",
    },  # merged into label 5
    "7": {"cluster": "5", "crop_size": [160, 96, 96], "organ": "right adrenal gland"},  # updated
    "8": {
        "cluster": "5",
        "crop_size": [160, 96, 96],
        "organ": "left adrenal gland",
    },  # merged into label 7
    "9": {"cluster": "6", "crop_size": [112, 112, 96], "organ": "gallbladder"},
    "10": {
        "cluster": "3",
        "crop_size": [144, 144, 288],
        "organ": "esophagus",
    },  # merged into label 5
    "11": {"cluster": "8", "crop_size": [224, 224, 128], "organ": "stomach"},
    "12": {
        "cluster": "2",
        "crop_size": [256, 128, 160],
        "organ": "duodenum",
    },  # merged into label 4
    "13": {"cluster": "1", "crop_size": [128, 128, 128], "organ": "left kidney"},
}

cfg.dataset.organ_cluster = {
    "0": {"label": [1], "crop_size": [288, 288, 160]},
    "1": {"label": [2, 13], "crop_size": [128, 128, 128]},
    "2": {"label": [4, 12], "crop_size": [256, 128, 160]},  # multi-channel
    "3": {"label": [5, 6, 10], "crop_size": [144, 144, 288]},  # multi-channel
    # "4": {"label": [6], "crop_size": [96, 96, 192]},
    "5": {"label": [7, 8], "crop_size": [160, 96, 96]},  # multi-channel
    "6": {"label": [9], "crop_size": [112, 112, 96]},
    #  "7": {"label": [10], "crop_size": [96, 96, 160]},
    "8": {"label": [11], "crop_size": [224, 224, 128]},
    # "9": {"label": [12], "crop_size": [256, 128, 112]}
    "10": {"label": [3], "crop_size": [128, 128, 128]},
}

cfg.dataset.cluster = "2"
cfg.dataset.expand = [1.2, 1.2, 1.2]
cfg.dataset.num_classes = (
    len(cfg.dataset.organ_cluster[cfg.dataset.cluster]["label"]) + 1
    if cfg.dataset.cluster != "1"
    else 2
)

cfg.dataset.normalization_params = {}
cfg.dataset.normalization_params.min_value = -125
cfg.dataset.normalization_params.max_value = 225
cfg.dataset.normalization_params.clip = True

cfg.dataset.aug_params = {}
cfg.dataset.aug_params.on = True
cfg.dataset.aug_params.prob = 0.6
cfg.dataset.aug_params.translation = [-20, 20]  # mm
cfg.dataset.aug_params.scale = [0.9, 1.1]  # ratio
# rotation axis defined in rai frame, set it to [0,0,0] for random axis
cfg.dataset.aug_params.rotation_axis = [0, 0, 0]
cfg.dataset.aug_params.rotation_angle = [-10, 10]  # degree

cfg.training_params = {}
# 0 stands for not to resume, while positive numbers stands for the epoch number to resume
cfg.training_params.resume_epoch = 0
cfg.training_params.batch_size = 4
cfg.training_params.max_epochs = 500
cfg.training_params.save_epochs = 50
cfg.training_params.valid_epoch_freq = 10
# num_threads is for each gpu, not in total
cfg.training_params.num_threads = 2
cfg.training_params.random_seed = 42
cfg.training_params.lr = 1e-3
cfg.training_params.clip_grad = 5
cfg.training_params.use_amp = False
cfg.training_params.save_dir = os.path.join(exp_dir, "models", "flare_seg")

cfg.lr_policy = "multi_step"
cfg.lr_milestones = [300]

cfg.network = {}
cfg.network.net_name = "unet"
cfg.network.in_channels = 1
cfg.network.out_channels = cfg.dataset.num_classes
cfg.network.bias_value = 0
# should be one of "batchnorm", "groupnorm", "instancenorm"
cfg.network.normalization = "instancenorm"
cfg.network.dimensions = 3
cfg.network.base = 16
cfg.network.downsample = 4

cfg.loss = {}
# focal loss, dice loss and mix is also allowed
cfg.loss.loss_name = "mix"
cfg.loss.focal_alpha = [1] + [2] * (cfg.dataset.num_classes - 1)
cfg.loss.focal_gamma = 2
cfg.loss.balance = [0.2, 0.8]
