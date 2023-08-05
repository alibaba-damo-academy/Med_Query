# Copyright (c) DAMO Health

import os
import sys
from easydict import EasyDict

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class DefaultConfigs(object):
    def __init__(self):

        #########################
        #      Key params       #
        #########################
        # minimum confidence value to select predictions for evaluation.
        self.min_det_thresh = float(os.environ.get("MIN_DET_THRESH", 0.0))

        # phase of 'train', 'valid', 'test', default is 'train'
        self.phase = "train"

        # one out of [2, 3]. dimension the model operates in.
        self.dim = 3
        self.batch_size = 8
        # threads for loading data, set to zero for mixed cpu and gpu operations in DataLoader
        self.nThreads = 2

        # flag to use automatic mixed precision mode
        self.use_amp = False
        # perform resizing at test time
        self.test_aug = False

        #########################
        #         I/O           #
        #########################
        self.work_dir = os.path.expandvars("$WORK_DIR")
        self.exp_dir = os.path.join(self.work_dir, "flare_experiment")
        # Model Path
        self.snapshot = None
        self.MODEL_SAVE_PATH = os.path.join(self.exp_dir, "save_models")
        # Result Path
        self.RESULT_SAVE_PATH = os.path.join(self.exp_dir, "save_results")

        #########################
        #       DataLoader      #
        #########################
        self.dataset = EasyDict()
        self.dataset.spacing = [2, 2, 2]
        self.dataset.max_stride = 16
        self.dataset.max_bbox_size = [350.0, 256.0, 384.0]  # mm
        self.dataset.fixed_bbox_size = [384.0, 384.0, 512.0]  # mm
        self.dataset.use_fixed_bbox_size = False
        # data should be converted to RAI， crop_size in order x, y, z
        self.dataset.default_value = -1024
        # linear or nearest
        self.dataset.interpolation = "linear"
        self.dataset.trainset_csv = os.path.join(self.exp_dir, "trainset.csv")
        self.dataset.validset_csv = os.path.join(self.exp_dir, "validset.csv")
        self.dataset.testset_csv = os.path.join(self.exp_dir, "testset.csv")

        self.dataset.image_dir = os.path.join(self.exp_dir, "images_2mm")
        self.dataset.mask_dir = os.path.join(self.exp_dir, "masks_2mm_dilated")
        self.dataset.gt_path = os.path.join(self.exp_dir, "cases_info.csv")
        self.dataset.debug_dir = os.path.join(self.exp_dir, "debug")
        self.dataset.is_debugging = False

        self.dataset.normalization_params = {}
        self.dataset.normalization_params.min_value = -200
        self.dataset.normalization_params.max_value = 1000
        self.dataset.normalization_params.clip = True

        self.dataset.aug_params = {}
        self.dataset.aug_params.on = True
        self.dataset.aug_params.use_crop = True
        self.dataset.aug_params.prob = 0.8
        self.dataset.aug_params.erase_prob = 0.8
        self.dataset.aug_params.translation = [-20, 20]  # mm
        self.dataset.aug_params.scale = [0.9, 1.1]  # ratio
        # rotation axis defined in rai frame, set it to [0, 0, 0] for random axis
        self.dataset.aug_params.rotation_axis = [0, 1, 0]
        self.dataset.aug_params.rotation_angle = [-15, 15]  # degree

        #########################
        #  Schedule / Selection #
        #########################
        # learning rate policy: lambda | step | multi_step | cosine | plateau | warm_up_multi_step
        self.lr_policy = "warm_up_multi_step"
        self.warm_up_epochs = 200
        self.max_epochs = 1000
        self.lr_milestones = [600, 800]

        #########################
        #         Log           #
        #########################
        # whether to log additional info
        self.verbose = False

        #########################
        #     Postprocessing    #
        #########################
        self.final_max_instances_per_batch_element = 24

    def print_configs(self):
        message = ""
        message += "==================== Configs ====================\n"
        for k, v in sorted(vars(self).items()):
            message += "{:>25}: {:<30}\n".format(str(k), str(v))
        message += "====================== End ======================"
        print(message)


if __name__ == "__main__":
    cfg = DefaultConfigs()
    cfg.print_configs()
