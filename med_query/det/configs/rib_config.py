# Copyright (c) DAMO Health

import os
import sys
from easydict import EasyDict

from med_query.det.configs.default_config import DefaultConfigs

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class Configs(DefaultConfigs):
    def __init__(self):
        super(Configs, self).__init__()

        #########################
        #      Key params       #
        #########################
        self.type = "MedQuery"

        # minimum confidence value to select predictions for evaluation.
        self.min_det_thresh = float(os.environ.get("MIN_DET_THRESH", 0.0))

        self.batch_size = 8
        # threads for loading data, set to zero for mixed cpu and gpu operations in DataLoader
        self.nThreads = 2

        #########################
        #         I/O           #
        #########################
        self.work_dir = os.path.expandvars("$WORK_DIR")
        self.exp_dir = os.path.join(self.work_dir, "rib_experiment")
        # Model Path
        self.snapshot = None
        self.MODEL_SAVE_PATH = os.path.join(self.exp_dir, "models")
        # Result Path
        self.RESULT_SAVE_PATH = os.path.join(self.exp_dir, "results")

        #########################
        #       DataLoader      #
        #########################
        self.dataset = EasyDict()
        self.dataset.type = "RibDetSegDataset"
        self.dataset.spacing = [2, 2, 2]
        self.dataset.max_stride = 16
        self.dataset.max_bbox_size = [332.0, 246.0, 404.0]  # mm
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
        #     Architecture      #
        #########################
        self.num_classes = 24  # not including background
        self.backbone = dict(
            type="ResNet",
            depth=50,
            in_channels=1,
            base_channels=32,
            num_stages=3,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            out_indices=(2,),
            frozen_stages=-1,
            conv_cfg=dict(type="Conv3d"),
            norm_cfg=dict(type="BN3d", requires_grad=True, momentum=0.1),
            norm_eval=False,
            style="pytorch",
            dim=3,
        )
        self.bbox_head = dict(
            type="MedQueryHead",
            num_classes=self.num_classes,
            in_channels=512,
            num_query=self.num_classes + 1,
            with_positional_encoding=True,
            transformer=dict(
                type="Transformer3D",
                encoder=dict(
                    type="DetrTransformerEncoder",
                    num_layers=1,
                    transformerlayers=dict(
                        type="BaseTransformerLayer",
                        attn_cfgs=[
                            dict(
                                type="MultiheadAttention",
                                embed_dims=384,
                                num_heads=4,
                                attn_drop=0.1,
                            )
                        ],
                        ffn_cfgs=dict(
                            type="FFN",
                            embed_dims=384,
                            feedforward_channels=128,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type="ReLU", inplace=True),
                        ),
                        operation_order=("self_attn", "norm", "ffn", "norm"),
                    ),
                ),
                decoder=dict(
                    type="DetrTransformerDecoder",
                    return_intermediate=True,
                    num_layers=3,
                    transformerlayers=dict(
                        type="BaseTransformerLayer",
                        attn_cfgs=dict(
                            type="MultiheadAttention", embed_dims=384, num_heads=4, attn_drop=0.3
                        ),
                        ffn_cfgs=dict(
                            type="FFN",
                            embed_dims=384,
                            feedforward_channels=256,
                            num_fcs=2,
                            ffn_drop=0.3,
                            act_cfg=dict(type="ReLU", inplace=True),
                        ),
                        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    ),
                ),
            ),
            positional_encoding=dict(
                type="SinePositionalEncoding3D", num_feats=128, normalize=True
            ),
            loss_cls=dict(
                type="CrossEntropyLoss",
                bg_cls_weight=10.0,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=10.0),
            loss_angle=dict(type="L1Loss", loss_weight=10.0),
            train_cfg=dict(
                assigner=dict(
                    type="HungarianAssigner3D",
                    cls_cost=dict(type="ClassificationCost", weight=1.0),
                    box_cost=dict(type="BBoxL1Cost3D", weight=10.0, box_format="cxcyczwhd"),
                    ang_cost=dict(type="BBoxL1Cost3D", weight=10.0),
                    idx_cost=dict(type="IndexCost", num_classes=self.num_classes + 1, weight=4.0),
                )
            ),
            test_cfg=dict(max_per_img=30),
        )

        #########################
        #        Training       #
        #########################
        # initial epoch count
        self.epoch_count = 0
        # initial learning rate for adam
        self.lr = 0.0004
        # momentum term fo adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0.1
        self.clip_grad = 0.1

        # frequency of print log
        self.print_freq = 1
        # frequency of save model
        self.save_epoch_freq = 50
        # frequency of validation
        self.valid_epoch_freq = 50

        #########################
        #        Schedule       #
        #########################
        # learning rate policy: lambda | step | multi_step | cosine | plateau | warm_up_multi_step
        self.lr_policy = "warm_up_multi_step"
        self.warm_up_epochs = 200
        self.max_epochs = 1000
        self.lr_milestones = [800]


if __name__ == "__main__":
    cfg = Configs()
    cfg.print_configs()
