import os.path as op
import numpy as np

import RIDet3DAddon.torch.config_dir.parameter_pool as params


class Paths:
    RESULT_ROOT = "/home/cheetah/kim_workspace"
    DATAPATH = "/media/cheetah/IntHDD/kim_result8/deg_0"
    CHECK_POINT = op.join(RESULT_ROOT, "ckpt")
    CONFIG_FILENAME = '/home/cheetah/kim_workspace/new/RILabDetector_torch/RIDet3DAddon/torch/config.py'
    META_CFG_FILENAME = '/home/cheetah/kim_workspace/new/RILabDetector_torch/RIDet3DAddon/torch/config_dir/meta_config.py'


class Datasets:
    # specific dataset configs MUST have the same items

    class Kittibev:
        NAME = "kitti_bev"
        PATH = "/media/cheetah/IntHDD/datasets/kitti_detection/data_object_image_2/training"
        ORIGIN_PATH = "/media/cheetah/IntHDD/datasets/kitti_detection/data_object_image_2/training/image_2"

        CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van","Truck" ,"Cyclist", "Don't Care"]
        CATEGORY_REMAP = {}
        # CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck", "Cyclist", "DontCare",  "Person_sitting"]
        # CATEGORY_REMAP = {}

        CROP_TLBR = [0, 0, 0, 0]
        INCLUDE_LANE = [True, False][1]
        # meter per pixel
        CELL_SIZE = params.KittiBEVParam.Deg.CELL_SIZE
        POINT_XY_RANGE = params.KittiBEVParam.Deg.POINT_XY_RANGE
        TILT_ANGLE = params.KittiBEVParam.Deg.TILT_ANGLE
        INPUT_RESOLUTION = params.KittiBEVParam.Deg.INPUT_RESOLUTION

    # TARGET_DATASET = "Uplus"
    DATASET_CONFIG = None
    TARGET_DATASET = "kittibev"


class Dataloader:
    CATEGORY_NAMES = params.TfrParams.CATEGORY_NAMES
    MIN_PIX = params.TfrParams.MIN_PIX


class ModelOutput:
    FEATURE_SCALES = [8, 16, 32]
    FEAT_RAW = False
    OUTPUT3D = False

    NUM_ANCHORS_PER_SCALE = 1
    # MAIN -> FMAP, NMS -> INST
    GRTR_MAIN_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1,
                             "anchor_id": 1
                             }
    PRED_MAIN_COMPOSITION = params.TrainParams.get_pred_composition()
    PRED_HEAD_COMPOSITION = params.TrainParams.get_pred_composition(True)

    GRTR_NMS_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1}
    PRED_NMS_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, "ctgr_prob": 1, "score": 1, "anchor_ind": 1}

    # GRTR_3D_MAIN_COMPOSITION = {"xyz": 3, "hwl": 3, "theta": 1, "category": 1,
    #                             "anchor_id": 1
    #                             }
    GRTR_3D_MAIN_COMPOSITION = {"lxyz": 3, "hwl": 3, "cxyz": 3, "theta": 1, "category": 1,

                                "anchor_id": 1
                                }
    PRED_3D_MAIN_COMPOSITION = params.TrainParams.get_3d_pred_composition()

    PRED_3D_HEAD_COMPOSITION = params.TrainParams.get_3d_pred_composition(True)

    GRTR_3D_NMS_COMPOSITION = {"lxyz": 3, "hwl": 3, "cxyz": 3, "theta": 1, "category": 1}
    PRED_3D_NMS_COMPOSITION = {"lxyz": 3, "hwl": 3,  "theta": 1, "category": 1}

    NUM_MAIN_CHANNELS = sum(PRED_MAIN_COMPOSITION.values())

    # OUTPUT3D
    VP_BINS = 16


class FeatureDistribPolicy:
    POLICY_NAME = ["SinglePositivePolicy", "FasterRCNNPolicy", "MultiPositivePolicy"][0]
    IOU_THRESH = [0.5, 0.3]
    CENTER_RADIUS = [2.5, 2.5]
    MULTI_POSITIVE_WIEGHT = 0.8


class Architecture:
    ARCHITECTURE = ['detecter3D'][0]
    SIGMOID_DELTA = 0.2


class Optimizer:
    WEIGHT_DECAY = 0.0001
    WEIGHT_DECAY_BIAS = 0.0001
    WEIGHT_DECAY_NORM = 0.0
    MOMENTUM = 0.9


class Train:
    DEVICE = ['cuda', 'cpu'][0]
    TRAINING_PLAN = params.TrainingPlan.KITTIBEV_SIMPLE
    PIXEL_MEAN = [0.0, 0.0, 0.0]
    PIXEL_STD = [1.0, 1.0, 1.0]
    AUGMENT_PROBS = {"Flip": 1.0}


class Validation:
    TP_IOU_THRESH = [1, 0.5, 0.7, 0.5, 0.7, 0.5]
    DISTANCE_LIMIT = 25
    VAL_EPOCH = "latest"
    MAP_TP_IOU_THRESH = [0.7]
    MAX_BOX = 200


class NmsInfer:
    MAX_OUT = [0, 6, 11, 3,10,10]
    IOU_THRESH = [0, 0.67, 0.67, 0.67, 0.67, 0.42]
    SCORE_THRESH = [1, 0.31, 0.26, 0.31, 0.26, 0.16]


class NmsOptim:
    IOU_CANDIDATES = np.arange(0.3, 0.7, 0.1)
    SCORE_CANDIDATES = np.arange(0.1, 0.4, 0.05)
    MAX_OUT_CANDIDATES = np.arange(2, 15, 1)


class Log:
    LOSS_NAME = {ModelOutput.OUTPUT3D: ["bbox2d", "bbox3d",  "object", "category_2d", "category_3d", "theta"],
                }.get(True, ["bbox2d", "bbox3d",  "object", "category_2d", "category_3d", "theta"])
    VISUAL_HEATMAP = True

    class HistoryLog:
        SUMMARY = ["pos_obj", "neg_obj"]

    class ExhaustiveLog:
        DETAIL = ["pos_obj", "neg_obj", "box_yx", "box_hw", "true_class", "false_class"]
        COLUMNS_TO_MEAN = ["anchor", "ctgr", "box_2d", "object", "category_2d", "category_3d", "box_3d", "theta", "pos_obj",
                           "neg_obj", "box_hw", "box_yx", "true_class", "false_class"]
        COLUMNS_TO_SUM = ["anchor", "ctgr", "trpo", "grtr", "pred"]


