import os.path as op
import RIDet3DAddon.tflow.config_dir.parameter_pool as params
import numpy as np


class Paths:
    RESULT_ROOT = "/home/eagle/mun_workspace"
    DATAPATH = op.join(RESULT_ROOT, "tfrecord")
    CHECK_POINT = op.join(RESULT_ROOT, "ckpt")
    CONFIG_FILENAME = '/home/eagle/mun_workspace/Detector/RILabDetector/RIDet3DAddon/config.py'
    META_CFG_FILENAME = '/home/eagle/mun_workspace/Detector/RILabDetector/RIDet3DAddon/tflow/config_dir/meta_config.py'


class Datasets:
    # specific dataset configs MUST have the same items
    class Kitti:
        NAME = "kitti"
        PATH = "/home/eagle/mun_workspace/M3D-RPN/data/kitti_split1"
        # CATEGORIES_TO_USE = ["Pedestrian", "Car", "Cyclist", "Van", "Truck", "Person_sitting"]
        CATEGORIES_TO_USE = ["Pedestrian", "Car", "Cyclist"]
        CATEGORY_REMAP = {}
        # (4,13) * 64
        INPUT_RESOLUTION = (320, 1024)
        MAX_DEPTH = 50
        MIN_DEPTH = 0.01

    DATASET_CONFIG = None
    TARGET_DATASET = "kitti"


class Dataloader:
    DATASETS_FOR_TFRECORD = {
        "kitti": ("train", "val"),
    }
    MAX_BBOX_PER_IMAGE = 50
    MAX_DONT_PER_IMAGE = 50

    CATEGORY_NAMES = params.TfrParams.CATEGORY_NAMES
    SHARD_SIZE = 2000
    ANCHORS_PIXEL = None


class ModelOutput:
    FEATURE_SCALES = [8, 16, 32]
    FEAT_RAW = False
    IOU_AWARE = [True, False][1]

    NUM_ANCHORS_PER_SCALE = 1
    # MAIN -> FMAP, NMS -> INST
    GRTR_MAIN_COMPOSITION = {"yxhw": 4, "z": 1, "object": 1, "category": 1, "anchor_ind": 1}
    PRED_MAIN_COMPOSITION = params.TrainParams.get_pred_composition(IOU_AWARE)
    GRTR_3D_MAIN_COMPOSITION = {"yx": 2, "hwl": 3, "theta": 1, "category": 1, "anchor_ind": 1}
    PRED_3D_MAIN_COMPOSITION = params.TrainParams.get_3d_pred_composition(IOU_AWARE)
    PRED_HEAD_COMPOSITION = params.TrainParams.get_pred_composition(IOU_AWARE, True)
    PRED_3D_HEAD_COMPOSITION = params.TrainParams.get_3d_pred_composition(IOU_AWARE, True)

    GRTR_NMS_COMPOSITION = {"yxhw": 4, "z": 1, "object": 1, "category": 1}
    PRED_NMS_COMPOSITION = {"yxhw": 4, "z": 1, "object": 1, "category": 1, "ctgr_prob": 1, "score": 1, "anchor_ind": 1}

    GRTR_3D_NMS_COMPOSITION = {"yx": 2, "hwl": 3, "theta": 1, "category": 1}
    PRED_3D_NMS_COMPOSITION = {"yx": 2, "hwl": 3, "z": 1, "theta": 1, "category": 1}

    NUM_MAIN_CHANNELS = sum(PRED_MAIN_COMPOSITION.values())


class Architecture:
    BACKBONE = ["Resnet", "Darknet53", "CSPDarknet53", "Efficientnet"][2]
    NECK = ["FPN", "PAN", "BiFPN"][1]
    HEAD = ["Single", "Double", "Efficient"][0]
    BACKBONE_CONV_ARGS = {"activation": "mish", "scope": "back"}
    NECK_CONV_ARGS = {"activation": "leaky_relu", "scope": "neck"}
    # HEAD_CONV_ARGS = {"activation": False, "scope": "head"}
    HEAD_CONV_ARGS = {"activation": "leaky_relu", "scope": "head"}
    USE_SPP = True
    COORD_CONV = True
    SIGMOID_DELTA = 0.2

    class Resnet:
        LAYER = {50: ('BottleneckBlock', (3, 4, 6, 3)),
                 101: ('BottleneckBlock', (3, 4, 23, 3)),
                 152: ('BottleneckBlock', (3, 8, 36, 3))
                 }[50]
        CHENNELS = [64, 128, 256, 512, 1024, 2048]

    class Efficientnet:
        NAME = "EfficientNetB2"
        Channels = {"EfficientNetB0": (64, 3, 3), "EfficientNetB1": (88, 4, 3),
                    "EfficientNetB2": (112, 5, 3), "EfficientNetB3": (160, 6, 4),
                    "EfficientNetB4": (224, 7, 4), "EfficientNetB5": (288, 7, 4),
                    "EfficientNetB6": (384, 8, 5)}[NAME]
        Separable = [True, False][1]


class Train:
    CKPT_NAME = "test_1103_v1"
    MODE = ["eager", "graph", "distribute"][1]
    AUGMENT_PROBS = None
    # AUGMENT_PROBS = {"Flip": 1.0, "Blur": 0.2}
    # AUGMENT_PROBS = {"ColorJitter": 0.5, "Flip": 1.0, "CropResize": 1.0, "Blur": 0.2}
    DATA_BATCH_SIZE = 1
    BATCH_SIZE = DATA_BATCH_SIZE * 2 if AUGMENT_PROBS else DATA_BATCH_SIZE
    GLOBAL_BATCH = BATCH_SIZE
    TRAINING_PLAN = params.TrainingPlan.KITTI_SIMPLE
    DETAIL_LOG_EPOCHS = list(range(10, 200, 50))
    IGNORE_MASK = False
    # AUGMENT_PROBS = {"Flip": 0.2}

    # LOG_KEYS: select options in ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
    LOG_KEYS = ["pred_score"]
    USE_EMA = [True, False][1]
    EMA_DECAY = 0.9998
    INTRINSIC = np.zeros([3, 4])


class Scheduler:
    MIN_LR = 1e-10
    CYCLE_STEPS = 10000
    WARMUP_EPOCH = 0
    LOG = [True, False][0]


class FeatureDistribPolicy:
    POLICY_NAME = ["SinglePositivePolicy", "FasterRCNNPolicy", "MultiPositivePolicy"][0]
    IOU_THRESH = [0.5, 0.3]
    CENTER_RADIUS = [2.5, 2.5]
    MULTI_POSITIVE_WIEGHT = 0.8


class AnchorGeneration:
    ANCHOR_STYLE = "YoloxAnchor"
    # ANCHORS = np.array([[[40, 34], [58, 71], [123, 42]], [[79, 118], [125, 167], [239, 89]], [[161, 246], [260, 253], [261, 381]]])
    ANCHORS = None
    MUL_SCALES = [scale / 8 for scale in ModelOutput.FEATURE_SCALES]

    class YoloAnchor:
        BASE_ANCHOR = [80., 120.]
        ASPECT_RATIO = [0.2, 1., 2.]
        SCALES = [1]

    class RetinaNetAnchor:
        BASE_ANCHOR = [20, 20]
        ASPECT_RATIO = [0.5, 1, 2]
        SCALES = [2 ** x for x in [0, 1 / 3, 2 / 3]]

    class YoloxAnchor:
        BASE_ANCHOR = [64, 64]
        ASPECT_RATIO = [1]
        SCALES = [1]


class NmsInfer:
    MAX_OUT = [0, 10, 10, 10, 10, 10]
    IOU_THRESH = [1., 0.5, 0.5, 0.5, 0.5, 0.5]
    SCORE_THRESH = [1, 0.2, 0.2, 0.2, 0., 0.]
    IOU_3D_THRESH = [1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
    SCORE_3D_THRESH = [1, 0.24, 0.24, 0.24, 0.24, 0.24]


class NmsOptim:
    IOU_CANDIDATES = np.arange(0.02, 0.4, 0.02)
    SCORE_CANDIDATES = np.arange(0.02, 0.4, 0.02)
    MAX_OUT_CANDIDATES = np.arange(5, 10, 1)


class Validation:
    TP_IOU_THRESH = [1, 0.5, 0.5, 0.5, 0.5, 0.5]
    VAL_EPOCH = "latest"
    MAP_TP_IOU_THRESH = [0.5]
    MAX_BOX = 200


class Log:
    VISUAL_HEATMAP = True
    # LOSS_NAME = ["box_2d", "object", "category_2d", "category_3d", "yx", "hwl", "depth", "theta"]
    # LOSS_NAME = ["box_2d", "pos_object", "neg_object", "category_2d"]

    class HistoryLog:
        SUMMARY = ["pos_obj"]

    class ExhaustiveLog:
        # DETAIL = ["pos_obj", "iou_mean", "iou_aware", "box_yx", "box_hw", "true_class", "false_class",
        #           "box_yxz", "box_hwl"] \
        #     if ModelOutput.IOU_AWARE else ["pos_obj", "iou_mean", "box_yx", "box_hw", "true_class",
        #                                    "false_class", "box_yxz", "box_hwl"]
        DETAIL = ["pos_obj", "neg_obj", "iou_mean", "iou_aware", "box_yx", "box_hw", "true_class", "false_class",
                  "box_yxz", "box_hwl"] \
            if ModelOutput.IOU_AWARE else ["pos_obj", "neg_obj", "iou_mean", "box_yx", "box_hw", "true_class",
                                           "false_class", "box_yxz", "box_hwl"]
        # COLUMNS_TO_MEAN = ["anchor", "ctgr", "box_2d", "pos_object", "category_2d", "category_3d", "yx",
        #                    "hwl", "depth", "theta", "pos_obj",
        #                    "neg_obj", "iou_mean", "box_hw", "box_yx", "true_class", "false_class", "box_yxz", "box_hwl"]
        COLUMNS_TO_MEAN = ["anchor", "ctgr", "box_2d", "object", "category_2d", "category_3d", "yx",
                           "hwl", "depth", "theta", "pos_obj",
                           "neg_obj", "iou_mean", "box_hw", "box_yx", "true_class", "false_class", "box_yxz", "box_hwl"]
        COLUMNS_TO_SUM = ["anchor", "ctgr", "trpo", "grtr", "pred"]

        # COLUMNS_TO_MEAN = ["anchor", "ctgr", "box_2d", "object", "category_2d",
        #                    "pos_obj",
        #                    "neg_obj", "iou_mean", "box_hw", "box_yx", "true_class", "false_class"]
        # COLUMNS_TO_SUM = ["anchor", "ctgr", "trpo", "grtr", "pred"]
