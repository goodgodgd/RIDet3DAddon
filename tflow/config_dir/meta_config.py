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
        PATH = "/home/eagle/mun_workspace/kitti"
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
    GRTR_MAIN_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1}
    PRED_MAIN_COMPOSITION = params.TrainParams.get_pred_composition(IOU_AWARE)
    GRTR_3D_MAIN_COMPOSITION = {"yxhwl": 5, "z": 1, "theta": 1, "category": 1}
    PRED_3D_MAIN_COMPOSITION = params.TrainParams.get_3d_pred_composition(IOU_AWARE)
    PRED_HEAD_COMPOSITION = params.TrainParams.get_pred_composition(IOU_AWARE, True)
    PRED_3D_HEAD_COMPOSITION = params.TrainParams.get_3d_pred_composition(IOU_AWARE, True)

    GRTR_NMS_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1}
    PRED_NMS_COMPOSITION = {"yxhw": 4, "object": 1, "category": 1, "ctgr_prob": 1, "score": 1, "anchor_ind": 1}

    GRTR_3D_NMS_COMPOSITION = {"yxhwl": 5, "z": 1, "theta": 1, "category": 1}
    PRED_3D_NMS_COMPOSITION = {"yxhw": 4, "category": 1, "ctgr_prob": 1, "score": 1, "anchor_ind": 1}

    NUM_MAIN_CHANNELS = sum(PRED_MAIN_COMPOSITION.values())


class Architecture:
    BACKBONE = ["Resnet", "Darknet53", "CSPDarknet53", "Efficientnet"][1]
    NECK = ["FPN", "PAN", "BiFPN"][1]
    HEAD = ["Single", "Double", "Efficient"][1]
    BACKBONE_CONV_ARGS = {"activation": "mish", "scope": "back"}
    NECK_CONV_ARGS = {"activation": "leaky_relu", "scope": "neck"}
    # HEAD_CONV_ARGS = {"activation": False, "scope": "head"}
    HEAD_CONV_ARGS = {"activation": "leaky_relu", "scope": "head"}
    USE_SPP = [True, False][1]
    COORD_CONV = [True, False][1]
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
    CKPT_NAME = "3D_test"
    MODE = ["eager", "graph", "distribute"][1]
    DATA_BATCH_SIZE = 2
    BATCH_SIZE = DATA_BATCH_SIZE * 2
    GLOBAL_BATCH = BATCH_SIZE
    TRAINING_PLAN = params.TrainingPlan.KITTI_SIMPLE
    DETAIL_LOG_EPOCHS = list(range(0, 100, 10))
    IGNORE_MASK = True
    # AUGMENT_PROBS = {"Flip": 0.2}
    # AUGMENT_PROBS = {"ColorJitter": 0.5, "CropResize": 1.0, "Blur": 0.2}
    AUGMENT_PROBS = None
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
    POLICY_NAME = ["SinglePositivePolicy", "FasterRCNNPolicy", "MultiPositivePolicy"][2]
    IOU_THRESH = [0.5, 0.3]
    CENTER_RADIUS = [2.5, 2.5]
    MULTI_POSITIVE_WIEGHT = 0.8


class AnchorGeneration:
    ANCHOR_STYLE = "YoloxAnchor"
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
        BASE_ANCHOR = [8, 8]
        ASPECT_RATIO = [1]
        SCALES = [1]


class NmsInfer:
    MAX_OUT = [0, 19, 14, 9, 5, 7, 5, 5, 6, 5, 5, 10, 5]
    IOU_THRESH = [0, 0.38, 0.22, 0.24, 0.22, 0.4, 0.1, 0.18, 0.1, 0.1, 0.1, 0.2, 0.4]
    SCORE_THRESH = [1, 0.16, 0.16, 0.1, 0.12, 0.14, 0.38, 0.18, 0.18, 0.16, 0.28, 0.3, 0.14]


class NmsOptim:
    IOU_CANDIDATES = np.arange(0.1, 0.4, 0.02)
    SCORE_CANDIDATES = np.arange(0.02, 0.4, 0.02)
    MAX_OUT_CANDIDATES = np.arange(5, 20, 1)


class Validation:
    TP_IOU_THRESH = [1, 0.4, 0.5, 0.5, 0.5, 0.4, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.5]
    DISTANCE_LIMIT = 25
    VAL_EPOCH = "latest"
    MAP_TP_IOU_THRESH = [0.5]


class Log:
    LOSS_NAME = ["iou", "object", "category"]

    class HistoryLog:
        SUMMARY = ["pos_obj", "neg_obj"]

    class ExhaustiveLog:
        DETAIL = ["pos_obj", "neg_obj", "iou_mean", "iou_aware", "box_yx", "box_hw", "true_class", "false_class"] \
            if ModelOutput.IOU_AWARE else ["pos_obj", "neg_obj", "iou_mean", "box_yx", "box_hw", "true_class",
                                           "false_class"]
        COLUMNS_TO_MEAN = ["anchor", "ctgr", "iou", "object", "category", "pos_obj",
                           "neg_obj", "iou_mean", "box_hw", "box_yx", "true_class", "false_class"]
        COLUMNS_TO_SUM = ["anchor", "ctgr", "trpo", "grtr", "pred"]
