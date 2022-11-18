import os.path as op
import RIDet3DAddon.tflow.config_dir.parameter_pool as params
import numpy as np


class Paths:
    RESULT_ROOT = "/home/eagle/mun_workspace"
    DATAPATH = op.join(RESULT_ROOT, "tfrecord")
    CHECK_POINT = op.join(RESULT_ROOT, "ckpt")
    CONFIG_FILENAME = '/home/eagle/mun_workspace/Detector/RILabDetector/RIDet3DAddon/config.py'
    META_CFG_FILENAME = '/home/eagle/mun_workspace/Detector/RILabDetector/RIDet3DAddon/analyze_model/config_dir/meta_config.py'


class Datasets:
    # specific dataset configs MUST have the same items
    class Analyze:
        NAME = "analyze"
        PATH = "/home/eagle/mun_workspace/ckpt/test_1115_v1/result/analysis"
        CATEGORIES_TO_USE = ["Pedestrian", "Car", "Cyclist"]
        CATEGORY_REMAP = {}

    DATASET_CONFIG = None
    TARGET_DATASET = "analyze"


class Dataloader:
    DATASETS_FOR_TFRECORD = {
        "analyze": ("train", "val"),
    }
    MAX_BBOX_PER_IMAGE = 50
    MAX_DONT_PER_IMAGE = 50

    CATEGORY_NAMES = params.TfrParams.CATEGORY_NAMES
    SHARD_SIZE = 2000
    ANCHORS_PIXEL = None

