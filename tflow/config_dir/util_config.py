import numpy as np

import RIDet3DAddon.tflow.config_dir.config_generator as cg
import RIDet3DAddon.config as cfg3d


def get_channel_composition(is_gt: bool):
    if is_gt:
        return cfg3d.ModelOutput.GRTR_MAIN_COMPOSITION
    else:
        return cfg3d.ModelOutput.PRED_MAIN_COMPOSITION


def get_3d_channel_composition(is_gt: bool):
    if is_gt:
        return cfg3d.ModelOutput.GRTR_3D_MAIN_COMPOSITION
    else:
        return cfg3d.ModelOutput.PRED_3D_MAIN_COMPOSITION


def get_bbox_composition(is_gt: bool):
    if is_gt:
        return cfg3d.ModelOutput.GRTR_NMS_COMPOSITION
    else:
        return cfg3d.ModelOutput.PRED_NMS_COMPOSITION


def get_3d_bbox_composition(is_gt: bool):
    if is_gt:
        return cfg3d.ModelOutput.GRTR_3D_NMS_COMPOSITION
    else:
        return cfg3d.ModelOutput.PRED_3D_NMS_COMPOSITION


def get_img_shape(code="HW", dataset="kitti", scale_div=1):
    dataset_cfg = cfg3d.Datasets.TARGET_DATASET
    imsize = dataset_cfg.INPUT_RESOLUTION
    code = code.upper()
    if code == "H":
        return imsize[0] // scale_div
    elif code == "W":
        return imsize[1] // scale_div
    elif code == "HW":
        return imsize[0] // scale_div, imsize[1] // scale_div
    elif code == "WH":
        return imsize[1] // scale_div, imsize[0] // scale_div
    elif code == "HWC":
        return imsize[0] // scale_div, imsize[1] // scale_div, 3
    elif code == "BHWC":
        return cfg3d.Train.BATCH_SIZE, imsize[0] // scale_div, imsize[1] // scale_div, 3
    else:
        assert 0, f"Invalid code: {code}"


def get_valid_category_mask(dataset="kitti"):
    """
    :param dataset: dataset name
    :return: binary mask e.g. when
        Dataloader.MAJOR_CATE = ["Person", "Car", "Van", "Bicycle"] and
        Dataset.CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck"]
        Dataset.CATEGORY_REMAP = {"Pedestrian": "Person"}
        this function returns [1 1 1 0] because ["Person", "Car", "Van"] are included in dataset categories
        but "Bicycle" is not
    """
    dataset_cfg = cg.set_dataset_and_get_config(dataset)
    renamed_categories = [dataset_cfg.CATEGORY_REMAP[categ] if categ in dataset_cfg.CATEGORY_REMAP else categ
                          for categ in dataset_cfg.CATEGORIES_TO_USE]
    if dataset == "kitti":
        for i, categ in enumerate(cfg3d.Dataloader.CATEGORY_NAMES["category"]):
            if categ not in renamed_categories:
                renamed_categories.insert(i, categ)

    mask = np.zeros((len(cfg3d.Dataloader.CATEGORY_NAMES["category"]),), dtype=np.int32)
    for categ in renamed_categories:
        if categ in cfg3d.Dataloader.CATEGORY_NAMES["category"]:
            index = cfg3d.Dataloader.CATEGORY_NAMES["category"].index(categ)
            if index < len(cfg3d.Dataloader.CATEGORY_NAMES["category"]):
                mask[index] = 1
    return mask
