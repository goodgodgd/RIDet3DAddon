import RIDet3DAddon.config as cfg3d
import RIDet3DAddon.tflow.config_dir.util_config as uc
from utils.tflow.util_function import concat_box_output, merge_dim_hwa, slice_feature, scale_align_featmap


def convert_box_scale_01_to_pixel(boxes_norm):
    """
    :boxes_norm: yxhw format boxes scaled into 0~1
    :return:
    """
    img_res = cfg3d.Datasets.DATASET_CONFIG.INPUT_RESOLUTION
    img_res = [*img_res, *img_res]
    output = [boxes_norm[..., :4] * img_res]
    output = concat_box_output(output, boxes_norm)
    return output


def merge_and_slice_features(featin, is_gt: bool, feat_type: str):
    featout = {}
    if feat_type == "inst2d":
        composition = uc.get_bbox_composition(is_gt)
        featout["merged"] = featin
        featout.update(slice_feature(featin, composition))

    if feat_type == "inst3d":
        composition = uc.get_3d_bbox_composition(is_gt)
        featout["merged"] = featin
        featout.update(slice_feature(featin, composition))

    if feat_type.startswith("feat2d"):
        composition = uc.get_channel_composition(is_gt)
        featout["merged"] = featin
        newfeat = []
        for scale_data in featin:
            newfeat.append(slice_feature(scale_data, composition))
        newfeat = scale_align_featmap(newfeat)
        featout.update(newfeat)

    if feat_type.startswith("feat3d"):
        composition = uc.get_3d_channel_composition(is_gt)
        featout["merged"] = featin
        newfeat = []
        for scale_data in featin:
            newfeat.append(slice_feature(scale_data, composition))
        newfeat = scale_align_featmap(newfeat)
        featout.update(newfeat)
    return featout
