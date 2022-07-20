import RIDet3DAddon.config as cfg3d
import RIDet3DAddon.tflow.config_dir.util_config as uc
from utils.tflow.util_function import concat_box_output, merge_dim_hwa, slice_feature, scale_align_featmap


def convert_box_scale_01_to_pixel(boxes_norm):
    """
    :boxes_norm: yxhw format boxes scaled into 0~1
    :return:
    """
    img_res = cfg3d.Datasets.DATASET_CONFIGS.INPUT_RESOLUTION
    img_res = [*img_res, *img_res]
    output = [boxes_norm[..., :4] * img_res]
    output = concat_box_output(output, boxes_norm)
    return output


def merge_and_slice_features(features, is_gt):
    """
    :param features: this dict has keys feature_l,m,s and corresponding tensors are in (batch, grid_h, grid_w, anchors, dims)
    :param is_gt: is ground truth feature map?
    :return: sliced feature maps in each scale
    """
    # scales = [key for key in features if "feature" in key]  # ['feature_l', 'feature_m', 'feature_s']
    # scales += [key for key in features if "featraw" in key]  # ['featraw_l', 'featraw_m', 'featraw_s']
    sliced_features = {"inst": {}, "feat2d": [], "feat3d": []}
    for raw_feat in features["feat2d"]:
        merged_feat = merge_dim_hwa(raw_feat)
        channel_compos = uc.get_channel_composition(is_gt)
        sliced_features["feat2d"].append(slice_feature(merged_feat, channel_compos))

    sliced_features["feat2d"] = scale_align_featmap(sliced_features["feat2d"])

    for raw_feat in features["feat3d"]:
        merged_feat = merge_dim_hwa(raw_feat)
        channel_compos = uc.get_3d_channel_composition(is_gt)
        sliced_features["feat3d"].append(slice_feature(merged_feat, channel_compos))

    sliced_features["feat3d"] = scale_align_featmap(sliced_features["feat3d"])

    # TODO check featraw
    if cfg3d.ModelOutput.FEAT_RAW:
        raw_names = [name for name in features if "raw" in name]
        for raw_name in raw_names:
            raw_sliced = {raw_name: []}
            for raw_feat in features[raw_name]:
                merged_feat = merge_dim_hwa(raw_feat)
                channel_compos = uc.get_channel_composition(is_gt)
                raw_sliced[raw_name].append(slice_feature(merged_feat, channel_compos))
            raw_sliced[raw_name] = scale_align_featmap(raw_sliced[raw_name])
            sliced_features.update(raw_sliced)

    # scales = [key for key in features if "featraw" in key]  # ['feature_l', 'feature_m', 'feature_s']
    # for key in scales:
    #     raw_feat = features[key]
    #     merged_feat = merge_dim_hwa(raw_feat)
    #     channel_compos = cfg.ModelOutput.get_channel_composition(is_gt)
    #     sliced_features[key] = slice_feature(merged_feat, channel_compos)

    if "bboxes2d" in features["inst"]:
        bbox_compos = uc.get_bbox_composition(is_gt)
        sliced_features["inst"]["bboxes2d"] = slice_feature(features["inst"]["bboxes2d"], bbox_compos)

    if "bboxes3d" in features["inst"]:
        bbox_compos = uc.get_3d_bbox_composition(is_gt)
        sliced_features["inst"]["bboxes3d"] = slice_feature(features["inst"]["bboxes3d"], bbox_compos)

    if "dontcare" in features["inst"]:
        bbox_compos = uc.get_bbox_composition(is_gt)
        sliced_features["inst"]["dontcare"] = slice_feature(features["inst"]["dontcare"], bbox_compos)

    other_features = {key: val for key, val in features.items() if key not in sliced_features}
    sliced_features.update(other_features)
    return sliced_features

