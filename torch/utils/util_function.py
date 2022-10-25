import torch
import numpy as np
import shapely.affinity
import shapely.geometry

import RIDet3DAddon.torch.config as cfg3d
import RIDet3DAddon.torch.config_dir.util_config as uc3d
from utils.framework.util_function import concat_box_output, slice_feature, scale_align_featmap


def merge_and_slice_features(featin, is_gt: bool, feat_type: str):
    featout = {}
    if feat_type == "inst2d":
        composition = uc3d.get_bbox_composition(is_gt)
        featout["merged"] = featin
        featout.update(slice_feature(featin, composition, -1))

    if feat_type == "inst3d":
        composition = uc3d.get_3d_bbox_composition(is_gt)
        featout["merged"] = featin
        featout.update(slice_feature(featin, composition, -1))

    if feat_type.startswith("feat2d"):
        composition = uc3d.get_channel_composition(is_gt)
        featout["whole"] = featin
        newfeat = []
        for scale_data in featin:
            newfeat.append(slice_feature(scale_data, composition, 1))
        newfeat = scale_align_featmap(newfeat)
        featout.update(newfeat)

    if feat_type.startswith("feat3d"):

        composition = uc3d.get_3d_channel_composition(is_gt)
        featout["whole"] = featin
        newfeat = []
        for scale_data in featin:
            newfeat.append(slice_feature(scale_data, composition, 1))
        newfeat = scale_align_featmap(newfeat)
        featout.update(newfeat)

    return featout


def convert_tensor_to_numpy(feature):
    if isinstance(feature, dict):
        dict_feat = dict()
        for key, value in feature.items():
            dict_feat[key] = convert_tensor_to_numpy(value)
        return dict_feat
    elif isinstance(feature, list):
        list_feat = []
        for value in feature:
            list_feat.append(convert_tensor_to_numpy(value))
        return list_feat
    else:
        return feature.detach().cpu().numpy()


def convert_box_scale_01_to_pixel(boxes_norm):
    """
    :boxes_norm: yxhw format boxes scaled into 0~1
    :return:
    """
    img_res = cfg3d.Datasets.Kittibev.INPUT_RESOLUTION
    img_res = [*img_res, *img_res]
    output = [boxes_norm[..., :4] * img_res]
    output = concat_box_output(output, boxes_norm)
    return output


def convert_yx_to_xy(box_yx):
    x = box_yx[:, 1, ...]
    y = box_yx[:, 0, ...]
    return torch.stack([x, y], dim=1)


def convert_xy_to_yx(box_yx):
    x = box_yx[:, 0, ...]
    y = box_yx[:, 1, ...]
    return torch.stack([y, x], dim=1)


def intersection3d_7dof_box(boxes1_length, boxes1_height, boxes1_width,
                            boxes1_center, boxes1_rotation_z_radians,
                            boxes2_length, boxes2_height, boxes2_width,
                            boxes2_center, boxes2_rotation_z_radians):
    """Computes intersection between every pair of boxes in the box collections.
    Args:
      boxes1_length: Numpy array with shape [N].
      boxes1_height: Numpy array with shape [N].
      boxes1_width: Numpy array with shape [N].
      boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
        format of [cx, cy, cz].
      boxes1_rotation_z_radians: Numpy array with shape [N].
      boxes2_length: Numpy array with shape [M].
      boxes2_height: Numpy array with shape [M].
      boxes2_width: Numpy array with shape [M].
      boxes2_center: A Numpy array with shape [M, 3] holding M box centers in the
        format of [cx, cy, cz].
      boxes2_rotation_z_radians: Numpy array with shape [M].
    Returns:
      A Numpy array with shape [N, M] representing pairwise intersections.
    """
    n = boxes1_center.shape[0]
    m = boxes2_center.shape[0]
    if n == 0 or m == 0:
        return np.zeros([n, m], dtype=np.float32)
    boxes1_diag = diagonal_length(
        length=boxes1_length, height=boxes1_height, width=boxes1_width)
    boxes2_diag = diagonal_length(
        length=boxes2_length, height=boxes2_height, width=boxes2_width)
    dists = center_distances(
        boxes1_center=boxes1_center, boxes2_center=boxes2_center)

    intersections = []
    for i in range(n):
        box_i_length = boxes1_length[i]
        box_i_height = boxes1_height[i]
        box_i_width = boxes1_width[i]
        box_i_center_x = boxes1_center[i, 0]
        box_i_center_y = boxes1_center[i, 1]
        box_i_center_z = boxes1_center[i, 2]
        box_i_rotation_z_radians = boxes1_rotation_z_radians[i]
        box_diag = boxes1_diag[i]
        dist = dists[i, :]
        non_empty_i = (box_diag + boxes2_diag) >= dist
        intersection_i = np.zeros(m, np.float32)
        if non_empty_i.any():
            boxes2_center_nonempty = boxes2_center[non_empty_i]
            height_int_i, _ = _height_metrics(
                box_center_z=box_i_center_z,
                box_height=box_i_height,
                boxes_center_z=boxes2_center_nonempty[:, 2],
                boxes_height=boxes2_height[non_empty_i])
            rect_int_i = _get_rectangular_metrics(
                box_length=box_i_length,
                box_width=box_i_width,
                box_center_x=box_i_center_x,
                box_center_y=box_i_center_y,
                box_rotation_z_radians=box_i_rotation_z_radians,
                boxes_length=boxes2_length[non_empty_i],
                boxes_width=boxes2_width[non_empty_i],
                boxes_center_x=boxes2_center_nonempty[:, 0],
                boxes_center_y=boxes2_center_nonempty[:, 1],
                boxes_rotation_z_radians=boxes2_rotation_z_radians[non_empty_i])
            intersection_i[non_empty_i] = height_int_i * rect_int_i
        intersections.append(intersection_i)
    return np.stack(intersections, axis=0)


def volume(length, height, width):
    """Computes volume of boxes.
    Args:
      length: Numpy array with shape [N].
      height: Numpy array with shape [N].
      width: Numpy array with shape [N].
    Returns:
      A Numpy array with shape [N] representing the box volumes.
    """
    return length * height * width


def diagonal_length(length, height, width):
    """Computes volume of boxes.
    Args:
      length: Numpy array with shape [N].
      height: Numpy array with shape [N].
      width: Numpy array with shape [N].
    Returns:
      A Numpy array with shape [N] representing the length of the diagonal of
      each box.
    """
    return np.sqrt(np.square(length) + np.square(height) + np.square(width)) / 2


def center_distances(boxes1_center, boxes2_center):
    """Computes pairwise intersection-over-union between box collections.
    Args:
      boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
        format of [cx, cy, cz].
      boxes2_center: A Numpy array with shape [M, 3] holding M box centers in the
        format of [cx, cy, cz].
    Returns:
      A Numpy array with shape [N, M] representing pairwise center distances.
    """
    n = boxes1_center.shape[0]
    m = boxes2_center.shape[0]
    boxes1_center = np.tile(np.expand_dims(boxes1_center, axis=1), [1, m, 1])
    boxes2_center = np.tile(np.expand_dims(boxes2_center, axis=0), [n, 1, 1])
    return np.sqrt(np.sum(np.square(boxes2_center - boxes1_center), axis=2))


def _height_metrics(box_center_z, box_height, boxes_center_z, boxes_height):
    """Compute 3D height intersection and union between a box and a list of boxes.
    Args:
      box_center_z: A scalar.
      box_height: A scalar.
      boxes_center_z: A Numpy array of size [N].
      boxes_height: A Numpy array of size [N].
    Returns:
      height_intersection: A Numpy array containing the intersection along
        the gravity axis between the two bounding boxes.
      height_union: A Numpy array containing the union along the gravity
        axis between the two bounding boxes.
    """
    min_z_boxes = boxes_center_z - boxes_height / 2.0
    max_z_boxes = boxes_center_z + boxes_height / 2.0
    max_z_box = box_center_z + box_height / 2.0
    min_z_box = box_center_z - box_height / 2.0
    max_of_mins = np.maximum(min_z_box, min_z_boxes)
    min_of_maxs = np.minimum(max_z_box, max_z_boxes)
    offsets = min_of_maxs - max_of_mins
    height_intersection = np.maximum(0, offsets)
    height_union = (
            np.maximum(min_z_box, max_z_boxes) - np.minimum(min_z_box, min_z_boxes) -
            np.maximum(0, -offsets))
    return height_intersection, height_union


def _get_rectangular_metrics(box_length, box_width, box_center_x, box_center_y,
                             box_rotation_z_radians, boxes_length, boxes_width,
                             boxes_center_x, boxes_center_y,
                             boxes_rotation_z_radians):
    """Computes the intersection of the bases of 3d boxes.
    Args:
      box_length: A float scalar.
      box_width: A float scalar.
      box_center_x: A float scalar.
      box_center_y: A float scalar.
      box_rotation_z_radians: A float scalar.
      boxes_length: A np.float32 Numpy array of size [N].
      boxes_width: A np.float32 Numpy array of size [N].
      boxes_center_x: A np.float32 Numpy array of size [N].
      boxes_center_y: A np.float32 Numpy array of size [N].
      boxes_rotation_z_radians: A np.float32 Numpy array of size [N].
    Returns:
      intersection: A Numpy array containing intersection between the
        base of box and all other boxes.
    """
    m = boxes_length.shape[0]
    intersections = np.zeros([m], dtype=np.float32)
    try:
        contour_box = _get_box_contour(
            length=box_length,
            width=box_width,
            center_x=box_center_x,
            center_y=box_center_y,
            rotation_z_radians=box_rotation_z_radians)
        contours2 = _get_boxes_contour(
            length=boxes_length,
            width=boxes_width,
            center_x=boxes_center_x,
            center_y=boxes_center_y,
            rotation_z_radians=boxes_rotation_z_radians)
        for j in range(m):
            intersections[j] = contour_box.intersection(contours2[j]).area
    except Exception as e:  # pylint: disable=broad-except
        raise MyExceptionToCatch('Error calling shapely : {}'.format(e))
    return intersections


def _get_box_contour(length, width, center_x, center_y, rotation_z_radians):
    """Compute shapely contour."""
    c = shapely.geometry.box(-length / 2.0, -width / 2.0, length / 2.0,
                             width / 2.0)
    rc = shapely.affinity.rotate(c, rotation_z_radians, use_radians=True)
    return shapely.affinity.translate(rc, center_x, center_y)


def _get_boxes_contour(length, width, center_x, center_y, rotation_z_radians):
    """Compute shapely contour."""
    contours = []
    n = length.shape[0]
    for i in range(n):
        contour = _get_box_contour(
            length=length[i],
            width=width[i],
            center_x=center_x[i],
            center_y=center_y[i],
            rotation_z_radians=rotation_z_radians[i])
        contours.append(contour)
    return contours
