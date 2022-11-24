import numpy as np
import utils.tflow.util_function as uf
import RIDet3DAddon.tflow.utils.util_function as uf3d
import config as cfg


def count_true_positives(grtr, pred, num_ctgr, iou_thresh=cfg.Validation.TP_IOU_THRESH, per_class=False):
    """
    :param grtr: slices of features["bboxes"] {'yxhw': (batch, N, 4), 'category': (batch, N)}
    :param pred: slices of nms result {'yxhw': (batch, M, 4), 'category': (batch, M), ...}
    :param num_ctgr: number of categories
    :param iou_thresh: threshold to determine whether two boxes are overlapped
    :param per_class
    :return:
    """
    splits = split_true_false(grtr, pred, iou_thresh)
    # ========== use split instead grtr, pred
    grtr_valid_tp = splits["grtr_tp"]["yxhw"][..., 2:3] > 0
    grtr_valid_fn = splits["grtr_fn"]["yxhw"][..., 2:3] > 0
    pred_valid_tp = splits["pred_tp"]["yxhw"][..., 2:3] > 0
    pred_valid_fp = splits["pred_fp"]["yxhw"][..., 2:3] > 0

    if per_class:
        grtr_tp_count = count_per_class(splits["grtr_tp"], grtr_valid_tp, num_ctgr)
        grtr_fn_count = count_per_class(splits["grtr_fn"], grtr_valid_fn, num_ctgr)
        pred_tp_count = count_per_class(splits["pred_tp"], pred_valid_tp, num_ctgr)
        pred_fp_count = count_per_class(splits["pred_fp"], pred_valid_fp, num_ctgr)

        return {"trpo": pred_tp_count, "grtr": (grtr_tp_count + grtr_fn_count),
                "pred": (pred_tp_count + pred_fp_count)}
    else:
        grtr_count = np.sum(grtr_valid_tp + grtr_valid_fn)
        pred_count = np.sum(pred_valid_tp + pred_valid_fp)
        trpo_count = np.sum(pred_valid_tp)
        return {"trpo": trpo_count, "grtr": grtr_count, "pred": pred_count}


def split_true_false(grtr, pred, iou_thresh):
    splits = split_tp_fp_fn(pred, grtr, iou_thresh)
    return splits


def split_tp_fp_fn(pred, grtr, iou_thresh):
    batch, M, _ = pred["category"].shape
    valid_mask = grtr["object"]
    iou = uf.compute_iou_general(grtr["yxhw"], pred["yxhw"]).numpy()  # (batch, N, M)

    best_iou = np.max(iou, axis=-1)  # (batch, N)

    best_idx = np.argmax(iou, axis=-1)  # (batch, N)
    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["category"], iou_thresh)
    iou_match = best_iou > iou_thresh  # (batch, N)
    pred_ctgr_aligned = np.take_along_axis(pred["category"][..., 0], best_idx, 1)  # (batch, N, 8)

    ctgr_match = grtr["category"][..., 0] == pred_ctgr_aligned  # (batch, N)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_match, axis=-1)  # (batch, N, 1)

    grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)
    grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items()}
    grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items()}
    grtr_tp["iou"] = best_iou * grtr_tp_mask[..., 0]
    grtr_fn["iou"] = best_iou * grtr_fn_mask[..., 0]
    # last dimension rows where grtr_tp_mask == 0 are all-zero
    pred_tp_mask = indices_to_binary_mask(best_idx, grtr_tp_mask, M)
    pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
    pred_tp = {key: val * pred_tp_mask for key, val in pred.items()}
    pred_fp = {key: val * pred_fp_mask for key, val in pred.items()}
    for key in pred_tp.keys():
        pred_tp[key] = np.take_along_axis(pred_tp[key], best_idx[..., np.newaxis], 1)
        pred_fp[key] = np.take_along_axis(pred_fp[key], best_idx[..., np.newaxis], 1)

    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}


def split_dontcare_pred(pred_fp, grtr_dc):
    B, M, _ = pred_fp["category"].shape
    iou_dc = uf.compute_iou_general(grtr_dc["yxhw"], pred_fp["yxhw"])
    best_iou_dc = np.max(iou_dc, axis=-1)  # (batch, D)
    grtr_dc["iou"] = best_iou_dc
    dc_match = np.expand_dims(best_iou_dc > 0.5, axis=-1)  # (batch, D)
    best_idx_dc = np.argmax(iou_dc, axis=-1)
    pred_dc_mask = indices_to_binary_mask(best_idx_dc, dc_match, M)  # (batch, M, 1)
    dc_pred = {key: val * pred_dc_mask for key, val in pred_fp.items()}
    fp_pred = {key: val * (1 - pred_dc_mask) for key, val in pred_fp.items()}
    return fp_pred, dc_pred


def indices_to_binary_mask(best_idx, valid_mask, depth):
    best_idx_onehot = one_hot(best_idx, depth) * valid_mask
    binary_mask = np.expand_dims(np.max(best_idx_onehot, axis=1), axis=-1)  # (batch, M, 1)
    return binary_mask.astype(np.float32)


def get_iou_thresh_per_class(grtr_ctgr, tp_iou_thresh):
    ctgr_idx = grtr_ctgr.astype(np.int32)
    tp_iou_thresh = np.asarray(tp_iou_thresh, np.float32)
    iou_thresh = numpy_gather(tp_iou_thresh, ctgr_idx)
    return iou_thresh[..., 0]


def count_per_class(boxes, mask, num_ctgr):
    """
    :param boxes: slices of object info {'yxhw': (batch, N, 4), 'category': (batch, N), ...}
    :param mask: binary validity mask (batch, N')
    :param num_ctgr: number of categories
    :return: per-class object counts
    """
    boxes_ctgr = boxes["category"][..., 0].astype(np.int32)  # (batch, N')
    boxes_onehot = one_hot(boxes_ctgr, num_ctgr) * mask
    boxes_count = np.sum(boxes_onehot, axis=(0, 1))
    return boxes_count


def count_true_positives_3d(grtr, pred, valid_mask, iou_thresh=cfg.Validation.TP_IOU_THRESH):
    splits = split_tp_fp_fn_3d(grtr, pred, iou_thresh)
    grtr_valid_tp = splits["grtr_tp"]["yxhwl"][..., 2:3] > 0
    grtr_valid_fn = splits["grtr_fn"]["yxhwl"][..., 2:3] > 0
    pred_valid_tp = splits["pred_tp"]["yxhwl"][..., 2:3] > 0
    pred_valid_fp = splits["pred_fp"]["yxhwl"][..., 2:3] > 0
    grtr_count = np.sum(grtr_valid_tp + grtr_valid_fn)
    pred_count = np.sum(pred_valid_tp + pred_valid_fp)
    trpo_count = np.sum(pred_valid_tp)
    return {"trpo3d": trpo_count, "grtr3d": grtr_count, "pred3d": pred_count}


def split_tp_fp_fn_3d(grtr, pred, iou_thresh):
    batch, M, _ = pred["inst2d"]["category"].shape
    valid_mask = grtr["inst2d"]["object"]
    batch_3d_iou = list()

    for frame_idx in range(batch):
        iou3d = compute_3d_iou(grtr, pred, frame_idx, np.ones(grtr["inst3d"]["yx"].shape[1]).astype(np.bool),
                             np.ones(pred["inst3d"]["yx"].shape[1]).astype(np.bool))  # (batch, N, M)
        iou3d[np.where(iou3d == 0.)] = -1
        batch_3d_iou.append(iou3d)
    batch_3d_iou = np.stack(batch_3d_iou, axis=0)
    batch_3d_iou = np.nan_to_num(batch_3d_iou, nan=-1)
    best_3d_iou = np.max(batch_3d_iou, axis=-1)  # (batch, N)

    best_3d_idx = np.argmax(batch_3d_iou, axis=-1)  # (batch, N)
    if len(iou_thresh) > 1:
        iou_thresh = get_iou_thresh_per_class(grtr["inst2d"]["category"], iou_thresh)
    iou_match = best_3d_iou > iou_thresh  # (batch, N)

    pred_ctgr_aligned = np.take_along_axis(pred["inst2d"]["category"][..., 0], best_3d_idx, 1)  # (batch, N, 8)
    pred_oclu_aligned = np.take_along_axis(pred["inst3d"]["occluded"][..., 0], best_3d_idx, 1)  # (batch, N, 8)

    ctgr_match = grtr["inst2d"]["category"][..., 0] == pred_ctgr_aligned  # (batch, N)
    oclu_match = grtr["inst3d"]["occluded"][..., 0] == pred_oclu_aligned  # (batch, N)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_match * oclu_match, axis=-1)  # (batch, N, 1)

    grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)
    grtr_tp = {key: val * grtr_tp_mask for key, val in grtr["inst3d"].items()}
    grtr_fn = {key: val * grtr_fn_mask for key, val in grtr["inst3d"].items()}
    grtr_tp["z"] = grtr["inst2d"]["z"] * grtr_tp_mask
    grtr_fn["z"] = grtr["inst2d"]["z"] * grtr_fn_mask
    grtr_tp["iou"] = best_3d_iou[..., np.newaxis] * grtr_tp_mask
    grtr_fn["iou"] = best_3d_iou[..., np.newaxis] * grtr_fn_mask
    pred_tp_mask = indices_to_binary_mask(best_3d_idx, grtr_tp_mask, M)
    pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
    pred_tp = {key: val * pred_tp_mask for key, val in pred["inst3d"].items()}
    pred_fp = {key: val * pred_fp_mask for key, val in pred["inst3d"].items()}
    for key in pred_tp.keys():
        pred_tp[key] = np.take_along_axis(pred_tp[key], best_3d_idx[..., np.newaxis], 1)
        pred_fp[key] = np.take_along_axis(pred_fp[key], best_3d_idx[..., np.newaxis], 1)

    iou2d = uf.compute_iou_general(grtr["inst2d"]["yxhw"], pred["inst2d"]["yxhw"]).numpy()  # (batch, N, M)
    best_iou2d = np.max(iou2d, axis=-1)
    iou_match_2d = best_iou2d > iou_thresh  # (batch, N)
    best_2d_idx = np.argmax(iou2d, axis=-1)  # (batch, N)
    pred_ctgr_aligned = np.take_along_axis(pred["inst2d"]["category"][..., 0], best_2d_idx, 1)  # (batch, N, 8)# (batch, N)
    ctgr_match = grtr["inst2d"]["category"][..., 0] == pred_ctgr_aligned  # (batch, N)
    grtr_2d_tp_mask = np.expand_dims(iou_match_2d * ctgr_match, axis=-1)  # (batch, N, 1)
    pred_2d_tp_mask = np.take_along_axis(indices_to_binary_mask(best_2d_idx, grtr_2d_tp_mask, M),
                                         best_2d_idx[..., np.newaxis], 1)
    analyze_grtr_fn = {key: val * grtr_2d_tp_mask for key, val in grtr_fn.items()}
    analyze_pred_fp = {key: val * grtr_2d_tp_mask * grtr_fn_mask for key, val in pred_fp.items()}
    # for key in analyze_pred_tp.keys():
    #     analyze_pred_tp[key] = np.take_along_axis(analyze_pred_tp[key], best_2d_idx[..., np.newaxis], 1)

    return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn,
            "analyze_grtr_fn": analyze_grtr_fn, "analyze_pred_fp": analyze_pred_fp}


def compute_3d_iou(boxes1, boxes2, frame_idx, mask_1, mask_2):
    hwl_1, xyz_1, theta_1 = extract_param(boxes1, frame_idx, mask_1)
    hwl_2, xyz_2, theta_2 = extract_param(boxes2, frame_idx, mask_2)
    iou = iou3d_7dof_box(hwl_1[..., 2], hwl_1[..., 0], hwl_1[..., 1], xyz_1, theta_1,
                         hwl_2[..., 2], hwl_2[..., 0], hwl_2[..., 1], xyz_2, theta_2)
    return iou


def extract_param(boxes, frame_idx, score_mask):
    hwl = boxes["inst3d"]["hwl"][frame_idx, :][score_mask]
    xyz = np.stack([boxes["inst2d"]["z"][frame_idx, :, 0], -boxes["inst3d"]["yx"][frame_idx, :, 1], -boxes["inst3d"]["yx"][frame_idx, :, 0]],
                    axis=-1)[score_mask]
    theta = boxes["inst3d"]["theta"][frame_idx, :, 0][score_mask] - np.pi/2
    # xyz = np.stack([boxes["inst3d"]["yx"][frame_idx, :, 1], boxes["inst3d"]["yx"][frame_idx, :, 0], boxes["inst2d"]["z"][frame_idx, :, 0]],
    #               axis=-1)[score_mask]
    # theta = boxes["inst3d"]["theta"][frame_idx, :, 0][score_mask]
    return hwl, xyz, theta


def iou3d_7dof_box(boxes1_length, boxes1_height, boxes1_width, boxes1_center,
                   boxes1_rotation_z_radians, boxes2_length, boxes2_height,
                   boxes2_width, boxes2_center, boxes2_rotation_z_radians):
    """Computes pairwise intersection-over-union between box collections.
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
      A Numpy array with shape [N, M] representing pairwise iou scores.
    """
    n = boxes1_center.shape[0]
    m = boxes2_center.shape[0]
    if n == 0 or m == 0:
        return np.zeros([n, m], dtype=np.float32)
    boxes1_volume = uf3d.volume(
        length=boxes1_length, height=boxes1_height, width=boxes1_width)
    boxes1_volume = np.tile(np.expand_dims(boxes1_volume, axis=1), [1, m])
    boxes2_volume = uf3d.volume(
        length=boxes2_length, height=boxes2_height, width=boxes2_width)
    boxes2_volume = np.tile(np.expand_dims(boxes2_volume, axis=0), [n, 1])
    intersection = uf3d.intersection3d_7dof_box(
        boxes1_length=boxes1_length,
        boxes1_height=boxes1_height,
        boxes1_width=boxes1_width,
        boxes1_center=boxes1_center,
        boxes1_rotation_z_radians=boxes1_rotation_z_radians,
        boxes2_length=boxes2_length,
        boxes2_height=boxes2_height,
        boxes2_width=boxes2_width,
        boxes2_center=boxes2_center,
        boxes2_rotation_z_radians=boxes2_rotation_z_radians)
    union = boxes1_volume + boxes2_volume - intersection
    return intersection / union


def one_hot(grtr_category, category_shape):
    one_hot_data = np.eye(category_shape)[grtr_category.astype(np.int32)]
    return one_hot_data


def numpy_gather(params, index, dim=0):
    if dim is 1:
        batch_list = []
        for i in range(params.shape[0]):
            batch_param = params[i]
            batch_index = index[i]
            batch_gather = np.take(batch_param, batch_index)
            batch_list.append(batch_gather)
        gathar_param = np.stack(batch_list)
    else:
        gathar_param = np.take(params, index)
    return gathar_param

def test():
    # hwl_1 = np.array([[1.41,1.58,4.36]], np.float32)
    hwl_1 = np.array([[1.41,1.58,4.36]], np.float32)
    hwl_2 = np.array([[1.61376,1.66795,4.22737]], np.float32)
    # xyz_1 = np.array([[3.18,1.56,34.38]], np.float32)
    xyz_1 = np.array([[34.38, -3.18,-1.56]], np.float32)
    # xyz_2 = np.array([[3.23946,1.26178,33.05998]], np.float32)
    xyz_2 = np.array([[33.05998, -3.23946, -1.26178,]], np.float32)
    # theta_1 = np.array([[-1.58]], np.float32)
    theta_1 = np.array([[0]], np.float32)
    theta_2 = np.array([[0]], np.float32)
    iou = iou3d_7dof_box(hwl_1[..., 2], hwl_1[..., 0], hwl_1[..., 1], xyz_1, theta_1,
                         hwl_2[..., 2], hwl_2[..., 0], hwl_2[..., 1], xyz_2, theta_2)
    print(iou)

if __name__ == '__main__':
    test()
