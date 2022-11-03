import cv2
import numpy as np
import tensorflow as tf

import RIDet3DAddon.config as cfg3d
import utils.tflow.util_function as uf
import RIDet3DAddon.tflow.utils.util_function as uf3d


class NonMaximumSuppression:
    def __init__(self, max_out=cfg3d.NmsInfer.MAX_OUT,
                 iou_thresh=cfg3d.NmsInfer.IOU_THRESH,
                 score_thresh=cfg3d.NmsInfer.SCORE_THRESH,
                 iou_3d_thresh=cfg3d.NmsInfer.IOU_3D_THRESH,
                 score_3d_thresh=cfg3d.NmsInfer.SCORE_3D_THRESH,
                 category_names=cfg3d.Dataloader.CATEGORY_NAMES["category"],
                 ):
        self.max_out = max_out
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.iou_3d_thresh = iou_3d_thresh
        self.score_3d_thresh = score_3d_thresh
        self.category_names = category_names
        self.compute_iou3d = Compute3DIoU()

    def __call__(self, pred, max_out=None, iou_thresh=None, score_thresh=None, merged=False, is_3d=False):
        self.max_out = max_out if max_out is not None else self.max_out
        self.iou_thresh = iou_thresh if iou_thresh is not None else self.iou_thresh
        self.score_thresh = score_thresh if score_thresh is not None else self.score_thresh
        self.iou_3d_thresh = self.iou_3d_thresh
        self.score_3d_thresh = self.score_3d_thresh

        nms_2d_res, nms_3d_res = self.nms2d(pred["feat2d"], pred["feat3d"], merged)
        # if is_3d:
        #     nms_3d_res = self.nms3d(pred["inst3d"], pred["inst2d"], merged)
        #     return nms_3d_res
        return nms_2d_res, nms_3d_res

    # @tf.function
    def nms2d(self, pred2d, pred3d, merged=False):
        """
        :param pred: if merged True, dict of prediction slices merged over scales,
                        {'yxhw': (batch, sum of Nx, 4), 'object': ..., 'category': ...}
                     if merged False, dict of prediction slices for each scale,
                        {'feature_l': {'yxhw': (batch, Nl, 4), 'object': ..., 'category': ...}}
        :param merged
        :return: (batch, max_out, 8), 8: bbox, category, objectness, ctgr_prob, score
        """
        if merged is False:
            pred2d = self.append_anchor_inds(pred2d)
            pred2d = self.merged_scale(pred2d)
            pred3d = self.append_anchor_inds(pred3d)
            pred3d = self.merged_scale(pred3d)

        boxes_2d = uf.convert_box_format_yxhw_to_tlbr(pred2d["yxhw"])  # (batch, N, 4)
        categories_2d = tf.argmax(pred2d["category"], axis=-1)  # (batch, N)
        best_probs_2d = tf.reduce_max(pred2d["category"], axis=-1)  # (batch, N)
        objectness = pred2d["object"][..., 0]  # (batch, N)
        scores_2d = objectness * best_probs_2d  # (batch, N)
        batch, numbox, numctgr = pred2d["category"].shape

        anchor_inds = pred2d["anchor_ind"][..., 0]

        theta = pred3d["theta"][..., 0]
        # categories_3d = tf.argmax(pred2d["category"], axis=-1)  # (batch, N)
        # best_probs_3d = tf.reduce_max(pred2d["category"], axis=-1)  # (batch, N)
        # scores_3d = objectness * best_probs_3d  # (batch, N)

        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(1, numctgr):
            ctgr_mask = tf.cast(categories_2d == ctgr_idx, dtype=tf.float32)  # (batch, N)
            ctgr_boxes = boxes_2d * ctgr_mask[..., tf.newaxis]  # (batch, N, 4)

            ctgr_scores = scores_2d * ctgr_mask  # (batch, N)
            for frame_idx in range(batch):
                selected_indices = tf.image.non_max_suppression(
                    boxes=ctgr_boxes[frame_idx],
                    scores=ctgr_scores[frame_idx],
                    max_output_size=self.max_out[ctgr_idx],
                    iou_threshold=self.iou_thresh[ctgr_idx],
                    score_threshold=self.score_thresh[ctgr_idx],
                )
                # zero padding that works in tf.function
                numsel = tf.shape(selected_indices)[0]
                zero = tf.ones((self.max_out[ctgr_idx] - numsel), dtype=tf.int32) * -1
                selected_indices = tf.concat([selected_indices, zero], axis=0)
                batch_indices[frame_idx].append(selected_indices)

        # make batch_indices, valid_mask as fixed shape tensor
        batch_indices = [tf.concat(ctgr_indices, axis=-1) for ctgr_indices in batch_indices]
        batch_indices = tf.stack(batch_indices, axis=0)  # (batch, K*max_output)
        valid_mask = tf.cast(batch_indices >= 0, dtype=tf.float32)  # (batch, K*max_output)
        batch_indices = tf.maximum(batch_indices, 0)

        # list of (batch, N) -> (batch, N, 4)
        categories_2d = tf.cast(categories_2d, dtype=tf.float32)
        # "bbox": 4, "object": 1, "category": 1, "score": 1, "anchor_inds": 1
        result_2d = tf.stack([objectness, categories_2d, best_probs_2d, scores_2d, anchor_inds], axis=-1)
        result_2d = tf.concat([pred2d["yxhw"], pred2d["z"], result_2d], axis=-1)  # (batch, N, 10)
        result_2d = tf.gather(result_2d, batch_indices, batch_dims=1)  # (batch, K*max_output, 10)
        result_2d = result_2d * valid_mask[..., tf.newaxis]  # (batch, K*max_output, 10)

        # categories_3d = tf.cast(categories_3d, dtype=tf.float32)
        result_3d = tf.stack([theta, categories_2d], axis=-1)
        # result_3d = tf.concat([pred3d["yxz"], pred3d["hwl"], result_3d], axis=-1)  # (batch, N, 11)
        result_3d = tf.concat([pred3d["yx"], pred3d["hwl"], pred2d["z"], result_3d], axis=-1)  # (batch, N, 11)
        result_3d = tf.gather(result_3d, batch_indices, batch_dims=1)  # (batch, K*max_output, 11)
        result_3d = result_3d * valid_mask[..., tf.newaxis]  # (batch, K*max_output, 11)
        return result_2d, result_3d

    def append_anchor_inds(self, pred):
        pred["anchor_ind"] = []
        num_anchor = cfg3d.ModelOutput.NUM_ANCHORS_PER_SCALE
        for scale in range(len(cfg3d.ModelOutput.FEATURE_SCALES)):
            for key in pred:
                if key != "merged":
                    fmap_shape = pred[key][scale].shape[:-1]
                    break
            fmap_shape = (*fmap_shape, 1)

            ones_map = tf.ones(fmap_shape, dtype=tf.float32)
            anchor_list = range(scale * num_anchor, (scale + 1) * num_anchor)
            pred["anchor_ind"].append(self.anchor_indices(ones_map, anchor_list))
        return pred

    def merged_scale(self, pred):
        slice_keys = list(pred.keys())  # ['yxhw', 'object', 'category']
        merged_pred = {}
        # merge pred features over scales
        for key in slice_keys:
            if key != "merged":
                merged_pred[key] = tf.concat(pred[key], axis=1)  # (batch, N, dim)
        return merged_pred

    def anchor_indices(self, ones_map, anchor_list):
        batch, hwa, _ = ones_map.shape
        num_anchor = cfg3d.ModelOutput.NUM_ANCHORS_PER_SCALE
        anchor_index = tf.cast(anchor_list, dtype=tf.float32)[..., tf.newaxis]
        split_anchor_shape = tf.reshape(ones_map, (batch, hwa // num_anchor, num_anchor, 1))

        split_anchor_map = split_anchor_shape * anchor_index
        merge_anchor_map = tf.reshape(split_anchor_map, (batch, hwa, 1))

        return merge_anchor_map

    def compete_diff_categories(self, nms_res, foo_ctgr, bar_ctgr, iou_thresh, score_thresh):
        """
        :param nms_res: (batch, numbox, 10)
        :return:
        """
        batch, numbox = nms_res.shape[:2]
        boxes = nms_res[..., :4]
        category = nms_res[..., 5:6]
        score = nms_res[..., -1:]

        foo_ctgr = self.category_names.index(foo_ctgr)
        bar_ctgr = self.category_names.index(bar_ctgr)
        boxes_tlbr = uf.convert_box_format_yxhw_to_tlbr(boxes)
        batch_survive_mask = []
        for frame_idx in range(batch):
            foo_mask = tf.cast(category[frame_idx] == foo_ctgr, dtype=tf.float32)
            bar_mask = tf.cast(category[frame_idx] == bar_ctgr, dtype=tf.float32)
            target_mask = foo_mask + bar_mask
            target_score_mask = foo_mask + (bar_mask * 0.9)
            target_boxes = boxes_tlbr[frame_idx] * target_mask
            target_score = score[frame_idx] * target_score_mask

            selected_indices = tf.image.non_max_suppression(
                boxes=target_boxes,
                scores=target_score[:, 0],
                max_output_size=20,
                iou_threshold=iou_thresh,
                score_threshold=score_thresh,
            )
            if tf.size(selected_indices) != 0:
                selected_onehot = tf.one_hot(selected_indices, depth=numbox, axis=-1)  # (20, numbox)
                survive_mask = 1 - target_mask + tf.reduce_max(selected_onehot, axis=0)[..., tf.newaxis]  # (numbox,)
                batch_survive_mask.append(survive_mask)

        if len(batch_survive_mask) == 0:
            return nms_res

        batch_survive_mask = tf.stack(batch_survive_mask, axis=0)  # (batch, numbox)
        nms_res = nms_res * batch_survive_mask
        return nms_res

    def nms3d(self, pred3d, pred2d, merged=False):
        """
        :param pred: if merged True, dict of prediction slices merged over scales,
                        {'yxhw': (batch, sum of Nx, 4), 'object': ..., 'category': ...}
                     if merged False, dict of prediction slices for each scale,
                        {'feature_l': {'yxhw': (batch, Nl, 4), 'object': ..., 'category': ...}}
        :param merged
        :return: (batch, max_out, 8), 8: bbox, category, objectness, ctgr_prob, score
        """
        if merged is False:
            pred3d = self.append_anchor_inds(pred3d)
            pred3d = self.merged_scale(pred3d)
            pred2d = self.merged_scale(pred2d)

        categories = tf.argmax(pred3d["category"], axis=-1)  # (batch, N)
        best_probs = tf.reduce_max(pred3d["category"], axis=-1)  # (batch, N)
        objectness = pred2d["object"][..., 0]  # (batch, N)
        scores = objectness * best_probs  # (batch, N)
        batch, numbox, numctgr = pred3d["category"].shape

        # z = pred3d["z"][..., 0]
        theta = pred3d["theta"][..., 0]
        anchor_inds = pred3d["anchor_ind"][..., 0]

        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(0, numctgr):
            ctgr_mask = tf.cast(categories == ctgr_idx, dtype=tf.float32)  # (batch, N)

            ctgr_scores = scores * ctgr_mask  # (batch, N)
            score_mask = tf.cast(ctgr_scores > self.score_thresh[ctgr_idx], tf.bool).numpy()
            for frame_idx in range(batch):
                batch_score = score_mask[frame_idx]
                overlap = self.compute_iou3d(pred3d, pred3d, frame_idx, score_mask[frame_idx], score_mask[frame_idx])
                index = np.where(batch_score == 1)[0]
                overlap_map = np.zeros((batch_score.shape[0], batch_score.shape[0]))
                overlap_map[index[..., np.newaxis], index[np.newaxis, ...]] = overlap
                overlap_score = ctgr_scores[frame_idx].numpy()[batch_score]
                overlap_score_map = np.zeros((batch_score.shape[0]))
                overlap_score_map[index] = overlap_score
                selected_indices = tf.image.non_max_suppression_overlaps(
                    overlaps=overlap_map,
                    scores=overlap_score_map,
                    max_output_size=self.max_out[ctgr_idx],
                    overlap_threshold=self.iou_thresh[ctgr_idx],
                    score_threshold=self.score_thresh[ctgr_idx]
                )
                # zero padding that works in tf.function
                numsel = tf.shape(selected_indices)[0]
                zero = tf.ones((self.max_out[ctgr_idx] - numsel), dtype=tf.int32) * -1
                selected_indices = tf.concat([selected_indices, zero], axis=0)
                batch_indices[frame_idx].append(selected_indices)

        # make batch_indices, valid_mask as fixed shape tensor
        batch_indices = [tf.concat(ctgr_indices, axis=-1) for ctgr_indices in batch_indices]
        batch_indices = tf.stack(batch_indices, axis=0)  # (batch, K*max_output)
        valid_mask = tf.cast(batch_indices >= 0, dtype=tf.float32)  # (batch, K*max_output)
        batch_indices = tf.maximum(batch_indices, 0)

        # list of (batch, N) -> (batch, N, 4)
        categories = tf.cast(categories, dtype=tf.float32)
        # "yxhwl": 5, "z": 1, "theta": 1, "category": 1, "category_probs": 1, "score": 1, "anchor_inds": 1
        result = tf.stack([theta, categories, best_probs, scores, anchor_inds], axis=-1)

        result = tf.concat([pred3d["yxz"], pred3d["hwl"], result], axis=-1)  # (batch, N, 11)
        result = tf.gather(result, batch_indices, batch_dims=1)  # (batch, K*max_output, 11)
        result = result * valid_mask[..., tf.newaxis]  # (batch, K*max_output, 11)
        return result


class Compute3DIoU:
    def __init__(self):
        pass

    def __call__(self, boxes1, boxes2, frame_idx, score_mask_1, score_mask_2):
        hwl_1, xyz_1, theta_1 = self.extract_param(boxes1, frame_idx, score_mask_1)
        hwl_2, xyz_2, theta_2 = self.extract_param(boxes2, frame_idx, score_mask_2)
        iou = self.iou3d_7dof_box(hwl_1[..., 2], hwl_1[..., 0], hwl_1[..., 1], xyz_1, theta_1,
                                  hwl_2[..., 2], hwl_2[..., 0], hwl_2[..., 1], xyz_2, theta_2)
        return iou

    def extract_param(self, boxes, frame_idx, score_mask):
        hwl = boxes["hwl"][frame_idx, :].numpy()[score_mask, :]
        xyz = tf.stack([boxes["yxz"][frame_idx, :, 1], boxes["yxz"][frame_idx, :, 0],
                        boxes["yxz"][frame_idx, :, 2]], axis=-1).numpy()[score_mask, :]
        theta = boxes["theta"][frame_idx, :, 0].numpy()[score_mask]
        return hwl, xyz, theta

    def iou3d_7dof_box(self, boxes1_length, boxes1_height, boxes1_width, boxes1_center,
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


# ==============================================
import matplotlib.pyplot as plt
import os


def test_3d_nms():
    print("======== test_3d_nms")
    pred3d = dict()
    pred2d = dict()

    yxhwl = np.array([[2, 2, 4, 4, 4], [4, 2, 4, 4, 4], [6, 2, 4, 4, 4], [8, 2, 4, 4, 4]], dtype=np.float32)
    z = np.array([[1], [1], [1], [1]], dtype=np.float32)
    ctgr = np.array([[1, 0, 0], [0.5, 0.2, 0.3], [0.8, 0.1, 0], [0, 0.2, 0.9]], dtype=np.float32)
    object = np.array([[0.9], [0.3], [0.8], [0.9]], dtype=np.float32)
    theta = np.zeros((4, 1), dtype=np.float32)
    pred3d["yxhwl"] = yxhwl
    pred3d["z"] = z
    pred3d["category"] = ctgr
    pred2d["object"] = object
    pred3d["theta"] = theta

    for key, value in pred3d.items():
        pred3d[key] = [np.tile(value, (4, 1, 1))]
    uf.print_structure("pred", pred3d)

    nms = NonMaximumSuppression()
    nms_3d_res = nms.nms3d(pred3d, pred2d)
    test = nms_3d_res.numpy()
    print(nms_3d_res.numpy())
    # draw_3d_box(nms_3d_res)


def draw_3d_box(box_3d):
    # box_3d : yxhwl, z, category, theta
    kitti_path = "/home/eagle/mun_workspace/kitti/training"
    image_path = os.path.join(kitti_path, "image_2/000000.png")
    lidar_file = image_path.replace("image_2", "calib").replace(".png", ".txt")
    image = cv2.imread(image_path)
    image = np.zeros_like(image)

    intrinsic = get_intrinsic(lidar_file)

    batch = box_3d.shape[0]
    for frame_idx in range(batch):
        box_3d_per_frame = box_3d[frame_idx]
        valid_box_3d = box_3d_per_frame[box_3d_per_frame > 0]
        corners_3d_cam2 = compute_3d_box_cam2(box_3d_per_frame[..., 2], box_3d_per_frame[..., 3], box_3d_per_frame[..., 4],
                            box_3d_per_frame[..., 1], box_3d_per_frame[..., 0], box_3d_per_frame[..., 5],
                            box_3d_per_frame[..., 7])


def get_intrinsic(lidar_file):
    calib_dict = dict()
    with open(lidar_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            new_line = []
            line = line.split(" ")
            if len(line) == 1:
                pass
            else:
                line[0] = line[0].rstrip(":")
                line[-1] = line[-1].rstrip("\n")
                for a in line[1:]:
                    new_line.append(float(a))
                calib_dict[line[0]] = new_line
        calib_dict["P2"] = np.reshape(np.array(calib_dict["P2"]), (3, 4))
        return calib_dict["P2"].astype(np.float32)


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [tf.zeros(38),tf.zeros(38),tf.zeros(38),tf.zeros(38),-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    test = np.stack([x_corners,y_corners,z_corners], axis=0)
    corners_3d_cam2 = np.dot(R, np.stack([x_corners,y_corners,z_corners], axis=0))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def get_box_corners_3d(boxes_length, boxes_height, boxes_width,
                       boxes_rotation_matrix, boxes_center):
  """Given 3D oriented boxes, computes the box corner positions.
  A 6dof oriented box is fully described by the size (dimension) of the box, and
  its 6DOF pose (R|t). We expect each box pose to be given as 3x3 rotation (R)
  and 3D translation (t) vector pointing to center of the box.
  We expect box pose given as rotation (R) and translation (t) are provided in
  same reference frame we expect to get the box corners. In other words, a point
  in box frame x_box becomes: x = R * x_box + t.
  box_sizes describe size (dimension) of each box along x, y and z direction of
  the box reference frame. Typically these are described as length, width and
  height respectively. So each row of box_sizes encodes [length, width, height].
                 z
                 ^
                 |
             2 --------- 1
            /|          /|
           / |         / |
          3 --------- 0  |
          |  |        |  |    --> y
          |  6 -------|- 5
          | /         | /
          |/          |/
          7 --------- 4
            /
           x
  Args:
    boxes_length: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_height: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_width: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_rotation_matrix: A tf.Tensor of shape (B, 3, 3) with rotation for B
      boxes.
    boxes_center: A tf.Tensor of shape (B, 3) with each row containing 3D
      translation component (t) of the box pose, pointing to center of the box.
  Returns:
    A tf.Tensor of shape (B, 8, 3) containing box corners.
  """
  if len(boxes_length.shape) != 2:
    raise ValueError('Box lengths should be rank 2.')
  if len(boxes_height.shape) != 2:
    raise ValueError('Box heights should be rank 2.')
  if len(boxes_width.shape) != 2:
    raise ValueError('Box widths should be rank 2.')
  if len(boxes_rotation_matrix.shape) != 3:
    raise ValueError('Box rotation matrices should be rank 3.')
  if len(boxes_center.shape) != 2:
    raise ValueError('Box centers should be rank 2.')

  num_boxes = tf.shape(boxes_length)[0]

  # Corners in normalized box frame (unit cube centered at origin)
  corners = tf.constant([
      [0.5, 0.5, 0.5],  # top
      [-0.5, 0.5, 0.5],  # top
      [-0.5, -0.5, 0.5],  # top
      [0.5, -0.5, 0.5],  # top
      [0.5, 0.5, -0.5],  # bottom
      [-0.5, 0.5, -0.5],  # bottom
      [-0.5, -0.5, -0.5],  # bottom
      [0.5, -0.5, -0.5],  # bottom
  ])
  # corners in box frame
  corners = tf.einsum(
      'bi,ji->bji', tf.concat([boxes_length, boxes_width, boxes_height],
                              axis=1), corners)
  # corners after rotation
  corners = tf.einsum('bij,bkj->bki', boxes_rotation_matrix, corners)
  # corners after translation
  corners = corners + tf.reshape(boxes_center, (num_boxes, 1, 3))

  return corners


if __name__ == "__main__":
    test_3d_nms()

