import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
import math
import shapely.affinity
import shapely.geometry

import utils.tflow.util_function as uf
from utils.util_class import MyExceptionToCatch
import RIDet3DAddon.config as cfg


class CustomConv2D:
    CALL_COUNT = -1

    def __init__(self, kernel_size=3, strides=1, padding="same", activation="leaky_relu", scope=None, bn=True,
                 bias_initializer=tf.constant_initializer(0.), kernel_initializer=tf.random_normal_initializer(stddev=0.001)):
        # save arguments for Conv2D layer
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope
        self.bn = bn
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

    def __call__(self, x, filters, name=None):
        CustomConv2D.CALL_COUNT += 1
        index = CustomConv2D.CALL_COUNT
        name = f"conv{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name

        x = layers.Conv2D(filters, self.kernel_size, self.strides, self.padding,
                          use_bias=not self.bn,
                          kernel_regularizer=tf.keras.regularizers.l2(0.001),
                          kernel_initializer=self.kernel_initializer,
                          bias_initializer=self.bias_initializer, name=name
                          )(x)

        if self.bn:
            x = layers.BatchNormalization()(x)

        if self.activation == "leaky_relu":
            x = layers.LeakyReLU(alpha=0.1)(x)
        elif self.activation == "mish":
            x = tfa.activations.mish(x)
        elif self.activation == "relu":
            x = layers.ReLU()(x)
        elif self.activation == "swish":
            x = tf.nn.swish(x)
        elif self.activation is False:
            x = x
        else:
            raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {self.activation}")

        return x


class CustomSeparableConv2D:
    CALL_COUNT = -1

    def __init__(self, kernel_size=3, strides=1, padding="same", activation="leaky_relu", scope=None, bn=True,
                 **kwargs):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope
        self.bn = bn
        self.kwargs = kwargs

    def __call__(self, x, filters, name=None):
        CustomSeparableConv2D.CALL_COUNT += 1
        index = CustomSeparableConv2D.CALL_COUNT
        name = f"spconv{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name
        x = layers.SeparableConv2D(filters, self.kernel_size, self.strides, self.padding,
                                   use_bias=not self.bn, name=name,
                                   **self.kwargs
                                   )(x)

        if self.bn:
            x = layers.BatchNormalization()(x)

        if self.activation == "leaky_relu":
            x = layers.LeakyReLU(alpha=0.1)(x)
        elif self.activation == "mish":
            x = tfa.activations.mish(x)
        elif self.activation == "relu":
            x = layers.ReLU()(x)
        elif self.activation == "swish":
            x = tf.nn.swish(x)
        elif self.activation is False:
            x = x
        else:
            raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {self.activation}")

        return x


class CustomMax2D:
    CALL_COUNT = -1

    def __init__(self, pool_size=3, strides=1, padding="same", scope=None):
        # save arguments for Conv2D layer
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.scope = scope

    def __call__(self, x, name=None):
        CustomMax2D.CALL_COUNT += 1
        index = CustomMax2D.CALL_COUNT
        name = f"maxpool{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name

        x = layers.MaxPooling2D(self.pool_size, self.strides, self.padding, name=name)(x)
        return x


class NonMaximumSuppression:
    def __init__(self, max_out=cfg.NmsInfer.MAX_OUT,
                 iou_thresh=cfg.NmsInfer.IOU_THRESH,
                 score_thresh=cfg.NmsInfer.SCORE_THRESH,
                 category_names=cfg.Dataloader.CATEGORY_NAMES["category"],
                 ):
        self.max_out = max_out
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.category_names = category_names

    def __call__(self, pred, max_out=None, iou_thresh=None, score_thresh=None, merged=False):
        self.max_out = max_out if max_out is not None else self.max_out
        self.iou_thresh = iou_thresh if iou_thresh is not None else self.iou_thresh
        self.score_thresh = score_thresh if score_thresh is not None else self.score_thresh

        nms_2d_res = self.nms2d(pred["feat2d"], merged)
        nms_3d_res = self.nms3d(pred["feat3d"], pred["feat2d"], merged)
        return nms_2d_res, nms_3d_res

    # @tf.function
    def nms2d(self, pred, merged=False):
        """
        :param pred: if merged True, dict of prediction slices merged over scales,
                        {'yxhw': (batch, sum of Nx, 4), 'object': ..., 'category': ...}
                     if merged False, dict of prediction slices for each scale,
                        {'feature_l': {'yxhw': (batch, Nl, 4), 'object': ..., 'category': ...}}
        :param merged
        :return: (batch, max_out, 8), 8: bbox, category, objectness, ctgr_prob, score
        """
        if merged is False:
            pred = self.merged_scale(pred, "object")

        boxes = uf.convert_box_format_yxhw_to_tlbr(pred["yxhw"])  # (batch, N, 4)
        categories = tf.argmax(pred["category"], axis=-1)  # (batch, N)
        best_probs = tf.reduce_max(pred["category"], axis=-1)  # (batch, N)
        objectness = pred["object"][..., 0]  # (batch, N)
        scores = objectness * best_probs  # (batch, N)
        batch, numbox, numctgr = pred["category"].shape

        anchor_inds = pred["anchor_ind"][..., 0]

        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(1, numctgr):
            ctgr_mask = tf.cast(categories == ctgr_idx, dtype=tf.float32)  # (batch, N)
            ctgr_boxes = boxes * ctgr_mask[..., tf.newaxis]  # (batch, N, 4)

            ctgr_scores = scores * ctgr_mask  # (batch, N)
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
        categories = tf.cast(categories, dtype=tf.float32)
        # "bbox": 4, "object": 1, "category": 1, "score": 1, "anchor_inds": 1
        result = tf.stack([objectness, categories, best_probs, scores, anchor_inds], axis=-1)

        result = tf.concat([pred["yxhw"], result], axis=-1)  # (batch, N, 10)
        result = tf.gather(result, batch_indices, batch_dims=1)  # (batch, K*max_output, 10)
        result = result * valid_mask[..., tf.newaxis]  # (batch, K*max_output, 10)
        return result

    def merged_scale(self, pred, key):
        pred["anchor_ind"] = []
        num_anchor = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
        for scale in range(len(cfg.ModelOutput.FEATURE_SCALES)):
            feature_shape = pred[key][scale].shape
            ones_map = tf.ones(feature_shape, dtype=tf.float32)
            s_anchor = scale * num_anchor
            if num_anchor > 1:
                e_anchor = (scale + 1) * num_anchor
            else:
                e_anchor = s_anchor + 1
            pred["anchor_ind"].append(self.anchor_indices(feature_shape, ones_map, range(s_anchor, e_anchor)))

        slice_keys = list(pred.keys())  # ['yxhw', 'object', 'category']
        merged_pred = {}
        # merge pred features over scales
        for key in slice_keys:
            if key != "merged":
                merged_pred[key] = tf.concat(pred[key], axis=1)  # (batch, N, dim)

        return merged_pred

    def anchor_indices(self, feat_shape, ones_map, anchor_list):
        batch, hwa, _ = feat_shape
        num_anchor = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
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

    def nms3d(self, pred, pred2d, merged=False):
        """
        :param pred: if merged True, dict of prediction slices merged over scales,
                        {'yxhw': (batch, sum of Nx, 4), 'object': ..., 'category': ...}
                     if merged False, dict of prediction slices for each scale,
                        {'feature_l': {'yxhw': (batch, Nl, 4), 'object': ..., 'category': ...}}
        :param merged
        :return: (batch, max_out, 8), 8: bbox, category, objectness, ctgr_prob, score
        """
        iou = list()
        if merged is False:
            pred = self.merged_scale(pred, "z")
            pred2d = self.merged_scale(pred2d, "object")

        # self.compute_iou3d(pred, pred2d)

        box_center = tf.concat([pred["yxhwl"][..., :2], pred["z"]], axis=-1)
        best_probs = tf.reduce_max(pred["category"], axis=-1)
        categories = tf.argmax(pred["category"], axis=-1)
        scores = pred2d["object"][..., 0] * best_probs
        batch, numbox, numctgr = pred["category"].shape
        batch_indices = [[] for i in range(batch)]
        for frame_ind in range(batch):
            iou = self.iou3d_7dof_box(pred["yxhwl"][frame_ind, :, 4], pred["yxhwl"][frame_ind, :, 2], pred["yxhwl"][frame_ind, :, 3],
                                box_center[frame_ind, ...], pred["theta"][frame_ind, :, 0], pred["yxhwl"][frame_ind, :, 4],
                                pred["yxhwl"][frame_ind, :, 2], pred["yxhwl"][frame_ind, :, 3], box_center[frame_ind, ...],
                                pred["theta"][frame_ind, :, 0])
            for ctgr_idx in range(1, numctgr):
                selected_indices = tf.image.non_max_suppression_overlaps(iou, scores[frame_ind], self.max_out[ctgr_idx],
                                                                self.iou_thresh[ctgr_idx], self.score_thresh[ctgr_idx])
                numsel = tf.shape(selected_indices)[0]
                zero = tf.ones((self.max_out[ctgr_idx] - numsel), dtype=tf.int32) * -1
                selected_indices = tf.concat([selected_indices, zero], axis=0)
                batch_indices[frame_ind].append(selected_indices)

            batch_indices = tf.stack(batch_indices, axis=0)
            valid_mask = tf.cast(batch_indices >= 0, dtype=tf.float32)
            batch_indices = tf.maximum(batch_indices, 0)
        return iou

    # roll matrix = [[0,0,1], [c, s ,0], [s,c ,0]] = rotation matrix
    # rotation matrix 파악, test돌려서 확인하기
    # TODO test code 작성하

    def compute_iou3d(self, pred, pred2d):
        box_center = tf.concat([pred["yxhwl"][..., :2], pred["z"]], axis=-1)
        best_probs = tf.reduce_max(pred["category"], axis=-1)
        scores = pred2d["object"][..., 0] * best_probs
        batch, numbox, numctgr = pred["category"].shape
        for frame_ind in range(batch):
            iou = self.iou3d_7dof_box(pred["yxhwl"][frame_ind, :, 4], pred["yxhwl"][frame_ind, :, 2], pred["yxhwl"][frame_ind, :, 3],
                                box_center[frame_ind, ...], pred["theta"][frame_ind, :, 0], pred["yxhwl"][frame_ind, :, 4],
                                pred["yxhwl"][frame_ind, :, 2], pred["yxhwl"][frame_ind, :, 3], box_center[frame_ind, ...],
                                pred["theta"][frame_ind, :, 0])

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
        boxes1_volume = self.volume(
            length=boxes1_length, height=boxes1_height, width=boxes1_width)
        boxes1_volume = np.tile(np.expand_dims(boxes1_volume, axis=1), [1, m])
        boxes2_volume = self.volume(
            length=boxes2_length, height=boxes2_height, width=boxes2_width)
        boxes2_volume = np.tile(np.expand_dims(boxes2_volume, axis=0), [n, 1])
        intersection = self.intersection3d_7dof_box(
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

    def intersection3d_7dof_box(self, boxes1_length, boxes1_height, boxes1_width,
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
        boxes1_diag = self.diagonal_length(
            length=boxes1_length, height=boxes1_height, width=boxes1_width)
        boxes2_diag = self.diagonal_length(
            length=boxes2_length, height=boxes2_height, width=boxes2_width)
        dists = self.center_distances(
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
                height_int_i, _ = self._height_metrics(
                    box_center_z=box_i_center_z,
                    box_height=box_i_height,
                    boxes_center_z=boxes2_center_nonempty[:, 2],
                    boxes_height=boxes2_height[non_empty_i])
                rect_int_i = self._get_rectangular_metrics(
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

    def volume(self, length, height, width):
        """Computes volume of boxes.
        Args:
          length: Numpy array with shape [N].
          height: Numpy array with shape [N].
          width: Numpy array with shape [N].
        Returns:
          A Numpy array with shape [N] representing the box volumes.
        """
        return length * height * width

    def diagonal_length(self, length, height, width):
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

    def center_distances(self, boxes1_center, boxes2_center):
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

    def _height_metrics(self, box_center_z, box_height, boxes_center_z, boxes_height):
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

    def _get_rectangular_metrics(self, box_length, box_width, box_center_x, box_center_y,
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
            contour_box = self._get_box_contour(
                length=box_length,
                width=box_width,
                center_x=box_center_x,
                center_y=box_center_y,
                rotation_z_radians=box_rotation_z_radians)
            contours2 = self._get_boxes_contour(
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

    def _get_box_contour(self, length, width, center_x, center_y, rotation_z_radians):
        """Compute shapely contour."""
        c = shapely.geometry.box(-length / 2.0, -width / 2.0, length / 2.0,
                                 width / 2.0)
        rc = shapely.affinity.rotate(c, rotation_z_radians, use_radians=True)
        return shapely.affinity.translate(rc, center_x, center_y)

    def _get_boxes_contour(self, length, width, center_x, center_y, rotation_z_radians):
        """Compute shapely contour."""
        contours = []
        n = length.shape[0]
        for i in range(n):
            contour = self._get_box_contour(
                length=length[i],
                width=width[i],
                center_x=center_x[i],
                center_y=center_y[i],
                rotation_z_radians=rotation_z_radians[i])
            contours.append(contour)
        return contours


class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=np.float32) * -math.log((1 - self.probability) / self.probability)

        return result


# ==============================================

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
    nms.nms3d(pred3d, pred2d)


if __name__ == "__main__":
    test_3d_nms()
