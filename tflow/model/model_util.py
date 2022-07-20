import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
import math

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

        nms_res = self.pure_nms(pred, merged)
        return nms_res

    # @tf.function
    def pure_nms(self, pred, merged=False):
        """
        :param pred: if merged True, dict of prediction slices merged over scales,
                        {'yxhw': (batch, sum of Nx, 4), 'object': ..., 'category': ...}
                     if merged False, dict of prediction slices for each scale,
                        {'feature_l': {'yxhw': (batch, Nl, 4), 'object': ..., 'category': ...}}
        :param merged
        :return: (batch, max_out, 8), 8: bbox, category, objectness, ctgr_prob, score
        """
        if merged is False:
            pred = self.merged_scale(pred)

        boxes = uf.convert_box_format_yxhw_to_tlbr(pred["yxhw"])  # (batch, N, 4)
        categories = tf.argmax(pred["category"], axis=-1)  # (batch, N)
        # if cfg.ModelOutput.MINOR_CTGR:
        #     sign_ctgr = tf.cast(categories == self.category_names.index("Traffic sign"), dtype=tf.float32) * \
        #                 tf.cast(tf.argmax(pred["sign_ctgr"], axis=-1), dtype=tf.float32)
        #     mark_ctgr = tf.cast(categories == self.category_names.index("Road mark"), dtype=tf.float32) * \
        #                 tf.cast(tf.argmax(pred["mark_ctgr"], axis=-1), dtype=tf.float32)
        #     minor_ctgr = sign_ctgr + mark_ctgr
        #     if cfg.ModelOutput.SPEED_LIMIT:
        #         sign_speed = tf.cast(sign_ctgr == self.sign_ctgr.index("TS_SPEED_LIMIT"), dtype=tf.float32) * \
        #                      tf.cast(tf.argmax(pred["sign_speed"], axis=-1), dtype=tf.float32)
        #         mark_speed = tf.cast(mark_ctgr == self.mark_ctgr.index("RM_SPEED_LIMIT"), dtype=tf.float32) * \
        #                      tf.cast(tf.argmax(pred["mark_speed"], axis=-1), dtype=tf.float32)
        #         speed_ctgr = sign_speed + mark_speed
        #     else:
        #         speed_ctgr = tf.zeros_like(categories, dtype=tf.float32)
        # else:
        #     minor_ctgr = tf.zeros_like(categories, dtype=tf.float32)
        #     speed_ctgr = tf.zeros_like(categories, dtype=tf.float32)

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

    def merged_scale(self, pred):
        pred["anchor_ind"] = []
        num_anchor = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
        for scale in range(len(cfg.ModelOutput.FEATURE_SCALES)):
            feature_shape = pred["object"][scale].shape
            ones_map = tf.ones(feature_shape, dtype=tf.float32)
            s_anchor = scale * num_anchor
            if num_anchor > 1:
                e_anchor = (scale + 1) * num_anchor
            else:
                e_anchor = s_anchor + 1
            if scale == 0:
                pred["anchor_ind"].append(self.anchor_indices(feature_shape, ones_map, range(s_anchor, e_anchor)))
            elif scale == 1:
                pred["anchor_ind"].append(self.anchor_indices(feature_shape, ones_map, range(s_anchor, e_anchor)))
            elif scale == 2:
                pred["anchor_ind"].append(self.anchor_indices(feature_shape, ones_map, range(s_anchor, e_anchor)))
            elif scale == 3:
                pred["anchor_ind"].append(self.anchor_indices(feature_shape, ones_map, range(s_anchor, e_anchor)))
            elif scale == 4:
                pred["anchor_ind"].append(self.anchor_indices(feature_shape, ones_map, range(s_anchor, e_anchor)))
        slice_keys = list(pred.keys())  # ['yxhw', 'object', 'category']
        merged_pred = {}
        # merge pred features over scales
        for key in slice_keys:
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


class SigmWM:
    def __init__(self):
        self.l = 0
        self.h = 1

    def __call__(self, x, delta):
        x = tf.sigmoid(x) * (self.h - self.l + 2 * delta) - self.l - delta
        return x
