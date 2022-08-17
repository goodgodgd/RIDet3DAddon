import numpy as np
import tensorflow as tf

import utils.tflow.util_function as uf
import RIDet3DAddon.config as cfg3d
import RIDet3DAddon.tflow.config_dir.util_config as uc
import model.tflow.model_util as mu


class FeatureDecoder:
    def __init__(self, anchors_per_scale,
                 channel_compos=uc.get_3d_channel_composition(False)):
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.anchors_per_scale = anchors_per_scale
        self.num_scale = len(cfg3d.ModelOutput.FEATURE_SCALES)
        self.margin = cfg3d.Architecture.SIGMOID_DELTA
        # self.const_3 = tf.constant(3, dtype=tf.float32)
        # self.const_log_2 = tf.math.log(tf.constant(2, dtype=tf.float32))
        self.channel_compos = channel_compos

    def decode(self, feature, intrinsic, box_2d):
        decoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):
            anchors_ratio = self.anchors_per_scale[scale_index]
            box_yx = self.decode_yx(feature["yxhwl"][scale_index][..., :2], box_2d[scale_index])
            box_hwl = self.decode_hwl(feature["yxhwl"][scale_index][..., 2:], anchors_ratio)
            decoded["z"].append(10 * tf.exp(feature["z"][scale_index]))
            box_yx = self.decode_inv_proj(box_yx, decoded["z"][scale_index], intrinsic)
            decoded["yxhwl"].append(tf.concat([box_yx, box_hwl], axis=-1))
            decoded["theta"].append(mu.sigmoid_with_margin(feature["theta"][scale_index], self.margin) * (np.pi / 2))
            decoded["category"].append(tf.nn.softmax(feature["category"][scale_index]))
            bbox_pred = [decoded[key][scale_index] for key in self.channel_compos]
            decoded["merged"].append(tf.concat(bbox_pred, axis=-1))
            assert decoded["merged"][scale_index].shape == feature["merged"][scale_index].shape
        return decoded

    def inverse(self, feature, intrinsic, box_2d):
        encoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):
            anchors_ratio = self.anchors_per_scale[scale_index]
            valid_mask = tf.cast(feature["yxhwl"][scale_index][..., :1] > 0, dtype=tf.float32)
            box_yx = self.encode_yx(feature["yxhwl"][scale_index][..., :2], box_2d[scale_index])
            box_hwl = self.encode_hwl(feature["yxhwl"][scale_index][..., 2:], anchors_ratio)
            encoded["z"].append(tf.math.multiply_no_nan(tf.math.log(feature["z"][scale_index]) / 10, valid_mask))
            box_yx = self.encode_inv_proj(box_yx, encoded["z"][scale_index], intrinsic)
            encoded["yxhwl"].append(tf.math.multiply_no_nan(tf.concat([box_yx, box_hwl], axis=-1), valid_mask))
            assert encoded["yxhwl"][scale_index].shape == feature["yxhwl"][scale_index].shape
            assert encoded["z"][scale_index].shape == feature["z"][scale_index].shape
        return encoded

    def decode_yx(self, yx_raw, box_2d):
        """
        :param yx_raw: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_dec = yx coordinates of box centers in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        yx_box = tf.tanh(yx_raw)
        yx_dec = box_2d[..., :2] + (yx_box * box_2d[..., 2:4])
        return yx_dec

    def decode_hwl(self, hwl_raw, anchors_ratio):
        """
        :param hwl_raw: (batch, grid_h, grid_w, anchor, 3)
        :param anchors_ratio: [height, width]s of anchors in ratio to image (0~1), (anchor, 2)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 3)
        """
        # TODO 3D anchor value?
        # num_anc, channel = anchors_ratio.shape  # (3, 2)
        # anchors_tf = tf.reshape(anchors_ratio, (1, 1, 1, num_anc, channel))
        anchors_tf = tf.reshape([1.], (1, 1, 1, 1, 1))
        # NOTE: exp activation may result in infinity
        # hw_dec = tf.exp(hw_raw) * anchors_tf
        # hw_dec: 0~3 times of anchor, the delayed sigmoid passes through (0, 1)
        # hw_dec = self.const_3 * tf.sigmoid(hw_raw - self.const_log_2) * anchors_tf
        hwl_dec = tf.exp(hwl_raw) * anchors_tf
        return hwl_dec

    def decode_inv_proj(self, box_yx, depth, intrinsic):
        box_y = -depth * ((box_yx[..., :1] - intrinsic[:, tf.newaxis, tf.newaxis, 1:2, 2:3]) /
                          intrinsic[:, tf.newaxis, tf.newaxis, 1:2, 1:2])
        box_x = -depth * ((box_yx[..., 1:2] - intrinsic[:, tf.newaxis, tf.newaxis, :1, 2:3]) /
                          intrinsic[:, tf.newaxis, tf.newaxis, :1, :1])
        box_3d_yx = tf.concat([box_y, box_x], axis=-1)
        return box_3d_yx

    def encode_yx(self, yx_dec, box_2d):
        """
        :param yx_dec: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_raw = yx logit (batch, grid_h, grid_w, anchor, 2)
        """
        yx_box = (yx_dec - box_2d[..., :2]) / box_2d[..., 2:4]
        yx_raw = (1 / 2) * tf.math.log((1 - yx_box) / (1 + yx_box))
        return yx_raw

    def encode_hwl(self, hwl_dec, anchors_ratio):
        """
        :param hwl_dec: (batch, grid_h, grid_w, anchor, 3)
        :param anchors_ratio: [height, width]s of anchors in ratio to image (0~1), (anchor, 2)
        :return: hw_raw = heights and widths, length logit (batch, grid_h, grid_w, anchor, 3)
        """
        # num_anc, channel = anchors_ratio.shape  # (3, 2)
        # anchors_tf = tf.reshape(anchors_ratio, (1, 1, 1, num_anc, channel))
        anchors_tf = tf.reshape([1.], (1, 1, 1, 1, 1))
        hwl_raw = tf.math.log(hwl_dec / anchors_tf)
        return hwl_raw

    def encode_inv_proj(self, box_yx, depth, intrinsic):
        box_y = (box_yx[..., :1] / (-depth) * intrinsic[:, tf.newaxis, tf.newaxis, 1:2, 1:2]) + \
                intrinsic[:, tf.newaxis, tf.newaxis, 1:2, 2:3]
        box_x = (box_yx[..., 1:2] / (-depth) * intrinsic[:, tf.newaxis, tf.newaxis, :1, :1]) + \
                intrinsic[:, tf.newaxis, tf.newaxis, :1, 2:3]
        box_raw = tf.concat([box_y, box_x], axis=-1)
        return np.nan_to_num(box_raw)
