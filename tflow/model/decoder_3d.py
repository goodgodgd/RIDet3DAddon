import math
import tensorflow as tf

import utils.util_function as uf
import config as cfg
import config_dir.util_config as uc


class FeatureDecoder:
    def __init__(self, anchors_per_scale,
                 channel_compos=uc.get_3d_channel_composition(False)):
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.anchors_per_scale = anchors_per_scale
        self.const_3 = tf.constant(3, dtype=tf.float32)
        self.const_log_2 = tf.math.log(tf.constant(2, dtype=tf.float32))
        self.channel_compos = channel_compos
        self.sensitive_value = 1.05

    def __call__(self, feature, scale_ind):
        """
        :param feature: raw feature map predicted by model (batch, grid_h, grid_w, anchor, channel)
        :param scale_ind: scale name e.g. 0~2
        :return: decoded feature in the same shape e.g. (yxhw, objectness, category probabilities)
        """
        slices = uf.slice_feature(feature, self.channel_compos)
        anchors_ratio = self.anchors_per_scale[scale_ind]

        decoded = dict()
        # TODO decode yx inverse projection
        box_yx = self.decode_yx(slices["yxhwl"][..., :2])
        box_hwl = self.decode_hwl(slices["yxhwl"][..., 2:], anchors_ratio)
        decoded["yxhwl"] = tf.concat([box_yx, box_hwl], axis=-1)
        decoded["z"] = 10 * tf.exp(slices["z"])
        decoded["theta"] = (tf.sigmoid(slices["theta"]) * 1.4 - 0.2) * (math.pi / 2)
        decoded["object"] = tf.sigmoid(slices["object"])
        decoded["category"] = tf.sigmoid(slices["category"])

        bbox_pred = [decoded[key] for key in self.channel_compos]
        bbox_pred = tf.concat(bbox_pred, axis=-1)

        assert bbox_pred.shape == feature.shape
        return tf.cast(bbox_pred, dtype=tf.float32)

    def decode_yx(self, yx_raw):
        """
        :param yx_raw: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_dec = yx coordinates of box centers in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        grid_h, grid_w = yx_raw.shape[1:3]
        """
        Original yolo v3 implementation: yx_dec = tf.sigmoid(yx_raw)
        For yx_dec to be close to 0 or 1, yx_raw should be -+ infinity
        By expanding activation range -0.2 ~ 1.4, yx_dec can be close to 0 or 1 from moderate values of yx_raw 
        """
        # grid_x: (grid_h, grid_w)
        grid_x, grid_y = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        # grid: (grid_h, grid_w, 2)
        grid = tf.stack([grid_y, grid_x], axis=-1)
        grid = tf.reshape(grid, (1, grid_h, grid_w, 1, 2))
        grid = tf.cast(grid, tf.float32)
        divider = tf.reshape([grid_h, grid_w], (1, 1, 1, 1, 2))
        divider = tf.cast(divider, tf.float32)

        yx_box = tf.sigmoid(yx_raw) * 1.4 - 0.2
        # [(batch, grid_h, grid_w, anchor, 2) + (1, grid_h, grid_w, 1, 2)] / (1, 1, 1, 1, 2)
        yx_dec = (yx_box + grid) / divider
        return yx_dec

    def decode_hwl(self, hwl_raw, anchors_ratio):
        """
        :param hwl_raw: (batch, grid_h, grid_w, anchor, 3)
        :param anchors_ratio: [height, width]s of anchors in ratio to image (0~1), (anchor, 2)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 3)
        """
        # num_anc, channel = anchors_ratio.shape     # (3, 2)
        anchors_tf = tf.reshape([1.], (1, 1, 1, 1, 1))
        # NOTE: exp activation may result in infinity
        # hw_dec = tf.exp(hw_raw) * anchors_tf
        # hw_dec: 0~3 times of anchor, the delayed sigmoid passes through (0, 1)
        # hw_dec = self.const_3 * tf.sigmoid(hw_raw - self.const_log_2) * anchors_tf
        hwl_dec = tf.exp(hwl_raw) * anchors_tf
        return hwl_dec

