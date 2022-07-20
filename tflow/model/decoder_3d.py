import math
import tensorflow as tf

import utils.tflow.util_function as uf
import RIDet3DAddon.tflow.config_dir.util_config as uc
import RIDet3DAddon.tflow.model.model_util as mu


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
        self.sigmwm = mu.SigmWM()

    def __call__(self, feature, scale_ind, intrinsic, box_2d):
        """
        :param feature: raw feature map predicted by model (batch, grid_h, grid_w, anchor, channel)
        :param scale_ind: scale name e.g. 0~2
        :return: decoded feature in the same shape e.g. (yxhw, objectness, category probabilities)
        """
        slices = uf.slice_feature(feature, self.channel_compos)
        anchors_ratio = self.anchors_per_scale[scale_ind]

        decoded = dict()
        box_yx = self.decode_yx(slices["yxhwl"][..., :2], box_2d)
        box_hwl = self.decode_hwl(slices["yxhwl"][..., 2:], anchors_ratio)
        decoded["z"] = 10 * tf.exp(slices["z"])
        box_yx = self.inv_proj(box_yx, decoded["z"], intrinsic)
        decoded["yxhwl"] = tf.concat([box_yx, box_hwl], axis=-1)
        decoded["theta"] = self.sigmwm(slices["theta"], math.pi/36) * (math.pi / 2)
        decoded["object"] = tf.sigmoid(slices["object"])
        decoded["category"] = tf.nn.softmax(slices["category"])

        bbox_pred = [decoded[key] for key in self.channel_compos]
        bbox_pred = tf.concat(bbox_pred, axis=-1)

        assert bbox_pred.shape == feature.shape
        return tf.cast(bbox_pred, dtype=tf.float32)

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
        # num_anc, channel = anchors_ratio.shape     # (3, 2)
        anchors_tf = tf.reshape([1.], (1, 1, 1, 1, 1))
        # NOTE: exp activation may result in infinity
        # hw_dec = tf.exp(hw_raw) * anchors_tf
        # hw_dec: 0~3 times of anchor, the delayed sigmoid passes through (0, 1)
        # hw_dec = self.const_3 * tf.sigmoid(hw_raw - self.const_log_2) * anchors_tf
        hwl_dec = tf.exp(hwl_raw) * anchors_tf
        return hwl_dec

    def inv_proj(self, box_yx, depth, intrinsic):
        box_y = -depth * ((box_yx[..., :1] - intrinsic[:, tf.newaxis, tf.newaxis, 1:2, 2:3]) /
                          intrinsic[:, tf.newaxis, tf.newaxis, 1:2, 1:2])
        box_x = -depth * ((box_yx[..., 1:2] - intrinsic[:, tf.newaxis, tf.newaxis, :1, 2:3]) /
                          intrinsic[:, tf.newaxis, tf.newaxis, :1, :1])
        box_3d_yx = tf.concat([box_y, box_x], axis=-1)
        return box_3d_yx

