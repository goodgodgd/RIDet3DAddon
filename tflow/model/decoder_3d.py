import numpy as np
import tensorflow as tf

import RIDet3DAddon.config as cfg3d
import RIDet3DAddon.tflow.config_dir.util_config as uc
import model.tflow.model_util as mu


class FeatureDecoder:
    def __init__(self, channel_compos=uc.get_3d_channel_composition(False)):
        self.num_scale = len(cfg3d.ModelOutput.FEATURE_SCALES)
        self.margin = cfg3d.Architecture.SIGMOID_DELTA
        self.channel_compos = channel_compos

    def decode(self, feature, intrinsic, box_2d):
        decoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):
            box_yx_2d = self.decode_yx(feature["yxhwl"][scale_index][..., :2], box_2d[scale_index])
            box_hwl = self.decode_hwl(feature["yxhwl"][scale_index][..., 2:])
            decoded["z"].append(10 * tf.exp(feature["z"][scale_index]))
            box_yx_3d = self.inverse_proj_to_3d(box_yx_2d, decoded["z"][scale_index], intrinsic)
            decoded["yxhwl"].append(tf.concat([box_yx_3d, box_hwl], axis=-1))
            decoded["theta"].append(mu.sigmoid_with_margin(feature["theta"][scale_index], self.margin) * (np.pi / 2))
            decoded["category"].append(tf.sigmoid(feature["category"][scale_index]))
            bbox_pred = [decoded[key][scale_index] for key in self.channel_compos]
            decoded["merged"].append(tf.concat(bbox_pred, axis=-1))
            assert decoded["merged"][scale_index].shape == feature["merged"][scale_index].shape
        return decoded

    def decode_yx(self, yx_raw, box_2d):
        """
        :param yx_raw: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_dec = yx coordinates of box centers in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        yx_in_2dbox = tf.tanh(yx_raw)
        yx_dec = box_2d[..., :2] + (yx_in_2dbox * box_2d[..., 2:4] * 0.5)
        return yx_dec

    def decode_hwl(self, hwl_raw):
        """
        :param hwl_raw: (batch, grid_h, grid_w, anchor, 3)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 3)
        """
        anchors_tf = tf.reshape([1.], (1, 1, 1, 1, 1))
        hwl_dec = tf.exp(hwl_raw) * anchors_tf
        return hwl_dec

    def inverse_proj_to_3d(self, box_yx, depth, intrinsic):
        box3d_y = depth * ((box_yx[..., :1] - intrinsic[:, tf.newaxis, tf.newaxis, 1:2, 2:3]) /
                           intrinsic[:, tf.newaxis, tf.newaxis, 1:2, 1:2])
        box3d_x = depth * ((box_yx[..., 1:2] - intrinsic[:, tf.newaxis, tf.newaxis, :1, 2:3]) /
                           intrinsic[:, tf.newaxis, tf.newaxis, :1, :1])
        box3d_yx = tf.concat([box3d_y, box3d_x], axis=-1)
        return box3d_yx
