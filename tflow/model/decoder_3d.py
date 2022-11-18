import numpy as np
import tensorflow as tf

import RIDet3DAddon.config as cfg3d
import RIDet3DAddon.tflow.config_dir.util_config as uc3d
import model.tflow.model_util as mu


class FeatureDecoder:
    def __init__(self, channel_compos=uc3d.get_3d_channel_composition(False)):
        self.num_scale = len(cfg3d.ModelOutput.FEATURE_SCALES)
        self.margin = cfg3d.Architecture.SIGMOID_DELTA
        self.channel_compos = channel_compos

    def decode(self, feature, intrinsic, feat_2d):
        decoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):
            box_yx_2d = self.decode_yx(feature["yx"][scale_index], feat_2d["yxhw"][scale_index])
            decoded["hwl"].append(self.decode_hwl(feature["hwl"][scale_index]))
            box_z = 10 * tf.exp(feature["z"][scale_index])
            decoded["yx"].append(self.inverse_proj_to_3d(box_yx_2d, box_z, intrinsic))
            decoded["z"].append(box_z)
            decoded["theta"].append(mu.sigmoid_with_margin(feature["theta"][scale_index], self.margin) * (np.pi / 2))
            decoded["occluded"].append(tf.sigmoid(feature["occluded"][scale_index]))
            # TODO sum 2d category, 3d category
            # object_3d = tf.sigmoid(feature["object"][scale_index])
            # category_3d = tf.sigmoid(feature["category"][scale_index])
            # decoded["category"].append(tf.reduce_mean(tf.stack([feat_2d["category"][scale_index], category_3d], axis=-1), axis=-1))
            # decoded["object"].append(tf.reduce_mean(tf.stack([feat_2d["object"][scale_index], object_3d], axis=-1), axis=-1))
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
