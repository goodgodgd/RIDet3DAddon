import tensorflow as tf
import numpy as np

import RIDet3DAddon.config as cfg3d
import RIDet3DAddon.tflow.config_dir.util_config as uc3d
import model.tflow.model_util as mu


class FeatureDecoder:
    def __init__(self, anchors_per_scale,
                 channel_compos=uc3d.get_channel_composition(False)):
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.anchors_per_scale = anchors_per_scale
        self.num_scale = len(cfg3d.ModelOutput.FEATURE_SCALES)
        self.margin = cfg3d.Architecture.SIGMOID_DELTA
        self.channel_compos = channel_compos

    def decode(self, feature, intrinsic):
        decoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):
            anchors_ratio = self.anchors_per_scale[scale_index]
            box_yx = self.decode_yx(feature["yxhw"][scale_index][..., :2])
            box_hw = self.decode_hw(feature["yxhw"][scale_index][..., 2:4], anchors_ratio)
            decoded["yxhw"].append(tf.concat([box_yx, box_hw], axis=-1))
            decoded["z"].append(tf.exp(feature["z"][scale_index]))
            # decoded["z"].append(1/tf.sigmoid(feature["z"][scale_index]))
            # decoded["z"].append(tf.math.log(feature["z"][scale_index]) / tf.math.log(0.5) * 10)
            decoded["category"].append(tf.sigmoid(feature["category"][scale_index]))

            box3d_in_2d_yx = self.decode_yx3d(feature["yx"][scale_index], decoded["yxhw"][scale_index])
            decoded["yx"].append(self.inverse_proj_to_3d(box3d_in_2d_yx, decoded["z"][scale_index], intrinsic))
            decoded["hwl"].append(self.decode_hwl(feature["hwl"][scale_index]))
            decoded["theta"].append(mu.sigmoid_with_margin(feature["theta"][scale_index], self.margin) * (np.pi / 2))
            decoded["occluded"].append(tf.sigmoid(feature["occluded"][scale_index]))

            if cfg3d.ModelOutput.IOU_AWARE:
                decoded["ioup"].append(tf.sigmoid(feature["ioup"][scale_index]))
                decoded["object"].append(self.obj_post_process(tf.sigmoid(feature["object"][scale_index]),
                                                               decoded["ioup"][scale_index]))
            else:
                decoded["object"].append(tf.sigmoid(feature["object"][scale_index]))
            bbox_pred = [decoded[key][scale_index] for key in self.channel_compos]
            decoded["merged"].append(tf.concat(bbox_pred, axis=-1))
            assert decoded["merged"][scale_index].shape == feature["merged"][scale_index].shape
        return decoded

    def obj_post_process(self, obj, ioup):
        iou_aware_factor = 0.4
        new_obj = tf.pow(obj, (1 - iou_aware_factor)) * tf.pow(ioup, iou_aware_factor)
        return new_obj

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

        # yx_box = tf.sigmoid(yx_raw) * 1.4 - 0.2
        yx_box = mu.sigmoid_with_margin(yx_raw, self.margin)
        # [(batch, grid_h, grid_w, anchor, 2) + (1, grid_h, grid_w, 1, 2)] / (1, 1, 1, 1, 2)
        yx_dec = (yx_box + grid) / divider
        return yx_dec

    def decode_hw(self, hw_raw, anchors_ratio):
        """
        :param hw_raw: (batch, grid_h, grid_w, anchor, 2)
        :param anchors_ratio: [height, width]s of anchors in ratio to image (0~1), (anchor, 2)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        num_anc, channel = anchors_ratio.shape     # (3, 2)
        anchors_tf = tf.reshape(anchors_ratio, (1, 1, 1, num_anc, channel))
        # NOTE: exp activation may result in infinity
        # hw_dec = tf.exp(hw_raw) * anchors_tf
        # hw_dec: 0~3 times of anchor, the delayed sigmoid passes through (0, 1)
        # hw_dec = self.const_3 * tf.sigmoid(hw_raw - self.const_log_2) * anchors_tf
        hw_dec = tf.exp(hw_raw) * anchors_tf
        return hw_dec

    def decode_yx3d(self, yx3d_raw, box2d_decoded):
        yx_in_2dbox = tf.tanh(yx3d_raw)
        yx_dec = box2d_decoded[..., :2] + (yx_in_2dbox * box2d_decoded[..., 2:4] * 0.5)
        return yx_dec

    def inverse_proj_to_3d(self, box_yx, depth, intrinsic):
        box3d_y = depth * ((box_yx[..., :1] - intrinsic[:, tf.newaxis, tf.newaxis, 1:2, 2:3]) /
                           intrinsic[:, tf.newaxis, tf.newaxis, 1:2, 1:2])
        box3d_x = depth * ((box_yx[..., 1:2] - intrinsic[:, tf.newaxis, tf.newaxis, :1, 2:3]) /
                           intrinsic[:, tf.newaxis, tf.newaxis, :1, :1])
        box3d_yx = tf.concat([box3d_y, box3d_x], axis=-1)
        return box3d_yx

    def decode_hwl(self, hwl_raw):
        """
        :param hwl_raw: (batch, grid_h, grid_w, anchor, 3)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 3)
        """
        anchors_tf = tf.reshape([1.], (1, 1, 1, 1, 1))
        hwl_dec = tf.exp(hwl_raw) * anchors_tf
        return hwl_dec


