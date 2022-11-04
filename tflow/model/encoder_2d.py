import numpy as np
import tensorflow as tf

import RIDet3DAddon.config as cfg3d
import RIDet3DAddon.tflow.config_dir.util_config as uc
import model.tflow.model_util as mu


class FeatureEncoder:
    def __init__(self, anchors_per_scale,
                 channel_compos=uc.get_channel_composition(False)):
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.anchors_per_scale = anchors_per_scale
        self.num_scale = len(cfg3d.ModelOutput.FEATURE_SCALES)
        self.margin = cfg3d.Architecture.SIGMOID_DELTA
        self.channel_compos = channel_compos

    def inverse(self, feature):
        encoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):
            valid_mask = tf.cast(feature["yxhw"][scale_index][..., :1] > 0, dtype=tf.float32)
            anchors_ratio = self.anchors_per_scale[scale_index]
            box_yx = self.encode_yx(feature["yxhw"][scale_index][..., :2], valid_mask)
            box_hw = self.encode_hw(feature["yxhw"][scale_index][..., 2:4], anchors_ratio, valid_mask)

            # if not np.isfinite(box_hw.numpy()).all():
            #     test = feature["yxhw"][scale_index].numpy()
            #     i = np.where(box_hw.numpy() == -np.inf)
            #     print(feature["yxhw"][scale_index].numpy()[np.where(box_hw.numpy() == -np.inf)])
            encoded["yxhw"].append(tf.concat([box_yx, box_hw], axis=-1))
            assert encoded["yxhw"][scale_index].shape == feature["yxhw"][scale_index].shape
        return encoded

    def encode_yx(self, decode_yx, valid_mask):
        """
        :param decode_yx: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_raw = yx logit (batch, grid_h, grid_w, anchor, 2)
        """

        grid_h, grid_w = decode_yx.shape[1:3]
        # grid_x: (grid_h, grid_w)
        grid_x, grid_y = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        # grid: (grid_h, grid_w, 2)
        grid = tf.stack([grid_y, grid_x], axis=-1)
        grid = tf.reshape(grid, (1, grid_h, grid_w, 1, 2))
        grid = tf.cast(grid, tf.float32)
        divider = tf.reshape([grid_h, grid_w], (1, 1, 1, 1, 2))
        divider = tf.cast(divider, tf.float32)
        # yx_dec = (yx_box + grid) / divider
        yx_box = (decode_yx * divider - grid)
        yx_box *= valid_mask
        yx_raw = mu.inv_sigmoid_with_margin(yx_box, self.margin)
        assert np.isfinite(yx_raw).all(), yx_raw
        return yx_raw

    def encode_hw(self, decode_hw, anchors_ratio, valid_mask):
        num_anc, channel = anchors_ratio.shape  # (3, 2)
        anchors_tf = tf.reshape(anchors_ratio, (1, 1, 1, num_anc, channel))
        hw_raw = tf.math.log(decode_hw / anchors_tf)
        # divide nan problem
        hw_raw = tf.math.multiply_no_nan(hw_raw, valid_mask)

        return hw_raw