import tensorflow as tf

import utils.tflow.util_function as uf
import RIDet3DAddon.config as cfg3d
import RIDet3DAddon.tflow.config_dir.util_config as uc
import RIDet3DAddon.tflow.model.model_util as mu


class FeatureDecoder:
    def __init__(self, anchors_per_scale,
                 channel_compos=uc.get_channel_composition(False)):
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.anchors_per_scale = anchors_per_scale
        self.const_3 = tf.constant(3, dtype=tf.float32)
        self.const_log_2 = tf.math.log(tf.constant(2, dtype=tf.float32))
        self.channel_compos = channel_compos
        self.sigmwm = mu.SigmWM()

    def __call__(self, feature, scale_ind):
        """
        :param feature: raw feature map predicted by model (batch, grid_h, grid_w, anchor, channel)
        :param scale_ind: scale name e.g. 0~2
        :return: decoded feature in the same shape e.g. (yxhw, objectness, category probabilities)
        """
        slices = uf.slice_feature(feature, self.channel_compos)
        anchors_ratio = self.anchors_per_scale[scale_ind]

        decoded = dict()
        box_yx = self.decode_yx(slices["yxhw"][..., :2])
        box_hw = self.decode_hw(slices["yxhw"][..., 2:], anchors_ratio)
        decoded["yxhw"] = tf.concat([box_yx, box_hw], axis=-1)
        decoded["category"] = tf.nn.softmax(slices["category"])
        if cfg3d.ModelOutput.IOU_AWARE:
            decoded["ioup"] = tf.sigmoid(slices["ioup"])
            decoded["object"] = self.obj_post_process(tf.sigmoid(slices["object"]), decoded["ioup"])
        else:
            decoded["object"] = tf.sigmoid(slices["object"])

        bbox_pred = [decoded[key] for key in self.channel_compos]
        bbox_pred = tf.concat(bbox_pred, axis=-1)

        assert bbox_pred.shape == feature.shape
        return tf.cast(bbox_pred, dtype=tf.float32)

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
        yx_box = self.sigmwm(yx_raw, 0.2)
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

