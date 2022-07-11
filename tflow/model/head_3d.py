import tensorflow as tf

from model.model_util import PriorProbability
from utils.util_class import MyExceptionToCatch
import config as cfg
import model.model_util as mu
import utils.util_function as uf
import config_dir.util_config as uc


def head_factory(output_name, conv_args, training, num_anchors_per_scale, pred_composition):
    if output_name == "Double":
        return DoubleOutput3d(conv_args, num_anchors_per_scale, pred_composition)
    elif output_name == "Single":
        return SingleOutput3d(conv_args, num_anchors_per_scale, pred_composition)

    else:
        raise MyExceptionToCatch(f"[head_factory[ invalid output name : {output_name}")


class HeadBase:
    def __init__(self, conv_args, num_anchors_per_scale, pred_composition):
        self.conv2d = mu.CustomConv2D(kernel_size=3, strides=1, **conv_args)
        self.conv2d_k1 = mu.CustomConv2D(kernel_size=1, strides=1, **conv_args)
        self.conv2d_s2 = mu.CustomConv2D(kernel_size=3, strides=2, **conv_args)
        self.conv2d_k1na = mu.CustomConv2D(kernel_size=1, strides=1, activation=False, bn=False, scope="head")
        self.conv2d_output = mu.CustomConv2D(kernel_size=1, strides=1, activation=False, scope="output", bn=False)
        self.pool = tf.keras.layers.AvgPool2D(pool_size=(3, 3), padding='valid', strides=(1, 1))
        self.num_anchors_per_scale = num_anchors_per_scale
        self.pred_composition = pred_composition

    def feature_align(self, features, decode):
        aligned_feature = dict()
        for scale, (key, feature) in enumerate(features.items()):
            b, h, w, c = feature.shape
            instance_bbox = tf.reshape(uf.slice_feature(decode[scale], uc.get_channel_composition(False))["yxhw"],
                                       (b*h*w, -1))
            # TODO batch_map create broadcast
            batch_map = list()
            for i in tf.range(b):
                batch_map.append(tf.fill(h*w, i))
            batch_map = tf.concat(batch_map, axis=0)
            instance_bbox = uf.convert_box_format_yxhw_to_tlbr(instance_bbox)
            # feature : [1, h, w, c]
            # instance_bbox [b*h*w, c]
            # batch_map : tf.zeros(feature.shape[0])
            align_feature = tf.image.crop_and_resize(feature, instance_bbox, batch_map, (3, 3))
            pool_feature = self.pool(align_feature)
            # align_feature [b*h*w, crop_height, crop_width, c]
            aligned_feature[key] = tf.reshape(pool_feature, (b, h, w, c))

        return aligned_feature


class SingleOutput3d(HeadBase):
    def __init__(self, conv_args, num_anchors_per_scale, pred_composition):
        super().__init__(conv_args, num_anchors_per_scale, pred_composition)
        self.out_channels = sum(pred_composition.values())

    def __call__(self, input_features, decode_features):
        aligned_features = self.feature_align(input_features, decode_features)
        small = aligned_features["feature_3"]
        conv_sbbox = self.make_output(small, 256)
        medium = aligned_features["feature_4"]
        conv_mbbox = self.make_output(medium, 512)
        large = aligned_features["feature_5"]
        conv_lbbox = self.make_output(large, 1024)
        output_features = {"feature_3": conv_sbbox, "feature_4": conv_mbbox, "feature_5": conv_lbbox}
        return output_features

    def make_output(self, x, channel):
        x = self.conv2d(x, channel)
        x = self.conv2d_output(x, self.num_anchors_per_scale * self.out_channels)
        batch, height, width, channel = x.shape
        x_5d = tf.reshape(x, (batch, height, width, self.num_anchors_per_scale, self.out_channels))
        return x_5d


class DoubleOutput3d(HeadBase):
    def __init__(self, conv_args, num_anchors_per_scale, pred_composition):
        super().__init__(conv_args, num_anchors_per_scale, pred_composition)

    def __call__(self, input_features, decode_features):
        features = self.feature_align(input_features, decode_features)
        output_features = {}
        for scale, feature in features.items():
            conv_common = self.conv2d_k1(feature, 256)
            features = []
            for key, channel in self.pred_composition.items():
                conv_out = self.conv2d(conv_common, 256)
                conv_out = self.conv2d(conv_out, 256)
                feat = self.conv2d_k1na(conv_out, channel * self.num_anchors_per_scale)
                features.append(feat)
            b, h, w, c = features[0].shape
            output_features[scale] = tf.concat(features, axis=-1)
            output_features[scale] = tf.reshape(output_features[scale], (b, h, w, self.num_anchors_per_scale, -1))
        return output_features


# ============================

def test_align():
    import numpy as np
    import cv2

    # features = np.ones((1, 10, 10), dtype=np.float32)
    features = np.arange(0, 800).reshape((2, 10, 10, 4)).astype(np.float32)
    # features = np.stack([features,features,features,features], axis=-1)
    x = np.linspace(0, 9, 10)
    y = np.linspace(0, 9, 10)
    x_grid, y_grid = np.meshgrid(x, y)
    box_features = np.stack((y_grid, x_grid), axis=-1) + 0.5
    hw = np.ones_like(box_features) * 2
    box_features = np.concatenate([box_features, hw], axis=-1) / 10
    remain_feature = np.zeros_like(box_features)
    box_features = np.concatenate([box_features, remain_feature], axis=-1)[np.newaxis, ..., np.newaxis, :].repeat(1, axis=0)

    head3d_model = head_factory("Double", {"activation": "leaky_relu", "scope": "head"}, True, 1,
                                cfg.ModelOutput.PRED_3D_HEAD_COMPOSITION)
    dict_feat = {"feat": features}
    list_box_feat = list()
    list_box_feat.append(box_features.astype(np.float32))
    aligned_features = head3d_model.feature_align(dict_feat, list_box_feat)
    aligned_feat = aligned_features["feat"].numpy()
    y, x = 2, 2
    feat_around_yx = features[0, y - 2:y + 3, x - 2:x + 3, 0]
    print(f"original feat at {y},{x}\n", feat_around_yx)
    print(f"aligned_feat at {y},{x}\n", aligned_feat[0, y, x, 0])
    weights = np.array([[0.25, 0.5, 0.25],
               [0.5, 1, 0.5],
               [0.25, 0.5, 0.25]], dtype=np.float32)
    expected = np.sum(feat_around_yx * weights) / 4
    print("expected result:", expected)
    print(f"features at 0,0\n", features[0, 0:2, 0:2, 0])
    print(f"aligned_feat at 0,0\n", aligned_feat[0, 0, 0, 0])


if __name__ == "__main__":
    test_align()
