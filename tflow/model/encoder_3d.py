import tensorflow as tf

import RIDet3DAddon.config as cfg3d


class FeatureEncoder:
    def __init__(self):
        self.num_scale = len(cfg3d.ModelOutput.FEATURE_SCALES)
        self.margin = cfg3d.Architecture.SIGMOID_DELTA
        self.img_shape = cfg3d.Datasets.DATASET_CONFIG.INPUT_RESOLUTION

    def inverse(self, feature):
        encoded = {"hwl": []}
        for scale_index in range(self.num_scale):
            valid_mask = tf.cast(feature["yx"][scale_index][..., :1] > 0, dtype=tf.float32)
            box_hwl_logit = self.encode_hwl(feature["hwl"][scale_index])
            encoded["hwl"].append(tf.math.multiply_no_nan(box_hwl_logit, valid_mask))
            assert encoded["hwl"][scale_index].shape == feature["hwl"][scale_index].shape
        return encoded

    def encode_yx(self, projected_yx, pred_box_2d):
        """
        :param projected_yx: (batch, grid_h * grid_w, anchor, 2) -> normalize
        :param pred_box_2d: (batch, grid_h * grid_w, anchor, 2) -> normalize
        :return: yx_raw = yx logit (batch, grid_h, grid_w, anchor, 2)
        """
        # TODO yx_box clip check
        yx_box = (projected_yx - pred_box_2d[..., :2]) / (pred_box_2d[..., 2:4] / 2)
        yx_box = tf.clip_by_value(yx_box, -0.9, 0.9)
        yx_raw = 0.5 * tf.math.atanh(yx_box)
        return yx_raw

    def encode_hwl(self, hwl_dec):
        """
        :param hwl_dec: (batch, grid_h*grid_w*anchor, 3)
        :return: hw_raw = heights and widths, length logit (batch, grid_h*grid_w*anchor, 3)
        """
        # num_anc, channel = anchors_ratio.shape  # (3, 2)
        # anchors_tf = tf.reshape(anchors_ratio, (1, HWA, channel))
        anchors_tf = tf.reshape([1.], (1, 1, 1))
        hwl_raw = tf.math.log(hwl_dec / anchors_tf)
        return hwl_raw

    def project_to_image(self, box_yx, depth, intrinsic):
        zero_depth_mask = tf.cast(depth > 0, tf.float32)
        box_y = (box_yx[..., :1] / (depth + 1e-12) * intrinsic[..., 1:2, 1:2]) + \
                intrinsic[..., 1:2, 2:3]
        box_x = (box_yx[..., 1:2] / (depth + 1e-12) * intrinsic[..., :1, :1]) + \
                intrinsic[..., :1, 2:3]
        box_raw = tf.concat([box_y, box_x], axis=-1) * zero_depth_mask
        return box_raw


# ======================================
import numpy as np
from RIDet3DAddon.tflow.model.decoder_3d import FeatureDecoder
import utils.tflow.util_function as uf


def inverse_test():
    encoder = FeatureEncoder()
    decoder = FeatureDecoder()
    feature = {"merged": [], "yxhwl": [], "z": [],
               "theta": [np.zeros((1, 20, 10, 1, 1), dtype=np.float32)], "category": [np.zeros((1, 20, 10, 1, 1), dtype=np.float32)]}
    # yxhwl = np.arange(0, 10, 0.01).astype(np.float32).reshape((1, 20, 10, 1, 5))
    z = np.arange(1.1, 51.1, 0.25).astype(np.float32).reshape((1, 20, 10, 1, 1))
    y = np.arange(1, 11, 0.05).astype(np.float32).reshape((1, 20, 10, 1, 1))
    x = np.arange(-10, 10, 0.1).astype(np.float32).reshape((1, 20, 10, 1, 1))
    hwl = np.ones((1, 20, 10, 1, 3)).astype(np.float32) * 5
    yxhwl = np.concatenate([y, x, hwl], axis=-1)
    feature["yxhwl"].append(yxhwl)
    feature["z"].append(z)
    feature["merged"].append(np.concatenate([yxhwl, feature["z"][0], feature["theta"][0], feature["category"][0]], axis=-1))
    pred_box = np.arange(0.5, 0.7, 0.00025).astype(np.float32).reshape((1, 20, 10, 1, 4))


    # pred_box_ = np.ones((1, 200, 2)).astype(np.float32)
    intrinsic = get_intrinsic().reshape((1, 3, 4))
    # test_decode = decoder.inverse_proj_to_3d(pred_box, feature["z"][0], intrinsic=intrinsic)
    pred_box_xy = encoder.project_to_image(yxhwl.reshape((1,200,5)), feature["z"][0].reshape((1,200,1)),  intrinsic)
    pred_hw = np.ones((1,200,2)).astype(np.float32) * 0.2
    pred_box = np.concatenate([pred_box_xy, pred_hw], axis=-1)

    # TODO 2d box 좌표와 3d box 좌표가 projection 했을때 동일한지?
    # (b,hw,c) 5,1,1,1
    check_feature = {key: np.copy(value) for key, value in feature.items()}
    for slice_key in feature.keys():
        if slice_key != "whole":
            for scale_index in range(len(feature[slice_key])):
                feature[slice_key][scale_index] = uf.merge_dim_hwa(
                    feature[slice_key][scale_index])

    pred_box = pred_box.reshape(1, 200, 4)
    pred_box_ = pred_box
    encode_feature = encoder.inverse(feature, intrinsic, pred_box)
    encode_feature["z"] = [np.log(feature["z"][0] / 10)]
    encode_feature["theta"] = [feature["theta"]]
    encode_feature["category"] = [feature["category"]]
    encode_feature["merged"] = [feature["merged"]]
    for slice_key in encode_feature.keys():
        for scale_index in range(len(encode_feature[slice_key])):
            batch, grid_h, grid_w, anchor, featdim = check_feature[slice_key][scale_index].shape
            encode_feature[slice_key][scale_index] = \
                tf.reshape(encode_feature[slice_key][scale_index], (batch, grid_h, grid_w, anchor, featdim))
    pred_box = pred_box.reshape(batch, grid_h, grid_w, anchor, 4)
    decode_feature = decoder.decode(encode_feature, intrinsic, pred_box)
    decode_box = decode_feature["yxhwl"][0].numpy().reshape((1, 200, 5))
    original_box = feature["yxhwl"][0].numpy()
    decode_feature = {"merged": [], "yxhwl": [], "z": [], "theta": [], "category": []}


def get_intrinsic():
    lidar_file = "/home/eagle/mun_workspace/kitti/training/calib/000000.txt"
    calib_dict = dict()
    with open(lidar_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            new_line = []
            line = line.split(" ")
            if len(line) == 1:
                pass
            else:
                line[0] = line[0].rstrip(":")
                line[-1] = line[-1].rstrip("\n")
                for a in line[1:]:
                    new_line.append(float(a))
                calib_dict[line[0]] = new_line
        calib_dict["P2"] = np.reshape(np.array(calib_dict["P2"]), (3, 4))
        return calib_dict["P2"].astype(np.float32)




if __name__ is "__main__":
    inverse_test()
