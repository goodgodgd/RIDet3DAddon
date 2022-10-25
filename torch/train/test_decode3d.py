import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2

import settings
import utils.framework.util_function as uf
import settings
import config_dir.config_generator as cg
import RIDet3DAddon.torch.config_dir.config_generator as cg3d
import RIDet3DAddon.torch.config as cfg3d
import RIDet3DAddon.torch.dataloader.data_util as du3d

import RIDet3DAddon.torch.config_dir.util_config as uc3d
import config as cfg
import RIDet3DAddon.torch.model.model_util as mu
from RIDet3DAddon.torch.train.feature_generator import FeatureMapDistributer
from RIDet3DAddon.torch.model.decoder_2d import FeatureDecoder
from RIDet3DAddon.torch.model.decoder_3d import FeatureDecoder3D
from RIDet3DAddon.torch.log.visual_log import VisualLog2d


if __name__ == '__main__':
    anchors = cfg.AnchorGeneration.ANCHORS
    image_shape = cfg3d.Datasets.Kittibev.INPUT_RESOLUTION
    anchors_per_scale = np.array([anchor / np.array([image_shape[:2]]) for anchor in anchors], dtype=np.float32)
    decoder = FeatureDecoder(anchors_per_scale)
    decoder3d = FeatureDecoder3D(anchors_per_scale)
    feature_creator = FeatureMapDistributer(cfg3d.FeatureDistribPolicy.POLICY_NAME, image_shape, anchors_per_scale)
    image = np.zeros((1, 384, 768, 3), dtype=np.float32)
    inst2d = np.zeros((1, 50, 6))
    inst3d = np.zeros((1, 50, 8))
    dontcare = np.zeros((1, 50, 6))


    # 566.png
    # 2653.png

    inst2d[0, 0, :] = np.array([0.3620, 0.4987, 0.2188, 0.0846, 1.0000, 2.0000])
    inst2d[0, 1, :] = np.array([0.3411, 0.3438, 0.1276, 0.0547, 1.0000, 2.0000])
    inst2d[0, 2, :] = np.array([0.4271, 0.6549, 0.2109, 0.1628, 1.0000, 2.0000])
    inst3d[0, 0:3, :] = [[2.7100e+01, -3.0000e-02, -9.7000e-01, 5.4000e+00, 2.0300e+00,
                          2.7200e+00, -2.0000e+00, 2.0000e+00],
                         [2.8650e+01, 7.4700e+00, -1.4600e+00, 3.0500e+00, 1.8100e+00,
                          1.5600e+00, -1.8800e+00, 2.0000e+00],
                         [2.3980e+01, -7.5400e+00, -9.4000e-01, 7.4100e+00, 2.4400e+00,
                          3.2500e+00, -3.0000e-01, 2.0000e+00]]
    # inst3d =

    features = {"image": torch.tensor(image), "inst2d": torch.tensor(inst2d), "inst3d": torch.tensor(inst3d),
                "dontcare": torch.tensor(dontcare)}

    features = feature_creator(features)
    # del features["feat2d_logit"]["mp_object"]
    feat2d_test1 = features["feat2d"]["yxhw"][0].cpu().detach().numpy()
    feat2d_logit_test1 = features["feat2d_logit"]["yxhw"][0].cpu().detach().numpy()
    feat2d_test2 = features["feat2d"]["yxhw"][1].cpu().detach().numpy()
    feat2d_logit_test2 = features["feat2d_logit"]["yxhw"][1].cpu().detach().numpy()
    feat2d_test3 = features["feat2d"]["yxhw"][2].cpu().detach().numpy()
    feat2d_logit_test3 = features["feat2d_logit"]["yxhw"][2].cpu().detach().numpy()
    feat3d_test1 = features["feat3d"]["xyz"][0].cpu().detach().numpy()
    # feat3d_logit_test1 = features["feat3d_logit"]["xyz"][0].cpu().detach().numpy()
    feat3d_test2 = features["feat3d"]["xyz"][1].cpu().detach().numpy()
    # feat3d_logit_test2 = features["feat3d_logit"]["xyz"][1].cpu().detach().numpy()
    feat3d_test3 = features["feat3d"]["xyz"][2].cpu().detach().numpy()
    # feat3d_logit_test3 = features["feat3d_logit"]["xyz"][2].cpu().detach().numpy()

    feat3d_lwh_test1 = features["feat3d"]["lwh"][0].cpu().detach().numpy()
    feat3d_lwh_logit_test1 = features["feat3d_logit"]["lwh"][0].cpu().detach().numpy()
    feat3d_lwh_test2 = features["feat3d"]["lwh"][1].cpu().detach().numpy()
    feat3d_lwh_logit_test2 = features["feat3d_logit"]["lwh"][1].cpu().detach().numpy()
    feat3d_lwh_test3 = features["feat3d"]["lwh"][2].cpu().detach().numpy()
    feat3d_lwh_logit_test3 = features["feat3d_logit"]["lwh"][2].cpu().detach().numpy()

    valid_mask1 = feat3d_test1[:, 0, ...] > 0
    valid_mask2 = feat3d_test2[:, 0, ...] > 0
    valid_mask3 = feat3d_test3[:, 0, ...] > 0

    out_feat3d_y = decoder3d.forward(features["feat3d_logit"], features['feat2d']["yxhw"])
    out_feat3d_np = uf.convert_tensor_to_numpy(out_feat3d_y)
    # out_feat3d_test1 = out_feat3d_np["xyz"][0] * valid_mask1
    # out_feat3d_test2 = out_feat3d_np["xyz"][1] * valid_mask2
    # out_feat3d_test3 = out_feat3d_np["xyz"][2] * valid_mask3

    out_feat3d_lwh_test1 = out_feat3d_np["lwh"][0] * valid_mask1
    out_feat3d_lwh_test2 = out_feat3d_np["lwh"][1] * valid_mask2
    out_feat3d_lwh_test3 = out_feat3d_np["lwh"][2] * valid_mask3


    valid_mask1_ = out_feat3d_lwh_test1 > 0
    valid_mask2_ = out_feat3d_lwh_test2 > 0
    valid_mask3_ = out_feat3d_lwh_test3 > 0
    valid_test1 = out_feat3d_lwh_test1 * valid_mask1_
    valid_test2 = out_feat3d_lwh_test2 * valid_mask2_
    valid_test3 = out_feat3d_lwh_test3 * valid_mask3_
    for i in range(features["image"].shape[0]):
        inst_image = features["image"][i].copy()
        inst_image = VisualLog2d("", 0).draw_boxes(inst_image, features["inst2d"], i, (255, 0, 0))

        # image = self.visual_log2d.draw_lanes(image, features["inst_lane"], i, (255, 0, 255))
        feat = []
        for scale in range(3):
            feature = features["feat2d"]["whole"][scale][i]
            test = feature[4, ...] > 0
            feature = feature[:, feature[4, ...] > 0]
            feat.append(feature)
        feat_boxes = np.concatenate(feat, axis=-1)
        feat_boxes = uf.convert_box_format_yxhw_to_tlbr(feat_boxes.T)
        if len(feat_boxes) == 0:
            feat_boxes = np.array([[0, 0, 0, 0]], dtype=np.float32)
        feat_image = features["image"][i].copy()
        feat_image = du3d.draw_box(feat_image, feat_boxes)

        out_feat = []
        for scale in range(3):
            out_feature = out_feat2d_np["whole"][scale][i]
            test = out_feature[4, ...] > 0
            out_feature = out_feature[:, out_feature[4, ...] > 0]
            out_feat.append(out_feature)
        out_feat_boxes = np.concatenate(out_feat, axis=-1)
        out_feat_boxes = uf.convert_box_format_yxhw_to_tlbr(out_feat_boxes.T)
        if len(out_feat_boxes) == 0:
            out_feat_boxes = np.array([[0, 0, 0, 0]], dtype=np.float32)
        out_feat_image = features["image"][i].copy()
        out_feat_image = du3d.draw_box(out_feat_image, out_feat_boxes)

        total_image = np.concatenate([feat_image, inst_image, out_feat_image], axis=0)

        cv2.imshow(F"test{i}", total_image)
        cv2.waitKey(0)

    print()
