import torch
from torch import nn
from torchvision.ops import roi_align

from utils.util_class import MyExceptionToCatch
import utils.framework.util_function as uf
import model.framework.model_util as mu
import RIDet3DAddon.torch.config as cfg3d


def head_factory(output_name, conv_args, in_channels, num_anchors_per_scale, pred_composition):
    if output_name == "Double":
        return DoubleOutput3d(conv_args, in_channels, num_anchors_per_scale, pred_composition)
    elif output_name == "Single":
        return SingleOutput3d(conv_args, in_channels, num_anchors_per_scale, pred_composition)
    else:
        raise MyExceptionToCatch(f"[head_factory[ invalid output name : {output_name}")


class HeadBase(nn.Module):
    def __init__(self, conv_kwargs, in_channels, num_anchors_per_scale, pred_composition):
        super(HeadBase, self).__init__()
        self.num_anchors_per_scale = num_anchors_per_scale
        self.pred_composition = pred_composition
        self.out_channels = sum(pred_composition.values())
        self.device = cfg3d.Train.DEVICE
        self.aling_conv = {"feature_3": mu.CustomConv2D(in_channels=128,
                                                        out_channels=256,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=0,
                                                        **conv_kwargs),
                           "feature_4": mu.CustomConv2D(in_channels=256,
                                                        out_channels=512,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=0,
                                                        **conv_kwargs),
                           "feature_5": mu.CustomConv2D(in_channels=512,
                                                        out_channels=1024,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=0,
                                                        **conv_kwargs)
                           }

    def feature_align(self, features, decode):
        aligned_features = {}
        for scale, (key, feature) in enumerate(features.items()):
            instance_bbox = decode[scale]
            b, c, a, h, w = instance_bbox.shape
            instance_bbox = uf.convert_box_format_yxhw_to_tlbr(instance_bbox)
            bbox = [torch.reshape(instance_bbox[i], (c, -1)).permute(1, 0) for i in range(b)]
            align_feature = roi_align(feature, bbox, 3)
            conv_feat = self.aling_conv[key](align_feature)
            aligned_features[key] = torch.reshape(conv_feat, (b, h, w, -1)).permute(0, 3, 1, 2)

        return aligned_features


class SingleOutput3d(HeadBase):
    def __init__(self, conv_kwargs, in_channels, num_anchors_per_scale, pred_composition):
        super(SingleOutput3d, self).__init__(conv_kwargs, in_channels, num_anchors_per_scale, pred_composition)
        self.feature_3_output = self.make_output(in_channels=in_channels["feature_3"],
                                                 out_channels=256,
                                                 conv_kwargs=conv_kwargs)
        self.feature_4_output = self.make_output(in_channels=in_channels["feature_4"],
                                                 out_channels=512,
                                                 conv_kwargs=conv_kwargs)
        self.feature_5_output = self.make_output(in_channels=in_channels["feature_5"],
                                                 out_channels=1024,
                                                 conv_kwargs=conv_kwargs)
        self.bbox_out = {"feature_3": self.feature_3_output,
                         "feature_4": self.feature_4_output,
                         "feature_5":  self.feature_5_output
                         }
        self.out_channels = {"feature_3": 256, "feature_4": 512, "feature_5": 1024}

    def forward(self, input_features, decode):
        align_features = self.feature_align(input_features, decode)
        output_features = list()
        for key, feat in align_features.items():
            bbox_out = self.bbox_out[key](feat)
            batch, channel, height, width = bbox_out.shape
            bbox_out = torch.reshape(bbox_out, (batch, -1, self.num_anchors_per_scale, height, width))
            output_features.append(bbox_out)

        return output_features

    def make_output(self, in_channels, out_channels, conv_kwargs):
        output = nn.Sequential(
            *[

                mu.CustomConv2D(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                **conv_kwargs),

                mu.CustomConv2D(in_channels=out_channels,
                                out_channels=self.num_anchors_per_scale * self.out_channels,
                                kernel_size=1,
                                stride=1,
                                activation=False,
                                bn=False)
            ]
        )

        return output


class DoubleOutput3d(HeadBase):
    def __init__(self, conv_kwargs, in_channels, num_anchors_per_scale, pred_composition):
        super(DoubleOutput3d, self).__init__(conv_kwargs, in_channels, num_anchors_per_scale, pred_composition)

        self.sbbox_conv = mu.CustomConv2D(in_channels=in_channels["feature_3"],
                                          out_channels=256,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          **conv_kwargs)

        self.mbbox_conv = mu.CustomConv2D(in_channels=in_channels["feature_4"],
                                          out_channels=256,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          **conv_kwargs)

        self.lbbox_conv = mu.CustomConv2D(in_channels=in_channels["feature_5"],
                                          out_channels=256,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          **conv_kwargs)

        self.comm_conv = mu.CustomConv2D(in_channels=256,
                                         out_channels=256,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         **conv_kwargs)

        self.cls_outconv = mu.CustomConv2D(in_channels=256,
                                           out_channels=self.pred_composition["cls"] * self.num_anchors_per_scale,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           **conv_kwargs)

        self.reg_outconv = mu.CustomConv2D(in_channels=256,
                                           out_channels=self.pred_composition["reg"] * self.num_anchors_per_scale,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           **conv_kwargs)

    def forward(self, input_features):
        sbbox_feature = input_features["feature_3"]
        mbbox_feature = input_features["feature_4"]
        lbbox_feature = input_features["feature_5"]

        common_sbbox = self.sbbox_conv(sbbox_feature)
        common_mbbox = self.mbbox_conv(mbbox_feature)
        common_lbbox = self.lbbox_conv(lbbox_feature)
        out_features = {"feature_3": common_sbbox, "feature_4": common_mbbox, "feature_5": common_lbbox}
        for key, feat in out_features.items():
            cls_out = self.comm_conv(feat)
            cls_out = self.comm_conv(cls_out)
            cls_out = self.cls_outconv(cls_out)

            res_out = self.comm_conv(feat)
            res_out = self.comm_conv(res_out)

            res_out = self.reg_outconv(res_out)
            features = torch.concat([cls_out, res_out], dim=1)
            b, c, h, w = features.shape
            out_features[key] = torch.reshape(features, (b, -1, self.num_anchors_per_scale, h, w,))

        return out_features

# ============================

def test_align():
    import numpy as np

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
                                cfg3d.ModelOutput.PRED_3D_HEAD_COMPOSITION)
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
