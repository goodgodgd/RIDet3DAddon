import torch
from torch import nn

from utils.util_class import MyExceptionToCatch
import RIDet3DAddon.torch.model.model_util as mu
import RIDet3DAddon.torch.model.weight_init as wi


def head_factory(output_name, conv_args, in_channels, num_anchors_per_scale, pred_composition):
    if output_name == "Double":
        return DoubleOutput(conv_args, in_channels, num_anchors_per_scale, pred_composition)
    elif output_name == "Single":
        return SingleOutput(conv_args, in_channels, num_anchors_per_scale, pred_composition)
    else:
        raise MyExceptionToCatch(f"[head_factory[ invalid output name : {output_name}")


class HeadBase(nn.Module):
    def __init__(self, conv_kwargs, in_channels, num_anchors_per_scale, pred_composition):
        super(HeadBase, self).__init__()
        self.num_anchors_per_scale = num_anchors_per_scale
        self.pred_composition = pred_composition
        self.out_channels = sum(pred_composition.values())


class SingleOutput(HeadBase):
    def __init__(self, conv_kwargs, in_channels, num_anchors_per_scale, pred_composition):
        super(SingleOutput, self).__init__(conv_kwargs, in_channels, num_anchors_per_scale, pred_composition)
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

    def forward(self, input_features):
        output_features = list()
        for key, feat in input_features.items():
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
        for out in output:
            wi.c2_xavier_fill(out.conv)
        return output


class DoubleOutput(HeadBase):
    def __init__(self, conv_kwargs, in_channels, num_anchors_per_scale, pred_composition):
        super(DoubleOutput, self).__init__(conv_kwargs, in_channels, num_anchors_per_scale, pred_composition)

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
