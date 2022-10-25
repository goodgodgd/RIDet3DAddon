import time

import torch
from torch import nn

import config as cfg
import utils.util_class as uc
import RIDet3DAddon.torch.utils.util_function as uf3d
import utils.framework.util_function as uf

def select_model(model_name):
    if model_name == "detecter3D":
        return Detecter3D
    else:
        raise uc.MyExceptionToCatch(f"[model] EMPTY")


class ModelBase(nn.Module):
    def __init__(self, backbone, neck, head, decoder):
        super(ModelBase, self).__init__()
        self.device = torch.device(cfg.Train.DEVICE)
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.decoder = decoder
        self.to(cfg.Train.DEVICE)

    def forward(self, batched_input):
        # ...
        # return {"backbone_l": backbone_l, "backbone_m": backbone_m, "backbone_s": backbone_s,
        #         "boxreg": boxreg, "category": catetory, "validbox": valid_box}
        raise NotImplementedError


class Detecter3D(ModelBase):
    def __init__(self, backbone, neck, head, decoder, head3d, decoder3d):
        super(Detecter3D, self).__init__(backbone, neck, head, decoder)
        self.head3d = head3d
        self.decoder3d = decoder3d

    def forward(self, batched_input):
        backbone_features = self.backbone(batched_input)
        neck_features = self.neck(backbone_features)

        head_features = self.head(neck_features)
        output_features = dict()
        output_features["feat2d_logit"] = uf3d.merge_and_slice_features(head_features, False, "feat2d")
        output_features["feat2d"] = self.decoder(output_features["feat2d_logit"])
        head3d_features = self.head3d(neck_features, output_features["feat2d"]["yxhw"])
        output_features["feat3d_logit"] = uf3d.merge_and_slice_features(head3d_features, False, "feat3d")
        output_features["feat3d"] = self.decoder3d(output_features["feat3d_logit"], output_features["feat2d"]["yxhw"])

        for key in output_features.keys():
            for slice_key in output_features[key].keys():
                if slice_key != "whole":
                    for scale_index in range(len(output_features[key][slice_key])):
                        output_features[key][slice_key][scale_index] = uf.merge_dim_hwa(
                            output_features[key][slice_key][scale_index])
        return output_features
