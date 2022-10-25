import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

import config as cfg
import RIDet3DAddon.torch.config_dir.util_config as uc3d
import RIDet3DAddon.torch.model.model_util as mu
import RIDet3DAddon.torch.config as cfg3d
import RIDet3DAddon.torch.model.model_util as mu3d

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class FeatureDecoder(nn.Module):
    def __init__(self, anchors_per_scale,
                 channel_compos=uc3d.get_channel_composition(False)):
        super(FeatureDecoder, self).__init__()
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.anchors_per_scale = anchors_per_scale
        self.channel_compos = channel_compos
        self.margin = cfg3d.Architecture.SIGMOID_DELTA
        self.device = cfg.Train.DEVICE
        self.num_scale = len(cfg3d.ModelOutput.FEATURE_SCALES)

    def forward(self, features):
        decoded = {key: [] for key in features.keys()}
        for scale_index in range(self.num_scale):
            anchors_ratio = torch.tensor(self.anchors_per_scale[scale_index])
            box_yx = self.decode_yx(features["yxhw"][scale_index][:, :2, ...])
            box_lw = self.decode_lw(features["yxhw"][scale_index][:, 2:4, ...], anchors_ratio)
            decoded["yxhw"].append(torch.cat([box_yx, box_lw], dim=1))
            decoded["object"].append(torch.sigmoid(features["object"][scale_index]))
            decoded["category"].append(torch.sigmoid(features["category"][scale_index]))
            bbox_pred = [decoded[key][scale_index] for key in self.channel_compos]
            decoded["whole"].append(torch.cat(bbox_pred, dim=1))
            assert decoded["whole"][scale_index].shape == features["whole"][scale_index].shape
        return decoded

    def decode_yx(self, yx_raw):
        """
        :param yx_raw: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_dec = yx coordinates of box centers in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        grid_h, grid_w = yx_raw.shape[-2:]
        # grid_x: (grid_h, grid_w)
        grid_x, grid_y = torch.meshgrid(torch.arange(0, grid_w), torch.arange(0, grid_h))
        # grid_x, grid_y = torch.meshgrid(torch.range(0, grid_w - 1), torch.range(0, grid_h - 1))
        # grid: (grid_h, grid_w, 2)
        grid = torch.stack([grid_y, grid_x], dim=-1).T
        grid = grid.view(1, 2, 1, grid_h, grid_w)
        grid = grid.to(self.device, torch.float32)
        grid_np = grid.cpu().detach().numpy()
        divider = torch.tensor([grid_h, grid_w])
        divider = divider.view(1, 2, 1, 1, 1)
        divider = divider.to(self.device, torch.float32)
        # TODO
        yx_box = mu.sigmoid_with_margin(yx_raw, self.margin)
        # [(batch, grid_h, grid_w, anchor, 2) + (1, grid_h, grid_w, 1, 2)] / (1, 1, 1, 1, 2)
        yx_dec = (yx_box + grid) / divider
        if torch.max(yx_dec) > 10000 or torch.min(yx_dec) < -10000:
            print(torch.max(yx_dec))
            a = 1
        return yx_dec

    def decode_lw(self, lw_raw, anchors_ratio):
        """
        :param lw_raw: (batch, grid_h, grid_w, anchor, 2)
        :param anchors_ratio: [height, width]s of anchors in ratio to image (0~1), (anchor, 2)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """

        num_anc, channel = anchors_ratio.shape  # (3, 2)
        anchors_tf = torch.reshape(anchors_ratio, (1, channel, num_anc, 1, 1)).to(self.device, torch.float32)
        # lw_dec = mu.sigmoid_with_margin(lw_raw, 0, 0, 1) * anchors_tf
        lw_raw = torch.clamp(lw_raw, max=_DEFAULT_SCALE_CLAMP, min=-_DEFAULT_SCALE_CLAMP)

        valid_mask = ~torch.isclose(lw_raw[:, 0:1,...], torch.tensor(0.), 1e-10, 1e-10)
        lw_dec = torch.exp(lw_raw) * anchors_tf
        # print(torch.max(lw_dec))
        if torch.max(lw_dec) > 10000 or torch.min(lw_dec) < -10000:
            print(torch.max(lw_dec))
            a = 1
        return lw_dec

    def inverse(self, feature):
        encoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):
            valid_mask = feature["yxhw"][scale_index][:, :1, ...] > 0
            anchors_ratio = self.anchors_per_scale[scale_index]
            box_yx = self.encode_yx(feature["yxhw"][scale_index][:, :2, ...]) * valid_mask
            box_lw = torch.nan_to_num(self.encode_hw(feature["yxhw"][scale_index][:, 2:4, ...], anchors_ratio) * valid_mask)
            encoded["yxhw"].append(torch.cat([box_yx, box_lw], dim=1))
            encoded["object"].append(mu3d.inv_sigmoid_with_margin(feature["object"][scale_index]) * valid_mask)
            encoded["category"].append(mu3d.inv_sigmoid_with_margin(feature["category"][scale_index])* valid_mask)
            bbox_pred = [encoded[key][scale_index] for key in self.channel_compos]
            encoded["whole"].append(torch.cat(bbox_pred, dim=1))
            assert encoded["yxhw"][scale_index].shape == feature["yxhw"][scale_index].shape
        return encoded

    def encode_yx(self, deocde_yx):
        """
        :param deocde_yx: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_raw = yx logit (batch, grid_h, grid_w, anchor, 2)
        """
        grid_h, grid_w = deocde_yx.shape[-2:]
        # grid_x: (grid_h, grid_w)
        grid_x, grid_y = torch.meshgrid(torch.arange(0, grid_w), torch.arange(0, grid_h))
        # grid_x, grid_y = torch.meshgrid(torch.range(0, grid_w - 1), torch.range(0, grid_h - 1))
        # grid: (grid_h, grid_w, 2)
        grid = torch.stack([grid_y, grid_x], dim=-1).T
        grid = grid.view(1, 2, 1, grid_h, grid_w)
        grid = grid.to(self.device, torch.float32)
        divider = torch.tensor([grid_h, grid_w])
        divider = divider.view(1, 2, 1, 1, 1)
        divider = divider.to(self.device, torch.float32)
        # yx_dec = (yx_box + grid) / divider
        yx_box = deocde_yx * divider - grid
        # yx_box[yx_box < 0] = 0
        yx_raw = mu.inv_sigmoid_with_margin(yx_box, self.margin)
        assert torch.isfinite(yx_raw).all(), yx_raw
        return yx_raw

    def encode_hw(self, decode_hw, anchors_ratio):
        anchors_ratio = torch.tensor(anchors_ratio, device=cfg.Train.DEVICE, dtype=torch.float32)[None, None, None, :, :]
        anchors_ratio = anchors_ratio.permute(0, 4, 3, 1, 2)
        hw_raw = torch.log(decode_hw / anchors_ratio)
        return hw_raw


