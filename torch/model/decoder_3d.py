import time

import torch
from torch import nn
import numpy as np
import math

import RIDet3DAddon.torch.config as cfg3d
import RIDet3DAddon.torch.config_dir.util_config as uc3d
import RIDet3DAddon.torch.model.model_util as mu3d
import RIDet3DAddon.torch.utils.util_function as uf3d

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class FeatureDecoder3D(nn.Module):
    def __init__(self, anchors_per_scale,
                 channel_compos=uc3d.get_3d_channel_composition(False)):
        super(FeatureDecoder3D, self).__init__()
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.anchors_per_scale = anchors_per_scale
        self.channel_compos = channel_compos
        self.device = cfg3d.Train.DEVICE
        self.margin = cfg3d.Architecture.SIGMOID_DELTA
        self.cell_size = cfg3d.Datasets.DATASET_CONFIG.CELL_SIZE
        image_shape = cfg3d.Datasets.DATASET_CONFIG.INPUT_RESOLUTION
        self.image_shape = torch.tensor(image_shape, device=self.device, dtype=torch.float32).reshape(1, 2, 1, 1, 1)
        self.num_scale = len(cfg3d.ModelOutput.FEATURE_SCALES)
        self.tilt = torch.tensor(cfg3d.Datasets.DATASET_CONFIG.TILT_ANGLE).to(device=self.device, dtype=torch.float32)
        self.new_tensor = torch.tensor([1, 1/2], device=self.device, dtype=torch.float32).reshape(1, 2, 1, 1, 1)
        self.rotation_matric = torch.tensor([[
            [torch.cos(self.tilt), 0, -torch.sin(self.tilt)],
            [0, 1, 0],
            [torch.sin(self.tilt), 0, torch.cos(self.tilt)]
        ]], device=self.device, dtype=torch.float32)
        self.anchors_tf = torch.tensor([1.]).reshape(1, 1, 1, 1, 1).to(device=self.device, dtype=torch.float32)

    def forward(self, feature, box_2d):
        """
        :param feature: raw feature map predicted by model (batch, grid_h, grid_w, anchor, channel)
        :param scale_ind: scale name e.g. 0~2
        :return: decoded feature in the same shape e.g. (yxhw, objectness, category probabilities)
        """
        decoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):

            decoded["hwl"].append(self.decode_hwl(feature["hwl"][scale_index]))
            decoded["lxyz"].append(self.decode_xyz(feature["lxyz"][scale_index], box_2d[scale_index], decoded["hwl"][scale_index]))
            decoded["theta"].append(
                mu3d.sigmoid_with_margin(feature["theta"][scale_index], 0, 0, 1) * (torch.pi / 2))
            decoded["category"].append(torch.sigmoid(feature["category"][scale_index]))
            bbox_pred = [decoded[key][scale_index] for key in self.channel_compos]
            decoded["whole"].append(torch.cat(bbox_pred, dim=1))
            assert decoded["whole"][scale_index].shape == feature["whole"][scale_index].shape
        return decoded

    def decode_xyz(self, xyz_raw, box_2d, hwl_dec):

        x_raw, y_raw, z_raw = torch.split(xyz_raw, [1, 1, 1], dim=1)
        yx_raw = torch.cat([y_raw, x_raw], dim=1)
        yx_img = torch.tanh(yx_raw)
        yx_img = box_2d[:, :2, ...] + (yx_img * box_2d[:, 2:4, ...])
        yx_dec = yx_img - self.new_tensor
        tilt_xy = - self.cell_size * self.image_shape * yx_dec
        tilt_z = mu3d.sigmoid_with_margin(z_raw, 0, -1, 3) / torch.cos(self.tilt) \
                 - tilt_xy[:, :1, ...] * torch.tan(self.tilt)
        tilt_xyz = torch.cat([tilt_xy, tilt_z], dim=1).reshape((xyz_raw.shape[0], 3, -1))
        xyz_dec = torch.matmul(self.rotation_matric, tilt_xyz).reshape(xyz_raw.shape)
        # xyz_dec_np = xyz_dec.detach().cpu().numpy()
        # xyz_dec[:, 2,...] = xyz_dec[:, 2,...]# - hwl_dec[:,0,...]/2
        # xyz_dec___np = xyz_dec.detach().cpu().numpy()
        # xyz_dec = xyz_dec -
        return xyz_dec

    def decode_hwl(self, hwl_raw):
        """
        :param lw_raw: (batch, grid_h, grid_w, anchor, 3)
         :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 3)
        """
        hwl_raw = torch.clamp(hwl_raw, max=_DEFAULT_SCALE_CLAMP, min=-_DEFAULT_SCALE_CLAMP)
        return torch.exp(hwl_raw) * self.anchors_tf

    def inverse(self, feature, box_2d):
        encoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):
            # box_2d_np = box_2d[scale_index].detach().cpu().numpy()
            # xyz_np = feature["xyz"][scale_index].detach().cpu().numpy()
            valid_mask = feature["lxyz"][scale_index][:, :1, ...] > 0
            # encoded["xyz"].append(self.encode_xyz(feature["xyz"][scale_index], box_2d[scale_index]) * valid_mask)
            encoded["hwl"].append(self.encode_hwl(feature["hwl"][scale_index]) * valid_mask)
            # encoded["theta"].append(mu3d.inv_sigmoid_with_margin(feature["theta"][scale_index], self.margin) * (
            #             2 / torch.pi)* valid_mask)
            # encoded["category"].append(mu3d.inv_sigmoid_with_margin(feature["category"][scale_index])* valid_mask)
            # bbox_pred = [encoded[key][scale_index] for key in self.channel_compos]
            # encoded["whole"].append(torch.cat(bbox_pred, dim=1))
            # assert encoded["xyz"][scale_index].shape == feature["xyz"][scale_index].shape
            # assert encoded["hwl"][scale_index].shape == feature["hwl"][scale_index].shape
            # assert encoded["theta"][scale_index].shape == feature["theta"][scale_index].shape
        return encoded

    def encode_lxyz(self, lxyz_dec, box_2d):
        """
        :param yx_dec: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_raw = yx logit (batch, grid_h, grid_w, anchor, 2)
        """

        box_2d_np = box_2d.detach().cpu().numpy()
        batch, channel, anchor, h, w = lxyz_dec.shape
        lxyz_dec = lxyz_dec.reshape(batch, channel, -1)
        # box_2d = box_2d.reshape(batch, channel + 1, -1)
        tilt_lxyz = torch.matmul(self.rotation_matric.permute(0, 2, 1), lxyz_dec).reshape(lxyz_dec.shape)
        tilt_lxy = tilt_lxyz[:, :2, ...].reshape(batch, 2, anchor, h, w)
        tilt_lz = tilt_lxyz[:, 2:3, ...].reshape(batch, 1, anchor, h, w)
        lz_raw = (tilt_lz + tilt_lxy[:, :1, ...] * torch.tan(self.tilt)) * torch.cos(self.tilt)

        lz_raw = mu3d.inv_sigmoid_with_margin(lz_raw, 0, -1, 3)
        lyx_dec = - tilt_lxy/(self.cell_size * self.image_shape)
        lyx_img = lyx_dec + self.new_tensor
        lyx_img = torch.nan_to_num((lyx_img - box_2d[:, :2, ...]) /  box_2d[:, 2:4, ...],nan=0, posinf=0,neginf=0)
        lyx_raw = torch.arctanh(lyx_img)
        ly_raw, lx_raw = torch.split(lyx_raw, [1, 1], dim=1)
        lxyz_raw = torch.cat([lx_raw, ly_raw, lz_raw], dim=1)
        return torch.nan_to_num(lxyz_raw)

    def encode_hwl(self, hwl_dec):
        return torch.nan_to_num(torch.log(hwl_dec * self.anchors_tf))