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
            decoded["xyz"].append(self.decode_xyz(feature["xyz"][scale_index], box_2d[scale_index]))
            decoded["lwh"].append(self.decode_lwh(feature["lwh"][scale_index]))
            decoded["theta"].append(
                mu3d.sigmoid_with_margin(feature["theta"][scale_index], 0, 0, 1) * (torch.pi / 2))
            decoded["category"].append(torch.sigmoid(feature["category"][scale_index]))
            bbox_pred = [decoded[key][scale_index] for key in self.channel_compos]
            decoded["whole"].append(torch.cat(bbox_pred, dim=1))
            assert decoded["whole"][scale_index].shape == feature["whole"][scale_index].shape
        return decoded

    def decode_xyz(self, xyz_raw, box_2d):

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
        return xyz_dec

    def decode_lwh(self, lwh_raw):
        """
        :param lw_raw: (batch, grid_h, grid_w, anchor, 3)
         :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 3)
        """
        lwh_raw = torch.clamp(lwh_raw, max=_DEFAULT_SCALE_CLAMP, min=-_DEFAULT_SCALE_CLAMP)
        return torch.exp(lwh_raw) * self.anchors_tf

    def inverse(self, feature, box_2d):
        encoded = {key: [] for key in feature.keys()}
        for scale_index in range(self.num_scale):
            # box_2d_np = box_2d[scale_index].detach().cpu().numpy()
            # xyz_np = feature["xyz"][scale_index].detach().cpu().numpy()
            valid_mask = feature["xyz"][scale_index][:, :1, ...] > 0
            # encoded["xyz"].append(self.encode_xyz(feature["xyz"][scale_index], box_2d[scale_index]) * valid_mask)
            encoded["lwh"].append(self.encode_lwh(feature["lwh"][scale_index]) * valid_mask)
            # encoded["theta"].append(mu3d.inv_sigmoid_with_margin(feature["theta"][scale_index], self.margin) * (
            #             2 / torch.pi)* valid_mask)
            # encoded["category"].append(mu3d.inv_sigmoid_with_margin(feature["category"][scale_index])* valid_mask)
            # bbox_pred = [encoded[key][scale_index] for key in self.channel_compos]
            # encoded["whole"].append(torch.cat(bbox_pred, dim=1))
            # assert encoded["xyz"][scale_index].shape == feature["xyz"][scale_index].shape
            # assert encoded["lwh"][scale_index].shape == feature["lwh"][scale_index].shape
            # assert encoded["theta"][scale_index].shape == feature["theta"][scale_index].shape
        return encoded

    def encode_xyz(self, xyz_dec, box_2d):
        """
        :param yx_dec: (batch, grid_h, grid_w, anchor, 2)
        :return: yx_raw = yx logit (batch, grid_h, grid_w, anchor, 2)
        """

        box_2d_np = box_2d.detach().cpu().numpy()
        batch, channel, anchor, h, w = xyz_dec.shape
        xyz_dec = xyz_dec.reshape(batch, channel, -1)
        xyz_dec_np = xyz_dec.detach().cpu().numpy()
        # box_2d = box_2d.reshape(batch, channel + 1, -1)
        tilt_xyz = torch.matmul(self.rotation_matric.permute(0, 2, 1), xyz_dec).reshape(xyz_dec.shape)
        tilt_xy = tilt_xyz[:, :2, ...].reshape(batch, 2, anchor, h, w)
        tilt_z = tilt_xyz[:, 2:3, ...].reshape(batch, 1, anchor, h, w)
        tilt_xyz_np = tilt_xyz.detach().cpu().numpy()
        z_raw = (tilt_z + tilt_xy[:, :1, ...] * torch.tan(self.tilt)) * torch.cos(self.tilt)
        z_raw_np = z_raw.detach().cpu().numpy()
        z_raw = mu3d.inv_sigmoid_with_margin(z_raw, 0, -1, 3)
        z_raw2_np = z_raw.detach().cpu().numpy()
        yx_dec = - tilt_xy/(self.cell_size * self.image_shape)
        yx_dec_np = yx_dec.detach().cpu().numpy()
        yx_img = yx_dec + self.new_tensor
        yx_img_np = yx_img.detach().cpu().numpy()
        yx_img = torch.nan_to_num((yx_img - box_2d[:, :2, ...]) /  box_2d[:, 2:4, ...],nan=0, posinf=0,neginf=0)
        yx_img2_np = yx_img.detach().cpu().numpy()
        yx_raw = torch.arctanh(yx_img)
        yx_raw_np = yx_raw.detach().cpu().numpy()
        y_raw, x_raw = torch.split(yx_raw, [1, 1], dim=1)
        y_raw_np = y_raw.detach().cpu().numpy()
        x_raw_np = x_raw.detach().cpu().numpy()
        xyz_raw = torch.cat([x_raw, y_raw, z_raw], dim=1)
        xyz_raw_np = xyz_raw.detach().cpu().numpy()
        return torch.nan_to_num(xyz_raw)

    def encode_lwh(self, lwh_dec):
        return torch.nan_to_num(torch.log(lwh_dec * self.anchors_tf))