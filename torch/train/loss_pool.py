import torch
import torch.nn.functional as F
import numpy as np
import math

import config as cfg
import utils.framework.util_function as uf


class LossBase:
    def __init__(self):
        self.device = cfg.Train.DEVICE

    def __call__(self, features, pred, auxi, scale):
        raise NotImplementedError()


class L1smooth(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        gird_hw = grtr["image"].shape[1:3]
        gird_hw = torch.tensor(np.array([[gird_hw[0], gird_hw[1], gird_hw[0], gird_hw[1]]], dtype=np.float32), dtype=torch.float32, device=self.device)
        object_mask = grtr["feat2d"]["object"][scale] == 1
        smoothl1 = torch.nn.SmoothL1Loss(reduction='none', beta=0.01)
        # huber_loss = torch.nn.HuberLoss(reduction='none', delta=1)
        # pred_yxhw_pixel = pred["feat2d"]["yxhw"][scale].cpu().detach().numpy() #* gird_hw
        # grtr_yxhw_pixel = grtr["feat2d"]["yxhw"][scale].cpu().detach().numpy() #* gird_hw
        l1_loss = smoothl1( pred["feat2d_logit"]["yxhw"][scale], grtr["feat2d_logit"]["yxhw"][scale] )
        l1_loss = uf.reduce_sum(l1_loss * object_mask, dim=-1)
        # l1_lossnp = l1_loss.cpu().detach().numpy()
        scalar_loss = uf.reduce_sum(l1_loss)
        return scalar_loss, l1_loss


class Box3DLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        object_mask = grtr["feat2d"]["object"][scale] == 1
        pred["feat3d"] = self.box_preprocess(grtr["feat3d"], pred["feat3d"], scale)
        huber_loss = torch.nn.SmoothL1Loss(reduction='none', beta=0.01)
        # huber_loss = torch.nn.HuberLoss(reduction='none', delta=1)
        box_xyz_loss = huber_loss(pred["feat3d"]["lxyz"][scale], grtr["feat3d"]["lxyz"][scale]) #* object_mask[..., 0]
        box_hwl_loss = huber_loss(pred["feat3d_logit"]["hwl"][scale], grtr["feat3d_logit"]["hwl"][scale]) #* object_mask[..., 0]

        box_xyz_loss = box_xyz_loss * object_mask
        box_hwl_loss = box_hwl_loss * object_mask
        box_3d_loss = uf.reduce_sum(torch.cat([box_xyz_loss, box_hwl_loss], dim=-1), dim=-1)
        scalar_loss = uf.reduce_sum(box_xyz_loss) + uf.reduce_sum(box_hwl_loss)
        return scalar_loss, box_3d_loss

    def box_preprocess(self, grtr, pred, scale):
        # TODO gt logit
        l = pred["hwl"][scale][..., 1:2]
        w = pred["hwl"][scale][..., 0:1]
        wl = pred["hwl"][scale][..., 0:2]
        lw = torch.cat([l, w], dim=-1)
        h = pred["hwl"][scale][..., 2:]
        M = uf.cast(torch.abs(pred["theta"][scale] - grtr["theta"][scale]) < (np.pi / 4), dtype=torch.float32)
        wl = M * lw + (1 - M) * wl
        pred["hwl"][scale] = torch.cat([wl, h], dim=-1)
        return pred


class BoxObjectnessLoss(LossBase):
    def __init__(self, pos_weight, neg_weight):
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: objectness loss (batch, HWA)
        """
        grtr_obj = grtr["feat2d"]["object"][scale].squeeze()
        pred_obj = pred["feat2d"]["object"][scale].squeeze()
        ignore_mask = auxi["ignore_mask"]
        conf_focal = torch.pow(grtr_obj - pred_obj, 2)
        obj_loss = F.binary_cross_entropy(pred_obj, grtr_obj, reduction="none")
        obj_loss = obj_loss * conf_focal
        obj_positive = obj_loss * grtr_obj * self.pos_weight
        obj_negative = obj_loss * (1 - grtr_obj) * ignore_mask * self.neg_weight
        scalar_loss = uf.reduce_sum(obj_positive) + uf.reduce_sum(obj_negative)
        return scalar_loss, obj_loss


class CategoryLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        pass

    def compute_category_loss(self, grtr, pred, mask):
        """
        :param grtr: (batch, HWA, 1)
        :param pred: (batch, HWA, K)
        :param mask: (batch, HWA, K or 1)
        :return: (batch, HWA)
        """
        num_cate = pred.shape[-1]
        # pred = pred.type(torch.FloatTensor)
        grtr_label = uf.cast(grtr, dtype=torch.int64).squeeze()
        grtr_onehot = F.one_hot(grtr_label, num_cate).to(torch.float32)
        # category_loss: (batch, HWA, K)
        category_loss = F.binary_cross_entropy(pred, grtr_onehot, reduction="none")
        loss_map = uf.reduce_sum(category_loss * mask, dim=-1)
        loss = uf.reduce_sum(loss_map)
        return loss, loss_map


class MajorCategoryLoss(CategoryLoss):
    def __init__(self, feat_name):
        self.feat_name = f"feat{feat_name}"

    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: category loss (batch, HWA, K)
        """
        object_mask, valid_category = grtr["feat2d"]["object"][scale] == 1, auxi["valid_category"]
        scalar_loss, category_loss = self.compute_category_loss(grtr[self.feat_name]["category"][scale],
                                                                pred[self.feat_name]["category"][scale],
                                                                object_mask * valid_category)
        return scalar_loss, category_loss


class ThetaLoss(LossBase):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, grtr, pred, auxi, scale):
        object_mask = grtr["feat2d"]["object"][scale] == 1
        c_theta = torch.cos(4*(grtr["feat3d"]["theta"][scale] - pred["feat3d"]["theta"][scale]))
        theta_loss = ((1 - 0.5/self.beta * torch.log((self.alpha+c_theta)/(self.alpha-c_theta))) * object_mask)[..., 0]
        scalar_loss = uf.reduce_sum(theta_loss)
        return scalar_loss, theta_loss
