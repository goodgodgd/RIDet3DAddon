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
        huber_loss = torch.nn.HuberLoss(reduction='none', delta=0.01)
        pred_yxhw_pixel = pred.cpu().detach().numpy() #* gird_hw
        grtr_yxhw_pixel = grtr.cpu().detach().numpy() #* gird_hw
        l1_loss = huber_loss( pred, grtr)
        l1_lossnp = l1_loss.cpu().detach().numpy()
        scalar_loss = uf.reduce_sum(l1_loss)
        return scalar_loss, l1_loss



def test_main():
    yx_raw = torch.zeros((48,50))
    grid_h, grid_w = yx_raw.shape
    # grid_x: (grid_h, grid_w)
    grid_x, grid_y = torch.meshgrid(torch.arange(0, grid_w), torch.arange(0, grid_h))
    # grid_x, grid_y = torch.meshgrid(torch.range(0, grid_w - 1), torch.range(0, grid_h - 1))
    # grid: (grid_h, grid_w, 2)
    grid = torch.stack([grid_y, grid_x], dim=-1).T
    grid = grid.view(1, 2, 1, grid_h, grid_w)
    grid = grid.to("cuda", torch.float32)
    divider = torch.tensor([grid_h, grid_w])
    divider = divider.view(1, 2, 1, 1, 1)
    divider = divider.to("cuda", torch.float32)
    pred = grid /divider
    grtr = torch.zeros_like(pred)



    com = {"yxhw": 2}
    huber_loss_1 = torch.nn.HuberLoss(reduction='none', delta=1.0)
    huber_loss_01 = torch.nn.HuberLoss(reduction='none', delta=0.01)
    smoothl1_1 = torch.nn.SmoothL1Loss(reduction='none', beta=1.0)
    smoothl1_01 = torch.nn.SmoothL1Loss(reduction='none', beta=0.01)

    # grtr = uf.slice_feature(grtr, com, 1)
    # pred = uf.slice_feature(pred, com, 1)
    pred = uf.merge_dim_hwa(pred)
    grtr = uf.merge_dim_hwa(grtr)
    pred_yxhw_pixel = pred.cpu().detach().numpy()  # * gird_hw
    grtr_yxhw_pixel = grtr.cpu().detach().numpy()  # * gird_hw


    l1_loss_1 = huber_loss_1(pred, grtr)
    l1_loss_01 = huber_loss_01(pred, grtr)
    smoothl1_loss_1 = smoothl1_1(pred, grtr)
    smoothl1_loss_01 = smoothl1_01(pred, grtr)
    l1_lossnp_1 = l1_loss_1.cpu().detach().numpy()
    l1_lossnp_01 = l1_loss_01.cpu().detach().numpy()
    smoothl1_lossnp_1 = smoothl1_loss_1.cpu().detach().numpy()
    smoothl1_lossnp_01 = smoothl1_loss_01.cpu().detach().numpy()
    scalar_loss_1 = uf.reduce_sum(l1_loss_1)
    scalar_loss_01 = uf.reduce_sum(l1_loss_01)
    scalar_sloss_1 = uf.reduce_sum(smoothl1_loss_1)
    scalar_sloss_01 = uf.reduce_sum(smoothl1_loss_01)
    print(scalar_loss_1)
    print(scalar_loss_01)


if __name__ == '__main__':
    test_main()