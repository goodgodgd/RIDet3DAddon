import time

import numpy as np
import math
import torch
from torch import nn
import torchvision
from torch.nn import functional as F

import utils.framework.util_function as uf
import model.framework.weight_init as wi
from utils.util_class import MyExceptionToCatch
import RIDet3DAddon.torch.config as cfg3d


class NonMaximumSuppression:
    def __init__(self, max_out=cfg3d.NmsInfer.MAX_OUT,
                 iou_thresh=cfg3d.NmsInfer.IOU_THRESH,
                 score_thresh=cfg3d.NmsInfer.SCORE_THRESH,
                 category_names=cfg3d.Dataloader.CATEGORY_NAMES["category"],

                 ):
        self.max_out = max_out
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.category_names = category_names

    def __call__(self, pred, max_out=None, iou_thresh=None, score_thresh=None, merged=False, is_3d=False):
        self.max_out = max_out if max_out is not None else self.max_out
        self.iou_thresh = iou_thresh if iou_thresh is not None else self.iou_thresh
        self.score_thresh = score_thresh if score_thresh is not None else self.score_thresh

        nms_res = self.pure_nms(pred, merged)
        return nms_res

    # @tf.function
    def pure_nms(self, pred, merged=False):
        """
        :param pred: if merged True, dict of prediction slices merged over scales,
                        {'yxhw': (batch, sum of Nx, 4), 'object': ..., 'category': ...}
                     if merged False, dict of prediction slices for each scale,
                        {'feature_l': {'yxhw': (batch, Nl, 4), 'object': ..., 'category': ...}}
        :param merged
        :return: (batch, max_out, 8), 8: bbox, category, objectness, ctgr_prob, score
        """
        pred_2d = pred["feat2d"]
        pred_3d = pred["feat3d"]
        if merged is False:
            pred_2d = self.append_anchor_inds(pred_2d)
            pred_2d = self.merged_scale(pred_2d)
            pred_3d = self.merged_scale(pred_3d)

        boxes = uf.convert_box_format_yxhw_to_tlbr(pred_2d["yxhw"])  # (batch, N, 4)
        categories = torch.argmax(pred_2d["category"], dim=-1)  # (batch, N)
        best_probs, best_probs_index = torch.max(pred_2d["category"], dim=-1)  # (batch, N)
        objectness = pred_2d["object"][..., 0]  # (batch, N)
        scores = objectness * best_probs  # (batch, N)
        batch, numbox, numctgr = pred_2d["category"].shape
        anchor_inds = pred_2d["anchor_ind"][..., 0]

        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(1, numctgr):
            ctgr_mask = (categories == ctgr_idx).to(dtype=torch.int64)  # (batch, N)
            score_mask = (scores >= self.score_thresh[ctgr_idx]).squeeze(-1)
            nms_mask = ctgr_mask * score_mask
            ctgr_boxes = boxes * nms_mask[..., None]  # (batch, N, 4)
            ctgr_scores = scores * nms_mask  # (batch, N)
            max_num = self.max_out[ctgr_idx]
            for frame_idx in range(batch):
                # NMS
                # nonzero_inds = torch.nonzero(ctgr_scores[frame_idx]).squeeze(-1)
                ctgr_boxes_tlbr = uf.convert_box_format_yxhw_to_tlbr(ctgr_boxes[frame_idx])
                selected_indices = torchvision.ops.nms(ctgr_boxes_tlbr,
                                                       ctgr_scores[frame_idx],
                                                       self.iou_thresh[ctgr_idx])
                selected_indices = selected_indices[:max_num]
                # box_indices = nonzero_inds[selected_indices]
                batch_indices[frame_idx].append(selected_indices)
        # make batch_indices, valid_mask as fixed shape tensor
        batch_indices = [torch.concat(ctgr_indices, dim=-1) for ctgr_indices in batch_indices]
        batch_indices = torch.stack(batch_indices, dim=0)  # (batch, K*max_output)
        batch_indices = torch.maximum(batch_indices, torch.zeros_like(batch_indices))

        # list of (batch, N) -> (batch, N, 4)
        categories = categories.to(dtype=torch.float32)
        # "bbox": 4, "object": 1, "category": 1, "minor_ctgr": 1, "distance": 1, "score": 1, "anchor_inds": 1
        result2d = torch.stack([objectness, categories, best_probs, scores, anchor_inds], dim=-1)
        result2d = torch.concat([pred_2d["yxhw"], result2d], dim=-1)  # (batch, N, 10)

        result2d_valid = result2d[..., -2:-1] > min(self.score_thresh)
        result2d = result2d * result2d_valid
        batch_indices_2d = batch_indices[..., None].expand(batch, sum(self.max_out[1:]), result2d.shape[-1])

        result2d = torch.gather(result2d, 1, batch_indices_2d)  # (batch, K*max_output, 10)

        valid_mask = result2d[..., 7] > 1e-05
        result2d = result2d * valid_mask[..., None]  # (batch, K*max_output, 10)

        result3d = torch.concat([pred_3d["lxyz"], pred_3d["hwl"], pred_3d["theta"], categories[..., None]], dim=-1)
        result3d = result3d * result2d_valid
        # result3d = torch.concat([pred_3d[key] for key in pred_3d.keys() if key is not "whole"], dim=-1)
        batch_indices_3d = batch_indices[..., None].expand(batch, sum(self.max_out[1:]), result3d.shape[-1])
        result3d = torch.gather(result3d, 1, batch_indices_3d)  # (batch, K*max_output, 10)
        result3d = result3d * valid_mask[..., None]

        return result2d, result3d

    def append_anchor_inds(self, pred):
        pred["anchor_ind"] = []
        num_anchor = cfg3d.ModelOutput.NUM_ANCHORS_PER_SCALE
        for scale in range(len(cfg3d.ModelOutput.FEATURE_SCALES)):
        # for scale in range(1):
            for key in pred:
                if key != "whole":
                    fmap_shape = pred[key][scale].shape[:-1]
                    break
            fmap_shape = (*fmap_shape, 1)

            ones_map = torch.ones(fmap_shape, dtype=torch.float32, device=cfg3d.Train.DEVICE)
            anchor_list = range(scale * num_anchor, (scale + 1) * num_anchor)
            pred["anchor_ind"].append(self.anchor_indices(ones_map, anchor_list))
        return pred

    def merged_scale(self, pred):
        slice_keys = list(pred.keys())  # ['yxhw', 'object', 'category']
        merged_pred = {}
        # merge pred features over scales
        for key in slice_keys:
            if key != "whole":
                merged_pred[key] = torch.cat(pred[key], dim=1)  # (batch, N, dim)
        return merged_pred

    def anchor_indices(self, ones_map, anchor_list):
        batch, hwa, _ = ones_map.shape
        num_anchor = cfg3d.ModelOutput.NUM_ANCHORS_PER_SCALE
        anchor_list = torch.tensor(anchor_list, dtype=torch.float32, device=cfg3d.Train.DEVICE)
        anchor_index = anchor_list[..., None]
        split_anchor_shape = torch.reshape(ones_map, (batch,1, num_anchor, hwa // num_anchor))

        split_anchor_map = split_anchor_shape * anchor_index
        merge_anchor_map = torch.reshape(split_anchor_map, (batch, hwa, -1))

        return merge_anchor_map


def test_nms():
    # 2 n 4
    ctgr_boxes = torch.tensor([[[300, 300, 400, 400], [320, 320, 400, 400], [350, 350, 400, 400], [300, 600, 400, 700],
                                [500, 600, 600, 700], [300, 600, 350, 650], [0, 0, 0, 0], [0, 0, 0, 0]],
                               [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                [40, 40, 200, 200], [30, 30, 100, 100], [300, 300, 400, 400], [70, 70, 180, 180],
                                ], ], dtype=torch.float32)
                 #/ torch.tensor([1000, 1000, 1000, 1000], dtype=torch.float32)
    # 2 n
    ctgr_scores = torch.tensor([[0.4, 0.4, 0.2, 0.4, 0.4, 0.2, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.8, 0.5, ], ])
    ctgr_idx = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 0, 0, 0, 0], ])
    # iou_thresh = [0.3, 0.3, 0.3]

    for i in range(ctgr_scores.shape[0]):
        boxes = ctgr_boxes[i]
        scores = ctgr_scores[i]
        idxs = ctgr_idx[i]
        # keep = scores > 0.3
        # ctgr_boxes_slice = boxes[keep]
        # ctgr_scores_slice = scores[keep]
        # ctgr_idx_slice = idxs[keep]
        selected_indices = torchvision.ops.nms(
            boxes=boxes,
            scores=scores,
            iou_threshold=0.5,
        )

        print(selected_indices)


if __name__ == '__main__':
    test_nms()