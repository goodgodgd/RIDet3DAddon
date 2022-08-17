import pandas as pd
from timeit import default_timer as timer

import config as cfg
from log.metric import *
from log.logger_pool import LogMeanLoss, LogPositiveObj, LogNegativeObj


class HistoryLog:
    def __init__(self, loss_names):
        self.columns = loss_names + cfg.Log.HistoryLog.SUMMARY
        self.loggers = self.create_loggers(self.columns)
        self.data = pd.DataFrame()
        self.summary = dict()

    def __call__(self, step, grtr, pred, loss, total_loss):
        # result = {"step": step}
        result = dict()
        for key, log_object in self.loggers.items():
            result[key] = log_object(grtr, pred, loss)
        num_ctgr = pred["feat"]["category"][0].shape[-1]
        metric = count_true_positives(grtr["inst"]["bboxes"], pred["inst"]["bboxes"], grtr["inst"]["dontcare"], num_ctgr)
        result.update({"total_loss": total_loss.numpy()})
        result.update(metric)
        result["dist_diff"] = self.distance_diff(grtr, pred)
        self.data = self.data.append(result, ignore_index=True)

    def create_loggers(self, columns):
        loggers = dict()
        if "ciou" in columns:
            loggers["ciou"] = LogMeanLoss("ciou")
        if "iou_aware" in columns:
            loggers["iou_aware"] = LogMeanLoss("iou_aware")
        if "object" in columns:
            loggers["object"] = LogMeanLoss("object")
        if "category" in columns:
            loggers["category"] = LogMeanLoss("category")
        if "sign_ctgr" in columns:
            loggers["sign_ctgr"] = LogMeanLoss("sign_ctgr")
        if "mark_ctgr" in columns:
            loggers["mark_ctgr"] = LogMeanLoss("mark_ctgr")
        if "sign_speed" in columns:
            loggers["sign_speed"] = LogMeanLoss("sign_speed")
        if "mark_speed" in columns:
            loggers["mark_speed"] = LogMeanLoss("mark_speed")
        if "distance" in columns:
            loggers["distance"] = LogMeanLoss("distance")
        if "pos_obj" in columns:
            loggers["pos_obj"] = LogPositiveObj("pos_obj")
        if "neg_obj" in columns:
            loggers["neg_obj"] = LogNegativeObj("neg_obj")
        return loggers

    def distance_diff(self, grtr, pred):
        grtr_bbox_augmented = self.exapand_grtr_bbox(grtr, pred)
        iou_thresh = cfg.Validation.TP_IOU_THRESH
        pred_valid, pred_far = split_pred_far(pred["inst"]["bboxes"])
        grtr_far, grtr_valid = split_grtr_far(pred_far, grtr_bbox_augmented, iou_thresh)
        batch, M, _ = pred_valid["category"].shape
        iou = uf.compute_iou_general(grtr_valid["yxhw"], pred_valid["yxhw"]).numpy()  # (batch, N, M)

        best_iou = np.max(iou, axis=-1)  # (batch, N)

        best_idx = np.argmax(iou, axis=-1)  # (batch, N)
        if len(iou_thresh) > 1:
            iou_thresh = get_iou_thresh_per_class(grtr_valid["category"], iou_thresh)
        iou_match = best_iou > iou_thresh  # (batch, N)

        pred_ctgr_aligned = numpy_gather(pred_valid["category"], best_idx, 1)  # (batch, N, 8)
        pred_dist_aligned = numpy_gather(pred_valid["distance"], best_idx, 1)  # (batch, N, 8)
        grtr_dist_mask = grtr_valid["distance"][..., 0] > 0
        ctgr_match = grtr_valid["category"][..., 0] == pred_ctgr_aligned  # (batch, N)
        grtr_dist_tp_mask = np.expand_dims(iou_match * ctgr_match * grtr_dist_mask, axis=-1)
        grtr_valid_dist = grtr_valid["distance"] * grtr_dist_tp_mask
        pred_valid_dist = pred_dist_aligned * grtr_dist_tp_mask[..., 0]
        grtr_dist = grtr_valid_dist[grtr_dist_tp_mask > 0]
        pred_dist = pred_valid_dist[grtr_dist_tp_mask[..., 0] > 0]
        distance_diff = np.abs(grtr_dist - pred_dist) / (grtr_dist + 1e-5)
        return np.mean(distance_diff)

    def exapand_grtr_bbox(self, grtr, pred):
        grtr_boxes = self.merge_scale_hwa(grtr["feat"])
        pred_boxes = self.merge_scale_hwa(pred["feat"])

        best_probs = np.max(pred_boxes["category"], axis=-1, keepdims=True)
        grtr_boxes["pred_ctgr_prob"] = best_probs
        grtr_boxes["pred_object"] = pred_boxes["object"]
        grtr_boxes["pred_score"] = best_probs * pred_boxes["object"]

        batch, numbox, _ = grtr["inst"]["bboxes"]["yxhw"].shape
        objectness = grtr_boxes["object"]
        for key in grtr_boxes:
            features = []
            for frame_idx in range(batch):
                valid_mask = objectness[frame_idx, :, 0].astype(np.bool)
                feature = grtr_boxes[key]
                feature = feature[frame_idx, valid_mask, :]
                feature = np.pad(feature, [(0, numbox - feature.shape[0]), (0, 0)])
                features.append(feature)

            features = np.stack(features, axis=0)
            grtr_boxes[key] = features

        return grtr_boxes

    def merge_scale_hwa(self, features):
        stacked_feat = {}
        slice_keys = list(features.keys())  # ['yxhw', 'object', 'category']
        for key in slice_keys:
            # list of (batch, HWA in scale, dim)
            # scaled_preds = [features[scale_name][key] for scale_name in range(len(features))]
            scaled_preds = np.concatenate(features[key], axis=1)  # (batch, N, dim)
            stacked_feat[key] = scaled_preds
        return stacked_feat

    def finalize(self, start):
        epoch_time = (timer() - start) / 60
        self.make_summary(epoch_time)
        # make summary dataframe

    def make_summary(self, epoch_time):
        mean_result = self.data.mean(axis=0).to_dict()
        sum_result = self.data.sum(axis=0).to_dict()
        sum_result = {"recall": sum_result["trpo"] / (sum_result["grtr"] + 1e-5),
                      "precision": sum_result["trpo"] / (sum_result["pred"] + 1e-5)}
        metric_keys = ["trpo", "grtr", "pred"]
        summary = {key: val for key, val in mean_result.items() if key not in metric_keys}
        summary.update(sum_result)
        summary["time_m"] = round(epoch_time, 5)
        print("epoch summary:", summary)
        self.summary = summary

    def get_summary(self):
        return self.summary









