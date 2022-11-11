import pandas as pd
from timeit import default_timer as timer

import config as cfg
from RIDet3DAddon.tflow.log.metric import *
from RIDet3DAddon.tflow.log.logger_pool import LogMeanLoss, LogPositiveObj, LogNegativeObj
from RIDet3DAddon.tflow.model.nms import Compute3DIoU


class HistoryLog:
    def __init__(self, loss_names):
        self.columns = loss_names + cfg.Log.HistoryLog.SUMMARY
        self.loggers = self.create_loggers(self.columns)
        self.data = pd.DataFrame()
        self.summary = dict()
        self.iou3d = Compute3DIoU()

    def __call__(self, step, grtr, pred, loss, total_loss):
        result = dict()
        for key, log_object in self.loggers.items():
            result[key] = log_object(grtr, pred, loss)
        num_ctgr = pred["feat2d"]["category"][0].shape[-1]
        metric = count_true_positives(grtr["inst2d"], pred["inst2d"], num_ctgr)
        result.update({"total_loss": total_loss.numpy()})
        result.update(metric)
        # metric_3d = count_true_positives_3d(grtr["inst3d"], pred["inst3d"], grtr["inst2d"]["object"], num_ctgr)
        # result.update(metric_3d)
        self.data = self.data.append(result, ignore_index=True)

    def create_loggers(self, columns):
        loggers = dict()
        if "box_2d" in columns:
            loggers["box_2d"] = LogMeanLoss("box_2d")
        if "object" in columns:
            loggers["object"] = LogMeanLoss("object")
        # if "neg_object" in columns:
        #     loggers["neg_object"] = LogMeanLoss("neg_object")
        if "category" in columns:
            loggers["category"] = LogMeanLoss("category")
        if "yx" in columns:
            loggers["yx"] = LogMeanLoss("yx")
        if "hwl" in columns:
            loggers["hwl"] = LogMeanLoss("hwl")
        if "depth" in columns:
            loggers["depth"] = LogMeanLoss("depth")
        if "theta" in columns:
            loggers["theta"] = LogMeanLoss("theta")
        if "pos_obj" in columns:
            loggers["pos_obj"] = LogPositiveObj("pos_obj")
        if "neg_obj" in columns:
            loggers["neg_obj"] = LogNegativeObj("neg_obj")
        return loggers

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
        # sum_result = {"recall": sum_result["trpo"] / (sum_result["grtr"] + 1e-5),
        #               "precision": sum_result["trpo"] / (sum_result["pred"] + 1e-5),
        #               "recall3d": sum_result["trpo3d"] / (sum_result["grtr3d"] + 1e-5),
        #               "precision3d": sum_result["trpo3d"] / (sum_result["pred3d"] + 1e-5)
        #               }
        # metric_keys = ["trpo", "grtr", "pred", "trpo3d", "grtr3d", "pred3d"]
        summary = {key: val for key, val in mean_result.items() if key not in metric_keys}
        summary.update(sum_result)
        summary["time_m"] = round(epoch_time, 5)
        print("epoch summary:", summary)
        self.summary = summary

    def get_summary(self):
        return self.summary









