import numpy as np
import os.path as op
import pandas as pd
import os
from timeit import default_timer as timer

from RIDet3DAddon.tflow.log.exhaustive_log import ExhaustiveLog
from RIDet3DAddon.tflow.log.history_log import HistoryLog
from RIDet3DAddon.tflow.log.visual_log import VisualLog
from RIDet3DAddon.tflow.log.save_pred import SavePred
from RIDet3DAddon.tflow.log.save_analyze import SaveAnalyze
from RIDet3DAddon.tflow.log.metric import split_true_false, split_tp_fp_fn_3d
import utils.tflow.util_function as uf
import RIDet3DAddon.tflow.train.train_util as tu3d
import RIDet3DAddon.tflow.model.nms as nms
import RIDet3DAddon.config as cfg
import RIDet3DAddon.tflow.config_dir.util_config as uc


class Logger:
    def __init__(self, visual_log, exhaustive_log, loss_names, ckpt_path, epoch, is_train, val_only):
        self.history_logger = HistoryLog(loss_names)
        self.exhaustive_logger = ExhaustiveLog(loss_names) if exhaustive_log else None
        self.save_pred = SavePred(op.join(ckpt_path, "result"))
        self.save_analyze = SaveAnalyze(op.join(ckpt_path, "anaylze"))
        self.visual_logger = VisualLog(ckpt_path, epoch) if visual_log else None
        self.history_filename = op.join(ckpt_path, "history.csv")
        self.exhaust_path = op.join(ckpt_path, "exhaust_log")
        self.num_channel = cfg.ModelOutput.NUM_MAIN_CHANNELS
        if not op.isdir(self.exhaust_path):
            os.makedirs(self.exhaust_path, exist_ok=True)
        self.nms = nms.NonMaximumSuppression()
        self.is_train = is_train
        self.epoch = epoch
        self.ckpt_path = ckpt_path
        self.val_only = val_only

    def log_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        self.check_nan(grtr, "grtr")
        self.check_nan(pred, "pred")
        self.check_nan(loss_by_type, "loss")
        # pred["feat3d"] = tu3d.box_preprocess(grtr["feat3d"], pred["feat3d"], 0)
        nms_2d_box, nms_3d_box = self.nms(pred)

        pred["inst2d"] = uf.slice_feature(nms_2d_box, uc.get_bbox_composition(False))
        pred["inst3d"] = uf.slice_feature(nms_3d_box, uc.get_3d_bbox_composition(False))

        for key, feature_slices in grtr.items():
            grtr[key] = uf.convert_tensor_to_numpy(feature_slices)
        for key, feature_slices in pred.items():
            pred[key] = uf.convert_tensor_to_numpy(feature_slices)
        loss_by_type = uf.convert_tensor_to_numpy(loss_by_type)

        if step == 0 and self.epoch == 0:
            structures = {"grtr": grtr, "pred": pred, "loss": loss_by_type}
            self.save_model_structure(structures)

        self.history_logger(step, grtr, pred, loss_by_type, total_loss)
        # if self.exhaustive_logger:
        #     self.exhaustive_logger(step, grtr, pred, loss_by_type, total_loss)
        if self.visual_logger:
            splits = {}
            grtr_bbox_augmented = self.exapand_grtr_bbox(grtr, pred)
            splits["2d"] = split_true_false(grtr_bbox_augmented["inst2d"], pred["inst2d"], cfg.Validation.TP_IOU_THRESH)
            splits["3d"] = split_tp_fp_fn_3d(grtr_bbox_augmented, pred, cfg.Validation.TP_IOU_THRESH)
            # self.visual_logger(step, grtr, pred, splits)
            # self.save_pred(step, grtr, pred)
            self.save_analyze(step, grtr, pred, splits)

    def check_nan(self, features, feat_name):
        if "merged" not in feat_name:
            valid_result = True
            if isinstance(features, dict):
                for name, value in features.items():
                    self.check_nan(value, f"{feat_name}_{name}")
            elif isinstance(features, list):
                for idx, tensor in enumerate(features):
                    self.check_nan(tensor, f"{feat_name}_{idx}")
            else:
                if features.ndim == 0 and (np.isnan(features) or np.isinf(features) or features > 100000000000):
                    print(f"nan loss: {feat_name}, {features}")
                    valid_result = False
                elif not np.isfinite(features.numpy()).all():
                    print(f"nan {feat_name}:", np.quantile(features.numpy(), np.linspace(0, 1, self.num_channel)))
                    valid_result = False
            assert valid_result

    def finalize(self, start):
        self.history_logger.finalize(start)
        # if self.exhaustive_logger:
        #     self.save_exhaustive_log(start)
        if self.val_only:
            self.save_val_log()
        else:
            self.save_log()
        if self.visual_logger:
            self.save_analyze_data()

    def save_log(self):
        logger_summary = self.history_logger.get_summary()
        if self.is_train:
            train_summary = {"epoch": self.epoch}
            train_summary.update({"!" + key: val for key, val in logger_summary.items()})
            train_summary.update({"|": "|"})
            if op.isfile(self.history_filename):
                history_summary = pd.read_csv(self.history_filename, encoding='utf-8',
                                              converters={'epoch': lambda c: int(c)})
                history_summary = history_summary.append(train_summary, ignore_index=True)
            else:
                history_summary = pd.DataFrame([train_summary])
        else:
            history_summary = pd.read_csv(self.history_filename, encoding='utf-8',
                                          converters={'epoch': lambda c: int(c)})
            for key, val in logger_summary.items():
                history_summary.loc[self.epoch, "`" + key] = val

        history_summary["epoch"] = history_summary["epoch"].astype(int)
        print("=== history\n", history_summary)
        history_summary.to_csv(self.history_filename, encoding='utf-8', index=False, float_format='%.4f')

    def save_exhaustive_log(self, start):
        self.exhaustive_logger.finalize(start)
        exhaust_summary = self.exhaustive_logger.get_summary()
        if self.val_only:
            exhaust_filename = self.exhaust_path + "/exhaust_val.csv"
        else:
            exhaust_filename = self.exhaust_path + f"/epoch{self.epoch:02d}.csv"
        exhaust = pd.DataFrame(exhaust_summary)
        exhaust.to_csv(exhaust_filename, encoding='utf-8', index=False, float_format='%.4f')

    def save_val_log(self):
        logger_summary = self.history_logger.get_summary()
        val_filename = self.history_filename[:-4] + "_val.csv"
        epoch_summary = {"epoch": self.epoch}
        epoch_summary.update({"`" + key: val for key, val in logger_summary.items()})
        history_summary = pd.DataFrame([epoch_summary])
        history_summary["epoch"] = history_summary["epoch"].astype(int)
        print("=== validation history\n", history_summary)
        history_summary.to_csv(val_filename, encoding='utf-8', index=False, float_format='%.4f')

    def save_model_structure(self, structures):
        structure_file = op.join(self.ckpt_path, "structure.md")
        f = open(structure_file, "w")
        for key, structure in structures.items():
            f.write(f"- {key}\n")
            space_count = 1
            self.analyze_structure(structure, f, space_count)
        f.close()

    def analyze_structure(self, data, f, space_count, key=""):
        space = "    " * space_count
        if isinstance(data, list):
            for i, datum in enumerate(data):
                if isinstance(datum, dict):
                    # space_count += 1
                    self.analyze_structure(datum, f, space_count)
                    # space_count -= 1
                elif type(datum) == np.ndarray:
                    f.write(f"{space}- {key}: {datum.shape}\n")
                else:
                    f.write(f"{space}- {datum}\n")
                    space_count += 1
                    self.analyze_structure(datum, f, space_count)
                    space_count -= 1
        elif isinstance(data, dict):
            for sub_key, datum in data.items():
                if type(datum) == np.ndarray:
                    f.write(f"{space}- {sub_key}: {datum.shape}\n")
                else:
                    f.write(f"{space}- {sub_key}\n")

                space_count += 1
                self.analyze_structure(datum, f, space_count, sub_key)
                space_count -= 1

    def save_analyze_data(self):
        analyze_summary = self.save_analyze.get_summary()
        analyze_filename = op.join(self.ckpt_path, "anaylze/analyze_data.csv")
        analyze_summary.to_csv(analyze_filename, encoding='utf-8', index=False, float_format='%.4f')

    def exapand_grtr_bbox(self, grtr, pred):
        grtr_boxes_3d = self.merge_scale_hwa(grtr["feat3d"])
        grtr_boxes_2d = self.merge_scale_hwa(grtr["feat2d"])
        pred_boxes_2d = self.merge_scale_hwa(pred["feat2d"])

        best_probs = np.max(pred_boxes_2d["category"], axis=-1, keepdims=True)
        grtr_boxes_2d["pred_ctgr_prob"] = best_probs
        grtr_boxes_2d["pred_object"] = pred_boxes_2d["object"]
        grtr_boxes_2d["pred_score"] = best_probs * pred_boxes_2d["object"]

        grtr_boxes_3d["pred_ctgr_prob"] = best_probs
        grtr_boxes_3d["pred_object"] = pred_boxes_2d["object"]
        grtr_boxes_3d["pred_score"] = best_probs * pred_boxes_2d["object"]

        batch, _, __ = grtr["inst2d"]["yxhw"].shape
        # numbox = cfg.Validation.MAX_BOX
        numbox = grtr["inst2d"]["yxhw"].shape[1]
        objectness = grtr_boxes_2d["object"]
        for key in grtr_boxes_3d:
            features = []
            for frame_idx in range(batch):
                valid_mask = objectness[frame_idx, :, 0].astype(np.bool)
                feature = grtr_boxes_3d[key]
                feature = feature[frame_idx, valid_mask, :]
                feature = np.pad(feature, [(0, numbox - feature.shape[0]), (0, 0)])
                features.append(feature)

            features = np.stack(features, axis=0)
            grtr_boxes_3d[key] = features

        for key in grtr_boxes_2d:
            features = []
            for frame_idx in range(batch):
                valid_mask = objectness[frame_idx, :, 0].astype(np.bool)
                feature = grtr_boxes_2d[key]
                feature = feature[frame_idx, valid_mask, :]
                feature = np.pad(feature, [(0, numbox - feature.shape[0]), (0, 0)])
                features.append(feature)

            features = np.stack(features, axis=0)
            grtr_boxes_2d[key] = features
        return {"inst2d": grtr_boxes_2d, "inst3d": grtr_boxes_3d}

    def merge_scale_hwa(self, features):
        stacked_feat = {}
        slice_keys = list(features.keys())  # ['yxhw', 'object', 'category']
        for key in slice_keys:
            if key == "merged":
                continue
            # list of (batch, HWA in scale, dim)
            # scaled_preds = [features[scale_name][key] for scale_name in range(len(features))]
            scaled_preds = np.concatenate(features[key], axis=1)  # (batch, N, dim)
            stacked_feat[key] = scaled_preds
        return stacked_feat
