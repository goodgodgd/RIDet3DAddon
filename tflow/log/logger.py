import numpy as np
import os.path as op
import pandas as pd
import os

from log.exhaustive_log import ExhaustiveLog
from log.history_log import HistoryLog
from log.visual_log import VisualLog
import utils.tflow.util_function as uf
import RIDet3DAddon.tflow.utils.util_function as uf3d
import RIDet3DAddon.tflow.model.model_util as mu
import config as cfg
import RIDet3DAddon.tflow.config_dir.util_config as uc


class Logger:
    def __init__(self, visual_log, exhaustive_log, ckpt_path, epoch, is_train, val_only):
        self.history_logger = HistoryLog()
        self.exhaustive_logger = ExhaustiveLog() if exhaustive_log else None
        self.visual_logger = VisualLog(ckpt_path, epoch) if visual_log else None
        self.history_filename = op.join(ckpt_path, "history.csv")
        self.exhaust_path = op.join(ckpt_path, "exhaust_log")
        self.num_channel = cfg.ModelOutput.NUM_MAIN_CHANNELS
        if not op.isdir(self.exhaust_path):
            os.makedirs(self.exhaust_path, exist_ok=True)
        self.nms = mu.NonMaximumSuppression()
        self.is_train = is_train
        self.epoch = epoch
        self.ckpt_path = ckpt_path
        self.val_only = val_only

    def log_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        self.check_nan(grtr, "grtr")
        self.check_nan(pred, "pred")
        self.check_nan(loss_by_type, "loss")
        grtr_slices = uf3d.merge_and_slice_features(grtr, True)
        pred_slices = uf3d.merge_and_slice_features(pred, False)
        nms_boxes = self.nms(pred_slices["feat2d"])
        pred_slices["inst"]["bboxes2d"] = uf.slice_feature(nms_boxes, uc.get_bbox_composition(False))
        nms_boxes = self.nms(pred_slices["feat3d"])
        pred_slices["inst"]["bboxes3d"] = uf.slice_feature(nms_boxes, uc.get_3d_bbox_composition(False))

        for key, feature_slices in grtr_slices.items():
            grtr_slices[key] = uf.convert_tensor_to_numpy(feature_slices)
        for key, feature_slices in pred_slices.items():
            pred_slices[key] = uf.convert_tensor_to_numpy(feature_slices)
        loss_by_type = uf.convert_tensor_to_numpy(loss_by_type)

        if step == 0 and self.epoch == 0:
            structures = {"grtr": grtr_slices, "pred": pred_slices, "loss": loss_by_type}
            self.save_model_structure(structures)

        self.history_logger(step, grtr_slices, pred_slices, loss_by_type, total_loss)
        if self.exhaustive_logger:
            self.exhaustive_logger(step, grtr_slices, pred_slices, loss_by_type, total_loss)
        if self.visual_logger:
            self.visual_logger(step, grtr_slices, pred_slices)

    def check_nan(self, features, feat_name):
        valid_result = True
        for name, value in features.items():
            if isinstance(value, dict):
                for sub_name, tensor in value.items():
                    if not np.isfinite(tensor.numpy()).all():
                        print(f"nan {feat_name}:", sub_name, np.quantile(tensor.numpy(), np.linspace(0, 1, self.num_channel)))
                        valid_result = False
            elif isinstance(value, list):
                for idx, tensor in enumerate(value):
                    if tensor.ndim == 0 and (np.isnan(tensor) or np.isinf(tensor) or tensor > 100000000):
                        print(f"nan loss: {name}, {value}")
                        valid_result = False
                    elif not np.isfinite(tensor.numpy()).all():
                        print(f"nan {feat_name}: {idx} feat", np.quantile(tensor.numpy(), np.linspace(0, 1, self.num_channel)))
                        valid_result = False
            else:
                if value.ndim == 0 and (np.isnan(value) or np.isinf(value) or value > 100000000):
                    print(f"nan loss: {name}, {value}")
                    valid_result = False
                elif not np.isfinite(value.numpy()).all():
                    print(f"nan {feat_name}:", name, np.quantile(value.numpy(), np.linspace(0, 1, self.num_channel)))
                    valid_result = False
        assert valid_result

    def finalize(self, start):
        self.history_logger.finalize(start)
        if self.exhaustive_logger:
            self.save_exhaustive_log(start)
        if self.val_only:
            self.save_val_log()
        else:
            self.save_log()

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
