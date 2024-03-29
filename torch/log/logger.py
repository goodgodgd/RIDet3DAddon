import numpy as np
import os.path as op
import pandas as pd
import os
import glob
import cv2
from timeit import default_timer as timer

from RIDet3DAddon.torch.log.exhaustive_log import ExhaustiveLog
from RIDet3DAddon.torch.log.history_log import HistoryLog
from RIDet3DAddon.torch.log.visual_log import VisualLog2d, VisualLog3d
from dataloader.readers.kitti_reader import SensorConfig
import utils.torch.util_function as uf
import RIDet3DAddon.torch.model.nms as nms
import RIDet3DAddon.torch.config as cfg3d
import RIDet3DAddon.torch.config_dir.util_config as uc3d
from RIDet3DAddon.torch.log.save_pred import SavePred


class Logger:
    def __init__(self, visual_log, exhaustive_log, loss_names, ckpt_path, epoch, is_train, val_only):
        self.history_logger = HistoryLog(loss_names)
        self.save_pred = SavePred(op.join(ckpt_path, "result"))
        self.exhaustive_logger = ExhaustiveLog(loss_names) if exhaustive_log else None
        self.visual_logger_2d = VisualLog2d(ckpt_path, epoch) if visual_log else None
        self.visual_logger_3d = VisualLog3d(ckpt_path, epoch) if visual_log else None
        # self.visual_logger = VisualLog(ckpt_path, epoch)
        self.history_filename = op.join(ckpt_path, "history.csv")
        self.exhaust_path = op.join(ckpt_path, "exhaust_log")
        self.num_channel = cfg3d.ModelOutput.NUM_MAIN_CHANNELS
        if not op.isdir(self.exhaust_path):
            os.makedirs(self.exhaust_path, exist_ok=True)
        self.nms = nms.NonMaximumSuppression()
        self.is_train = is_train
        self.epoch = epoch
        self.ckpt_path = ckpt_path
        self.val_only = val_only
        self.orgin_path = cfg3d.Datasets.DATASET_CONFIG.ORIGIN_PATH

    def log_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        self.check_nan(grtr, "grtr")
        self.check_nan(pred, "pred")
        self.check_nan(loss_by_type, "loss")
        nms_2d_box, nms_3d_box = self.nms(pred)
        pred["inst2d"] = uf.slice_feature(nms_2d_box, uc3d.get_bbox_composition(False), -1)
        pred["inst3d"] = uf.slice_feature(nms_3d_box, uc3d.get_3d_bbox_composition(False), -1)
        # pred["inst3d"] = uf.slice_feature(nms_3d_box, uc.get_3d_bbox_composition(False))
        for key, feature_slices in grtr.items():
            grtr[key] = uf.convert_tensor_to_numpy(feature_slices)
        for key, feature_slices in pred.items():
            pred[key] = uf.convert_tensor_to_numpy(feature_slices)
        loss_by_type = uf.convert_tensor_to_numpy(loss_by_type)
        pred["inst3d"]["cxyz"] = self.velo_to_cam(pred["inst3d"], grtr["frame_names"])
        if step == 0 and self.epoch == 0:
            structures = {"grtr": grtr, "pred": pred, "loss": loss_by_type}
            self.save_model_structure(structures)

        self.history_logger(step, grtr, pred, loss_by_type, total_loss)
        # TODO
        if self.exhaustive_logger:
            self.exhaustive_logger(step, grtr, pred, loss_by_type, total_loss)
        # TODO draw 3d box 매칭되면 파란색, 매칭되지 않으면 빨간색의 6면
        if self.visual_logger_2d:
            # nms_3d_box = self.nms(pred, is_3d=True)
            self.visual_logger_2d(step, grtr, pred)
            self.visual_logger_3d(step, grtr, pred)
            if self.val_only:
                self.save_pred(step, grtr, pred)

    def check_nan(self, features, feat_name):
        valid_result = True
        if isinstance(features, dict):
            for name, value in features.items():
                self.check_nan(value, f"{feat_name}_{name}")
        elif isinstance(features, list):
            for idx, tensor in enumerate(features):
                self.check_nan(tensor, f"{feat_name}_{idx}")
        elif isinstance(features, str):
            pass
        else:
            features = features.cpu().detach().numpy()
            if features.ndim == 0 and (np.isnan(features) or np.isinf(features) or features > 100000000):
                print(f"nan loss: {feat_name}, {features}")
                valid_result = False
            elif not np.isfinite(features).all():
                print(f"nan {feat_name}:", np.quantile(features, np.linspace(0, 1, self.num_channel)))
                valid_result = False
        assert valid_result


    def camera_2_velo(self, centroid, calib, plane_model):
        pts_3d_ref = np.transpose(np.dot(np.linalg.inv(calib.R0), np.transpose(centroid[np.newaxis, ...])))
        n = pts_3d_ref.shape[0]
        pts_3d_ref = np.hstack((pts_3d_ref, np.ones((n, 1))))
        he = np.array([0, 0, plane_model[3] * 3 / 2]).reshape([1, 3])
        centroid = np.dot(pts_3d_ref, np.transpose(calib.C2V)) + he
        return centroid

    def velo_to_cam(self, pred, frame_names):
        batch_cxyz = []
        for i in range(pred["lxyz"].shape[0]):
            frame_num = frame_names[i].split("/")[-1]
            origin_file = os.path.join(self.orgin_path, f"{frame_num}")
            calib_file = origin_file.replace("image_2", "calib").replace("png", "txt")
            calib = SensorConfig(calib_file)
            cxyz = calib.project_velo_to_rect(pred["lxyz"][i])
            batch_cxyz.append(cxyz)
        return np.stack(batch_cxyz, axis=0)

    def cam_to_velo(self, pred, frame_names):
        batch_cxyz = []
        for i in range(len(frame_names)):
            frame_num = frame_names[i].split("/")[-1]
            origin_file = os.path.join(self.orgin_path, f"{frame_num}")
            calib_file = origin_file.replace("image_2", "calib").replace("png", "txt")
            calib = SensorConfig(calib_file)
            cxyz = calib.project_rect_to_velo(pred["cxyz"][i])
            batch_cxyz.append(cxyz)
        return np.stack(batch_cxyz, axis=0)

        # divider = np.array([[origin_image.shape[1]], [origin_image.shape[0]], [1]])
        # P2 /= divider
        return calib

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
