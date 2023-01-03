import os
import numpy as np
import pandas as pd
import os.path as op

import RIDet3DAddon.config as cfg3d
import utils.tflow.util_function as uf
from RIDet3DAddon.tflow.log.metric import split_true_false, split_tp_fp_fn_3d


class SaveAnalyze:
    def __init__(self, result_path):
        self.error_path = op.join(result_path, "error")
        if not op.isdir(self.error_path):
            os.makedirs(self.error_path, exist_ok=True)
        self.data_path = op.join(result_path, "box_data")
        if not op.isdir(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg3d.Dataloader.CATEGORY_NAMES["category"])}
        self.data = pd.DataFrame()

    def __call__(self, step, grtr, pred, splits):
        batch, _, __ = pred["inst2d"]["yxhw"].shape
        data = dict()
        for i in range(batch):
            pred_tp_3d = self.extract_batch_valid_data(splits["3d"]["pred_tp"], i, "category")
            pred_fp_3d = self.extract_batch_valid_data(splits["3d"]["analyze_pred_fp"], i, "category")
            if len(pred_tp_3d["yx"]) > 0:
                tp_error = {}
                for key in splits["3d"]['pred_tp'].keys():
                    if "probs" in key:
                        tp_error[key] = (1 - splits["3d"]['pred_tp'][key][i]) * (splits["3d"]["pred_tp"]["z"][i] > 0)
                    else:
                        tp_error[key] = np.abs(
                            splits["3d"]['grtr_tp'][key][i, ...] - splits["3d"]['pred_tp'][key][i, ...])
                        if key == "theta":
                            if np.any(tp_error[key] > 3*np.pi/2):
                                error_index = np.where(tp_error[key][..., 0] > 3 * np.pi / 2)
                                tp_error[key][error_index, 0] = np.abs(tp_error[key][error_index, 0] - 2 * np.pi)
                            if np.any(tp_error[key] > np.pi/2):
                                error_index = np.where(tp_error[key][..., 0] > np.pi/2)
                                tp_error[key][error_index, 0] = np.abs(tp_error[key][error_index, 0] - np.pi)
                tp_error_data = self.extract_valid_data(tp_error, "z")
                for n in range(tp_error_data["yx"].shape[0]):
                    self.data = self.data.append(self.update_data(pred_tp_3d, tp_error_data, n, "tp"), ignore_index=True)
            if len(pred_fp_3d["yx"]) > 0:
                fp_error = {}
                for key in splits["3d"]['analyze_pred_fp'].keys():
                    if "probs" in key:
                        fp_error[key] = (1 - splits["3d"]["analyze_pred_fp"][key][i]) * (splits["3d"]["analyze_pred_fp"]["z"][i] > 0)
                    else:
                        fp_error[key] = np.abs(
                            splits["3d"]['analyze_grtr_fn'][key][i, ...] - splits["3d"]['analyze_pred_fp'][key][i, ...])\
                                        * (splits["3d"]["analyze_pred_fp"]["z"][i] > 0)
                        if key == "theta":
                            if np.any(fp_error[key][..., 0] > 3*np.pi/2):
                                error_index = np.where(fp_error[key][..., 0] > 3*np.pi/2)
                                fp_error[key][error_index, 0] = np.abs(fp_error[key][error_index, 0] - 2 * np.pi)
                            if np.any(fp_error[key][..., 0] > np.pi/2):
                                error_index = np.where(fp_error[key][..., 0] > np.pi/2)
                                fp_error[key][error_index, 0] = np.abs(fp_error[key][error_index, 0] - np.pi)
                fp_error_data = self.extract_valid_data(fp_error, "z")
                for n in range(fp_error_data["yx"].shape[0]):
                    self.data = self.data.append(self.update_data(pred_fp_3d, fp_error_data, n, "fp"), ignore_index=True)
        # if len(data) > 0:
        #     self.data = self.data.append(data, ignore_index=True)

    def extract_valid_data(self, inst_data, mask_key):
        """
        remove zero padding from bboxes
        """
        valid_data = {}
        valid_mask = (inst_data[mask_key][..., 0] > 0).flatten()
        for key, data in inst_data.items():
            valid_data[key] = data[valid_mask]
        return valid_data

    def extract_batch_valid_data(self, inst_data, i, mask_key):
        """
        remove zero padding from bboxes
        """
        valid_data = {}
        valid_mask = (inst_data[mask_key][i] > 0).flatten()
        for key, data in inst_data.items():
            valid_data[key] = data[i][valid_mask]
        return valid_data

    def text_write_data(self, data, n):
        text_to_write = ''
        yx = data["yx"][n]
        z = data["z"][n]
        hwl = data["hwl"][n]
        ry = data["theta"][n]
        # if ry > 3*np.pi/2:
        #     ry = np.abs(ry - 2 * np.pi)
        # if ry > np.pi/2:
        #     ry = np.abs(ry - np.pi)
        # assert ry < np.pi/2, ry
        # alpha = np.arctan2(z, yx[-1])
        # occluded = error_data["occluded"][n]
        occluded_probs = data["occluded_probs"][n]
        objectness = data["object_probs"][n]
        category_probs = data["category_probs"][n]
        # category = error_data["category"][n]
        # ctgr = self.categories[ctgr[0]]
        category_probs = self.np2str(category_probs)
        occluded_probs = self.np2str(occluded_probs)
        text_to_write += (category_probs + occluded_probs +
                          f'{objectness[0]:.6f} {yx[1]:.6f} {yx[0] + (hwl[0] / 2):.6f} {hwl[0]:.6f} {hwl[1]:.6f} '
                          f'{hwl[2]:.6f} {z[0]:.6f} {ry[0]:.6f}\n')
        return text_to_write

    def np2str(self, data):
        str_data = ''
        for i in range(len(data)):
            str_data += f"{data[i]:.6f} "
        return str_data

    def update_data(self, bbox_3d, error, n, key):
        data = dict()
        data["y"] = bbox_3d["yx"][n][0]
        data["x"] = bbox_3d["yx"][n][1]
        data["z"] = bbox_3d["z"][n][0]
        data["h"] = bbox_3d["hwl"][n][0]
        data["w"] = bbox_3d["hwl"][n][1]
        data["l"] = bbox_3d["hwl"][n][2]
        data["theta"] = bbox_3d["theta"][n][0]
        data["occluded"] = bbox_3d["occluded"][n][0]
        data["occluded_probs_0"] = bbox_3d["occluded_probs"][n][0]
        data["occluded_probs_1"] = bbox_3d["occluded_probs"][n][1]
        data["occluded_probs_2"] = bbox_3d["occluded_probs"][n][2]
        data["object_probs"] = bbox_3d["object_probs"][n][0]
        data["category"] = bbox_3d["category"][n][0]
        data["category_probs_0"] = bbox_3d["category_probs"][n][0]
        data["category_probs_1"] = bbox_3d["category_probs"][n][1]
        data["category_probs_2"] = bbox_3d["category_probs"][n][2]
        data["category_probs_3"] = bbox_3d["category_probs"][n][3]

        data["|"] = "|"

        data["e_y"] = error["yx"][n][0]
        data["e_x"] = error["yx"][n][1]
        data["e_z"] = error["z"][n][0]
        data["e_h"] = error["hwl"][n][0]
        data["e_w"] = error["hwl"][n][1]
        data["e_l"] = error["hwl"][n][2]
        data["e_theta"] = error["theta"][n][0]
        data["e_occluded"] = error["occluded"][n][0]
        data["e_occluded_probs_0"] = error["occluded_probs"][n][0]
        data["e_occluded_probs_1"] = error["occluded_probs"][n][1]
        data["e_occluded_probs_2"] = error["occluded_probs"][n][2]
        data["e_object_probs"] = error["object_probs"][n][0]
        data["e_category"] = error["category"][n][0]
        data["e_category_probs_0"] = error["category_probs"][n][0]
        data["e_category_probs_1"] = error["category_probs"][n][1]
        data["e_category_probs_2"] = error["category_probs"][n][2]
        data["e_category_probs_3"] = error["category_probs"][n][3]
        data["status"] = f"{key}"
        return data

    def get_summary(self):
        return self.data

