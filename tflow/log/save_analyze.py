import os
import numpy as np
import pandas as pd
import os.path as op
import cv2

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
        batch, _, __ = pred["inst"]["yxhw"].shape
        for i in range(batch):
            pred_tp_3d = self.extract_batch_valid_data(splits["3d"]["pred_tp"], i, "category")
            pred_fp_3d = self.extract_batch_valid_data(splits["3d"]["analyze_pred_fp"], i, "category")
            pred_tp_3d["occlusion_ratio"] = self.occlusion_ratio(pred_tp_3d)
            pred_fp_3d["occlusion_ratio"] = self.occlusion_ratio(pred_fp_3d)
            if len(pred_tp_3d["yx"]) > 0:
                tp_error = {}
                for key in splits["3d"]['pred_tp'].keys():
                    if ("prob" in key) or ("score" in key):
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
                # tp_occl_valid = self.occlusion_ratio(
                #     self.extract_batch_valid_data(splits["3d"]['grtr_tp'], i, "z"))
                tp_error_data = self.extract_valid_data(tp_error, "z")
                # tp_error["occlusion_ratio"] = tp_occl_valid - pred_tp_3d["occlusion_ratio"]
                # pred_tp_3d["occlusion_ratio"] = self.occlusion_ratio(pred_tp_3d)
                # print(tp_error["occlusion_ratio"])
                # self.draw_test(grtr, pred_tp_3d)
                for n in range(tp_error_data["yx"].shape[0]):
                    self.data = self.data.append(self.update_data(pred_tp_3d, tp_error_data, n, "tp"), ignore_index=True)
            if len(pred_fp_3d["yx"]) > 0:
                fp_error = {}
                for key in splits["3d"]['analyze_pred_fp'].keys():
                    if ("prob" in key) or ("score" in key):
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
                # fp_occl_valid = self.occlusion_ratio(self.extract_batch_valid_data(splits["3d"]['analyze_grtr_fn'], i, "z"))
                # fp_occl_valid = self.occlusion_ratio(self.extract_batch_valid_data(splits["3d"]['analyze_grtr_fn'], i, "z"))
                fp_error_data = self.extract_valid_data(fp_error, "z")
                # print(pred_fp_3d["occlusion_ratio"].shape, fp_occl_valid.shape)
                # fp_error["occlusion_ratio"] = fp_occl_valid - pred_fp_3d["occlusion_ratio"]
                # print(fp_error["occlusion_ratio"])
                # pred_fp_3d["occlusion_ratio"] = self.occlusion_ratio(pred_fp_3d)
                # self.draw_test(grtr, pred_fp_3d)
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

    def extract_batch_valid_data_another_mask(self, inst_data, i, mask_data):
        """
        remove zero padding from bboxes
        """
        valid_data = {}
        valid_mask = (mask_data[i] > 0).flatten()
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
        data["object_probs"] = bbox_3d["object"][n][0]
        data["category"] = bbox_3d["category"][n][0]
        data["category_probs_0"] = bbox_3d["category_probs"][n][0]
        data["category_probs_1"] = bbox_3d["category_probs"][n][1]
        data["category_probs_2"] = bbox_3d["category_probs"][n][2]
        data["category_probs_3"] = bbox_3d["category_probs"][n][3]
        data["occlusion_ratio"] = bbox_3d["occlusion_ratio"][n][0]

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
        data["e_object_probs"] = error["object"][n][0]
        data["e_category"] = error["category"][n][0]
        data["e_category_probs_0"] = error["category_probs"][n][0]
        data["e_category_probs_1"] = error["category_probs"][n][1]
        data["e_category_probs_2"] = error["category_probs"][n][2]
        data["e_category_probs_3"] = error["category_probs"][n][3]
        data["e_occlusion_ratio"] = error["occlusion_ratio"][n][0]
        data["status"] = f"{key}"
        # print(f"{key} {bbox_3d['occlusion_ratio'][n][0]}")
        return data

    def get_summary(self):
        return self.data

    def occlusion_ratio(self, grtr_inst):
        batch = grtr_inst["yxhw"].shape[0]
        # batch_occlusion = [[] for i in range(batch)]
        # for b in range(batch):
        yxhw = grtr_inst["yxhw"]
        depth = grtr_inst["z"]
        valid_mask = yxhw[..., 2] > 0
        valid_yxhw = yxhw[valid_mask]
        valid_depth = depth[valid_mask]
        occlusion = np.zeros(valid_yxhw.shape[0])
        for i in range(len(valid_yxhw)):
            for j in range(i + 1, len(valid_yxhw)):
                if self.boxes_overlap(valid_yxhw[i], valid_yxhw[j]):
                    intersection_area = self.boxes_overlap_area(valid_yxhw[i], valid_yxhw[j])
                    if valid_depth[i] > valid_depth[j]:
                        bb_area = valid_yxhw[i][2] * valid_yxhw[i][3]
                        occlusion[i] += intersection_area / bb_area
                        # occlusion[i] += boxes_overlap_area(valid_yxhw[i], valid_yxhw[j])
                    else:
                        bb_area = valid_yxhw[j][2] * valid_yxhw[j][3]
                        occlusion[j] += intersection_area / bb_area
                            # occlusion[j] += boxes_overlap_area(valid_yxhw[i], valid_yxhw[j])
            # pad_zero = np.zeros(50 - len(occlusion))
            # occlusion = np.concatenate([occlusion, pad_zero], axis=0)
            # batch_occlusion[b] = occlusion
        occlusion = np.asarray(occlusion, dtype=np.float32)[..., np.newaxis]
        return occlusion

    def boxes_overlap(self, box1, box2):
        y, x, h1, w1 = box1
        y1 = y - h1 / 2
        x1 = x - w1 / 2
        y, x, h2, w2 = box2
        y2 = y - h2 / 2
        x2 = x - w2 / 2
        return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)

    def boxes_overlap_area(self, box1, box2):
        y, x, h1, w1 = box1
        y1 = y - h1 / 2
        x1 = x - w1 / 2
        y, x, h2, w2 = box2
        y2 = y - h2 / 2
        x2 = x - w2 / 2
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        return x_overlap * y_overlap

    def draw_test(self, grtr, pred_fp_3d):
        # batch = pred_fp_3d["yxhw"].shape[0]
        # for frame_idx in range(batch):
        image_grtr = uf.to_uint8_image(grtr["image"][0]).numpy()
        # valid_mask = pred_fp_3d["hwl"][frame_idx][:, 0] > 0  # (N,) h>0
        box_ctgr = pred_fp_3d["category"].astype(np.int32)  # (N, 1)
        # occluded = bboxes["occluded"][frame_idx][valid_mask, 0].astype(np.int32)  # (N, 1)
        if "occlusion_ratio" in pred_fp_3d.keys():
            occluded = pred_fp_3d["occlusion_ratio"]  # (N, 1)
        else:
            occluded = pred_fp_3d["occluded"].astype(np.int32)
        yx = pred_fp_3d["yx"]
        z = pred_fp_3d["z"]
        hwl = pred_fp_3d["hwl"]
        theta = pred_fp_3d["theta"]
        iou = None
        if "iou" in pred_fp_3d.keys():
            iou = pred_fp_3d["iou"]

            # box_ctgr = box_ctgr[valid_mask, 0].astype(np.int32)
        proj_box, proj_box_center, bev_box = self.extract_corner(grtr["intrinsic"][0], yx, z, hwl, theta)

        front_view = self.draw_cuboid(image_grtr, proj_box, proj_box_center, box_ctgr, occluded, z, (255, 0, 255))
        cv2.imshow("test", front_view)
        cv2.waitKey()

    def extract_corner(self, intrinsic, yx, z, hwl, theta):
        proj_box = list()
        proj_box_center = list()
        bev_box = list()
        for i in range(yx.shape[0]):
            rot = theta[i]
            y = yx[i, 0]
            x = yx[i, 1]
            depth = z[i, 0]
            h = hwl[i, 0]
            w = hwl[i, 1]
            l = hwl[i, 2]
            center = np.asarray([x, y, depth], dtype=np.float32)
            corner_2d = list()
            bev_coner = list()
            for i in [1, -1]:
                for j in [1, -1]:
                    for k in [-1, 1]:
                        point = np.copy(center)
                        point[0] = x + i * w / 2 * np.cos(-rot + np.pi / 2) + (j * i) * l / 2 * np.cos(-rot)
                        point[2] = depth + i * w / 2 * np.sin(-rot + np.pi / 2) + (j * i) * l / 2 * np.sin(-rot)
                        point[1] = y - k * h / 2
                        bev_coner.append(point)

                        point = np.append(point, 1)
                        point = np.dot(intrinsic, point)
                        point = point[:2] / point[2]
                        corner_2d.append(point)
            bev_box.append(np.stack(bev_coner, axis=0))
            proj_box.append(corner_2d)
            center_point = np.dot(intrinsic, np.append(center, 1))
            center_point = center_point[:2] / center_point[2]
            proj_box_center.append(center_point)
        return proj_box, proj_box_center, bev_box

    def draw_cuboid(self, image, proj_box, proj_box_center, category, occluded, z, color):
        proj_box = np.array(proj_box)
        proj_box = (np.reshape(proj_box, (-1, 8, 2)) * image.shape[::-1][1:3]).astype(np.int32)
        proj_box = proj_box.tolist()

        proj_box_center = np.array(proj_box_center)
        proj_box_center = (np.reshape(proj_box_center, (-1, 1, 2)) * image.shape[::-1][1:3]).astype(np.int32)
        proj_box_center = proj_box_center.tolist()

        for i, (point, center) in enumerate(zip(proj_box, proj_box_center)):
            annotation = "dontcare" if category[i, 0] < 0 else f"{self.categories[category[i, 0]]}"
            # occlude = occluded[i]
            occlude = f"{occluded[i, 0]:.4f}"
            depth = f"{z[i, 0]:.2f}"
            image = self.draw_line(image, point, center, color, annotation, occlude, depth)
        return image

    def draw_line(self, image, line, center, color, annotation, occlude, depth):
        for i in range(4):
            image = cv2.line(image, line[i * 2], line[i * 2 + 1], color)
        for i in range(8):
            image = cv2.line(image, line[i], line[(i + 2) % 8], color)
        cv2.putText(image, annotation, (line[1][0], line[1][1]), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        cv2.circle(image, (center[0][0], center[0][1]), 1, (255, 255, 0), 2)
        cv2.putText(image, depth, (center[0][0]+10, center[0][1]+10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 2)
        cv2.putText(image, str(occlude), (line[0][0], line[1][1]), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 2)
        return image

