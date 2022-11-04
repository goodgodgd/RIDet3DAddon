import os
import os.path as op
import numpy as np
import cv2

import RIDet3DAddon.config as cfg
from RIDet3DAddon.tflow.log.metric import split_true_false, split_tp_fp_fn_3d
import utils.tflow.util_function as uf


class VisualLog:
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

    def draw_boxes(self, image, bboxes, frame_idx, log_keys, color):
        raise NotImplementedError()


class VisualLog2d(VisualLog):
    def __init__(self, ckpt_path, epoch):
        self.grtr_log_keys = cfg.Train.LOG_KEYS
        self.pred_log_keys = cfg.Train.LOG_KEYS
        self.vlog_path = op.join(ckpt_path, "vlog2d", f"ep{epoch:02d}")
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["category"])}

    def __call__(self, step, grtr, pred):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        """
        self.visual_2d(step, grtr, pred)

    def visual_2d(self, step, grtr, pred):
        grtr_bbox_augmented = self.exapand_grtr_bbox(grtr, pred)
        splits = split_true_false(grtr_bbox_augmented, pred["inst2d"], cfg.Validation.TP_IOU_THRESH)
        batch = splits["grtr_tp"]["yxhw"].shape[0]

        for i in range(batch):
            # grtr_log_keys = ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
            image_grtr = uf.to_uint8_image(grtr["image"][i]).numpy()
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_tp"], i, self.grtr_log_keys, (0, 255, 0))
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_fn"], i, self.grtr_log_keys, (0, 0, 255))

            image_pred = uf.to_uint8_image(grtr["image"][i]).numpy()
            image_pred = self.draw_boxes(image_pred, splits["pred_tp"], i, [], (0, 255, 0))
            image_pred = self.draw_boxes(image_pred, splits["pred_fp"], i, [], (0, 0, 255))

            vlog_image = np.concatenate([image_pred, image_grtr], axis=0)
            if step % 50 == 10:
                cv2.imshow("detection_result", vlog_image)
                cv2.waitKey(10)
            filename = op.join(self.vlog_path, f"{step * batch + i:05d}.jpg")
            cv2.imwrite(filename, vlog_image)

    def exapand_grtr_bbox(self, grtr, pred):
        grtr_boxes = self.merge_scale_hwa(grtr["feat2d"])
        pred_boxes = self.merge_scale_hwa(pred["feat2d"])

        best_probs = np.max(pred_boxes["category"], axis=-1, keepdims=True)
        grtr_boxes["pred_ctgr_prob"] = best_probs
        grtr_boxes["pred_object"] = pred_boxes["object"]
        grtr_boxes["pred_score"] = best_probs * pred_boxes["object"]

        batch, _, __ = grtr["inst2d"]["yxhw"].shape
        numbox = cfg.Validation.MAX_BOX
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

    def draw_boxes(self, image, bboxes, frame_idx, log_keys, color):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param frame_idx
        :param log_keys
        :param color: box color
        :return: box drawn image
        """
        height, width = image.shape[:2]
        box_yxhw = bboxes["yxhw"][frame_idx]  # (N, 4)
        category = bboxes["category"][frame_idx]  # (N, 1)
        valid_mask = box_yxhw[:, 2] > 0  # (N,) h>0

        box_yxhw = box_yxhw[valid_mask, :] * np.array([[height, width, height, width]], dtype=np.float32)
        box_tlbr = uf.convert_box_format_yxhw_to_tlbr(box_yxhw)  # (N', 4)
        category = category[valid_mask, 0].astype(np.int32)  # (N',)

        valid_boxes = {}
        for key in log_keys:
            feature = (bboxes[key][frame_idx] * 100)
            feature = feature.astype(np.int32)
            valid_boxes[key] = feature[valid_mask, 0]

        for i in range(box_yxhw.shape[0]):
            y1, x1, y2, x2 = box_tlbr[i].astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            annotation = "dontcare" if category[i] < 0 else f"{self.categories[category[i]]}"
            for key in log_keys:
                annotation += f",{valid_boxes[key][i]:02d}" if key != "distance" else f",{valid_boxes[key][i]:.2f}"

            cv2.putText(image, annotation, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image


class VisualLog3d(VisualLog):
    def __init__(self, ckpt_path, epoch):
        self.grtr_log_keys = cfg.Train.LOG_KEYS
        self.pred_log_keys = cfg.Train.LOG_KEYS
        self.vlog_path = op.join(ckpt_path, "vlog3d", f"ep{epoch:02d}")
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["category"])}

    def __call__(self, step, grtr, pred):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        """
        self.visual_3d(step, grtr, pred)

    def visual_3d(self, step, grtr, pred):
        grtr_bbox_augmented_3d = self.exapand_grtr_bbox_3d(grtr, pred)
        grtr_bbox_augmented_2d = self.exapand_grtr_bbox_2d(grtr, pred)
        splits = split_tp_fp_fn_3d(grtr_bbox_augmented_3d, pred["inst3d"], grtr_bbox_augmented_2d["object"], cfg.Validation.TP_IOU_THRESH)
        batch = splits["grtr_tp"]["yxhwl"].shape[0]

        for i in range(batch):
            # grtr_log_keys = ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
            image_grtr = uf.to_uint8_image(grtr["image"][i]).numpy()
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_tp"], i, self.grtr_log_keys, (0, 255, 0), grtr["intrinsic"])
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_fn"], i, self.grtr_log_keys, (0, 0, 255), grtr["intrinsic"])

            image_pred = uf.to_uint8_image(grtr["image"][i]).numpy()
            image_pred = self.draw_boxes(image_pred, splits["pred_tp"], i, [], (0, 255, 0), grtr["intrinsic"])
            image_pred = self.draw_boxes(image_pred, splits["pred_fp"], i, [], (0, 0, 255), grtr["intrinsic"])

            vlog_image = np.concatenate([image_pred, image_grtr], axis=0)
            if step % 50 == 10:
                cv2.imshow("detection_result", vlog_image)
                cv2.waitKey(10)
            filename = op.join(self.vlog_path, f"{step * batch + i:05d}.jpg")
            cv2.imwrite(filename, vlog_image)

    def exapand_grtr_bbox_3d(self, grtr, pred):
        grtr_boxes_3d = self.merge_scale_hwa(grtr["feat3d"])
        grtr_boxes_2d = self.merge_scale_hwa(grtr["feat2d"])
        pred_boxes_3d = self.merge_scale_hwa(pred["feat3d"])
        pred_boxes_2d = self.merge_scale_hwa(pred["feat2d"])

        best_probs = np.max(pred_boxes_3d["category"], axis=-1, keepdims=True)
        grtr_boxes_3d["pred_ctgr_prob"] = best_probs
        grtr_boxes_3d["pred_object"] = pred_boxes_2d["object"]
        grtr_boxes_3d["pred_score"] = best_probs * pred_boxes_2d["object"]

        batch, _, __ = grtr["inst2d"]["yxhw"].shape
        numbox = cfg.Validation.MAX_BOX
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

        return grtr_boxes_3d

    def exapand_grtr_bbox_2d(self, grtr, pred):
        grtr_boxes = self.merge_scale_hwa(grtr["feat2d"])
        pred_boxes = self.merge_scale_hwa(pred["feat2d"])

        best_probs = np.max(pred_boxes["category"], axis=-1, keepdims=True)
        grtr_boxes["pred_ctgr_prob"] = best_probs
        grtr_boxes["pred_object"] = pred_boxes["object"]
        grtr_boxes["pred_score"] = best_probs * pred_boxes["object"]

        batch, _, __ = grtr["inst2d"]["yxhw"].shape
        numbox = cfg.Validation.MAX_BOX
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

    def draw_boxes(self, image, bboxes, frame_idx, color, intrinsic):
        height, width = image.shape[:2]
        box_yxhwl = bboxes["yxhwl"][frame_idx]  # (N, 4)
        category = bboxes["category"][frame_idx]  # (N, 1)
        valid_mask = box_yxhwl[:, 2] > 0  # (N,) h>0
        box_3d, box_3d_center = self.extract_corner(bboxes, intrinsic[frame_idx], frame_idx, valid_mask)
        category = category[valid_mask, 0].astype(np.int32)
        draw_image = self.draw_cuboid(image, box_3d, box_3d_center, category, bboxes["z"][frame_idx][valid_mask], color)
        # cv2.imshow("test", draw_image)
        # cv2.waitKey(100)
        return draw_image

    def extract_corner(self, bboxes, intrinsic, frame_idx, valid_mask):
        valid_yxhwl = bboxes["yxhwl"][frame_idx][valid_mask]
        valid_z = bboxes["z"][frame_idx][valid_mask]
        valid_theta = bboxes["theta"][frame_idx][valid_mask]
        box_3d = list()
        box_3d_center = list()
        for i in range(valid_yxhwl.shape[0]):
            z = valid_z[i]
            theta = valid_theta[i]
            y = valid_yxhwl[i, 0]
            x = valid_yxhwl[i, 1]
            h = valid_yxhwl[i, 2]
            w = valid_yxhwl[i, 3]
            l = valid_yxhwl[i, 4]
            center = np.asarray([x, y, z], dtype=np.float32)
            corner = list()
            for i in [1, -1]:
                for j in [1, -1]:
                    for k in [0, 1]:
                        point = np.copy(center)
                        point[0] = x + i * w / 2 * np.cos(-theta + np.pi / 2) + (j * i) * l / 2 * np.cos(-theta)
                        point[2] = z + i * w / 2 * np.sin(-theta + np.pi / 2) + (j * i) * l / 2 * np.sin(-theta)
                        point[1] = y - k * h

                        point = np.append(point, 1)
                        point = np.dot(intrinsic, point)
                        point = point[:2] / point[2]
                        corner.append(point)
            box_3d.append(corner)
            center_point = np.dot(intrinsic, np.append(center, 1))
            center_point = center_point[:2] / center_point[2]
            box_3d_center.append(center_point)
        return box_3d, box_3d_center

    def draw_cuboid(self, image, bbox, bbox_center, category, valid_depth, color):
        image = image.copy()
        bbox = np.array(bbox)
        bbox = (np.reshape(bbox, (-1, 8, 2)) * image.shape[::-1][1:3]).astype(np.int32)
        bbox = bbox.tolist()

        centers = np.array(bbox_center)
        centers = (np.reshape(centers, (-1, 1, 2)) * image.shape[::-1][1:3]).astype(np.int32)
        centers = centers.tolist()

        for i, (point, center) in enumerate(zip(bbox, centers)):
            annotation = "dontcare" if category[i] < 0 else f"{self.categories[category[i]]}"
            depth = f"{valid_depth[i][0]:.2f}"
            image = self.draw_line(image, point, center, color, annotation, depth)
        return image

    def draw_line(self, image, line, center, color, annotation, depth):
        for i in range(4):
            image = cv2.line(image, line[i * 2], line[i * 2 + 1], color)
        for i in range(8):
            image = cv2.line(image, line[i], line[(i + 2) % 8], color)
        cv2.putText(image, annotation, (line[0][0], line[0][1]), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        cv2.circle(image, (center[0][0], center[0][1]), 1, (255, 255, 0), 2)
        cv2.putText(image, depth, (center[0][0]+10, center[0][1]+10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 2)
        return image
