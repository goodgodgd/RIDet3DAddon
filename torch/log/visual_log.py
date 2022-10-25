import glob
import os
import os.path as op
import numpy as np
import cv2

import RIDet3DAddon.torch.config as cfg3d
from RIDet3DAddon.torch.log.metric import split_true_false, split_tp_fp_fn_3d
import utils.framework.util_function as uf
from dataloader.readers.kitti_reader import SensorConfig
import config as cfg


class VisualLog:
    def merge_scale_hwa(self, features):
        stacked_feat = {}
        slice_keys = list(features.keys())  # ['yxhw', 'object', 'category']
        for key in slice_keys:
            if key == "whole":
                continue
            # list of (batch, HWA in scale, dim)
            # scaled_preds = [features[scale_name][key] for scale_name in range(len(features))]
            scaled_preds = np.concatenate(features[key], axis=1)  # (batch, N, dim)
            stacked_feat[key] = scaled_preds
        return stacked_feat


class VisualLog2d(VisualLog):
    def __init__(self, ckpt_path, epoch):
        self.vlog_path = op.join(ckpt_path, "vlog2d", f"ep{epoch:02d}")
        self.visual_heatmap_path = op.join(ckpt_path, "heatmap", f"ep{epoch:02d}") if cfg3d.Log.VISUAL_HEATMAP else None
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)
        if not op.isdir(self.visual_heatmap_path):
            os.makedirs(self.visual_heatmap_path)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg3d.Dataloader.CATEGORY_NAMES["category"])}

    def __call__(self, step, grtr, pred):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        """
        self.visual_2d(step, grtr, pred)

    def visual_2d(self, step, grtr, pred):
        splits = split_true_false(grtr["inst2d"], pred["inst2d"], cfg3d.Validation.TP_IOU_THRESH)
        batch = splits["grtr_tp"]["yxhw"].shape[0]

        for i in range(batch):
            # grtr_log_keys = ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
            image_grtr = grtr["image"][i].astype(np.uint8)
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_tp"], i, (0, 255, 0))
            image_grtr = self.draw_boxes(image_grtr, splits["grtr_fn"], i, (0, 0, 255))

            image_pred = grtr["image"][i].astype(np.uint8)
            image_pred = self.draw_boxes(image_pred, splits["pred_tp"], i, (0, 255, 0))
            image_pred = self.draw_boxes(image_pred, splits["pred_fp"], i, (0, 0, 255))

            if self.visual_heatmap_path:
                image_zero = np.zeros((512, 1280, 3)).astype(np.uint8)
                self.draw_box_heatmap(grtr, pred, image_zero, i, step, batch)

            vlog_image = np.concatenate([image_pred, image_grtr], axis=0)
            if step % 50 == 10:
                cv2.imshow("detection_result", vlog_image)
                cv2.waitKey(10)
            filename = op.join(self.vlog_path, f"{step * batch + i:05d}.jpg")
            cv2.imwrite(filename, vlog_image)


    def draw_boxes(self, image, bboxes, frame_idx, color, intrinsic=None):
        """
        all input arguments are numpy arrays
        :param image: (H, W, 3)
        :param bboxes: {'yxhw': (B, N, 4), 'category': (B, N, 1), ...}
        :param frame_idx
        :param log_keys
        :param color: box color
        :return: box drawn image
        """
        # cv2.imshow("test", image)
        # cv2.waitKey(500)
        height, width = image.shape[:2]
        box_yxhw = bboxes["yxhw"][frame_idx]  # (N, 4)
        category = bboxes["category"][frame_idx]  # (N, 1)
        valid_mask = box_yxhw[:, 2] > 0  # (N,) h>0

        box_yxhw = box_yxhw[valid_mask, :] * np.array([[height, width, height, width]], dtype=np.float32)
        box_tlbr = uf.convert_box_format_yxhw_to_tlbr(box_yxhw)  # (N', 4)
        category = category[valid_mask, 0].astype(np.int32)  # (N',)

        for i in range(box_yxhw.shape[0]):
            y1, x1, y2, x2 = box_tlbr[i].astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            annotation = "dontcare" if category[i] < 0 else f"{self.categories[category[i]]}"
            cv2.putText(image, annotation, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)

        return image

    def draw_box_heatmap(self, grtr, pred, image_zero, i, step, batch):
        box_heatmap = list()
        for scale in range(len(cfg.ModelOutput.FEATURE_SCALES)):
            feat_shape = pred["feat2d"]["whole"][scale].shape[-2:]
            v_objectness = self.draw_object(image_zero,
                                            grtr["feat2d"]["object"][scale],
                                            pred["feat2d"]["object"][scale],
                                            i, feat_shape)
            box_heatmap.append(v_objectness)
        box_heatmap = np.concatenate(box_heatmap, axis=1)
        filename = op.join(self.visual_heatmap_path, f"{step * batch + i:05d}_box.jpg")
        cv2.imwrite(filename, box_heatmap)

    def draw_object(self, bev_img, gt_object_feature, pred_objectness_feat, batch_idx, feat_shape):
        # object
        gt_obj_imgs = []
        pred_obj_imgs = []

        org_img = bev_img.copy()
        gt_object_per_image = self.convert_img(gt_object_feature[batch_idx], feat_shape, org_img)
        pred_object_per_image = self.convert_img(pred_objectness_feat[batch_idx], feat_shape, org_img)
        gt_obj_imgs.append(gt_object_per_image)
        pred_obj_imgs.append(pred_object_per_image)
        gt_obj_img = np.concatenate(gt_obj_imgs, axis=1)
        pred_obj_img = np.concatenate(pred_obj_imgs, axis=1)
        obj_img = np.concatenate([pred_obj_img, gt_obj_img], axis=0)
        return obj_img

    def convert_img(self, feature, feat_shape, org_img):
        feature_image = feature.reshape(feat_shape) * 255
        if feature_image.shape[-1] != 3:
            feature_image = cv2.cvtColor(feature_image, cv2.COLOR_GRAY2BGR)
        feature_image = cv2.resize(feature_image, (1280, 512), interpolation=cv2.INTER_NEAREST)
        feature_image = org_img + feature_image
        feature_image[-1, :] = [255, 255, 255]
        feature_image[:, -1] = [255, 255, 255]
        return feature_image


class VisualLog3d(VisualLog):
    def __init__(self, ckpt_path, epoch):
        self.vlog_path = op.join(ckpt_path, "vlog3d", f"ep{epoch:02d}")
        self.bev_view_path = op.join(ckpt_path, "vlog3d", "bev", f"ep{epoch:02d}")
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)
        if not op.isdir(self.bev_view_path):
            os.makedirs(self.bev_view_path)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg3d.Dataloader.CATEGORY_NAMES["category"])}
        self.orgin_path = cfg3d.Datasets.DATASET_CONFIG.ORIGIN_PATH

    def __call__(self, step, grtr, pred):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        """
        self.visual_3d(step, grtr, pred)

    def visual_3d(self, step, grtr, pred):
        splits = split_tp_fp_fn_3d(grtr['inst3d'], pred["inst3d"], grtr['inst2d']["object"], cfg3d.Validation.TP_IOU_THRESH)
        batch = splits["grtr_tp"]["xyz"].shape[0]

        for i in range(batch):
            # grtr_log_keys = ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
            org_image, intrinsic = self.get_origin_image(grtr["frame_names"][i])
            image_bev = grtr["image"][i].astype(np.uint8)
            image_grtr = org_image.copy()
            image_grtr, bev_image = self.draw_boxes(image_grtr, image_bev, splits["grtr_tp"], i, (0, 255, 0), (255, 0, 0),
                                         intrinsic)
            image_grtr, bev_image = self.draw_boxes(image_grtr, image_bev, splits["grtr_fn"], i, (0, 0, 255), (255, 0, 0),
                                         intrinsic)

            image_pred = org_image.copy()
            image_pred, bev_image = self.draw_boxes(image_pred, image_bev, splits["pred_tp"], i, (0, 255, 0), (0, 0, 255),
                                         intrinsic)
            image_pred, bev_image = self.draw_boxes(image_pred, image_bev, splits["pred_fp"], i, (0, 0, 255), (0, 0, 255),
                                         intrinsic)



            vlog_image = np.concatenate([image_pred, image_grtr], axis=0)
            if step % 50 == 10:
                cv2.imshow("detection_result", vlog_image)
                cv2.waitKey(10)
            filename = op.join(self.vlog_path, f"{step * batch + i:05d}.jpg")
            bev_filename = op.join(self.bev_view_path, f"{step * batch + i:05d}.jpg")
            cv2.imwrite(filename, vlog_image)
            cv2.imwrite(bev_filename, bev_image)

    def get_origin_image(self, frame_name):
        frame_num = frame_name.split("/")[-1]
        origin_file = glob.glob(os.path.join(self.orgin_path, f"*/{frame_num}"))[0]
        calib_file = origin_file.replace("image_2", "calib").replace("png","txt")
        origin_image = cv2.imread(origin_file)
        P2 = SensorConfig(calib_file).P

        divider = np.array([[origin_image.shape[1]], [origin_image.shape[0]], [1]])
        P2 /= divider
        return origin_image, P2

    def draw_boxes(self, image, bev_image, bboxes, frame_idx, p_fp_color, gt_pr_color, intrinsic):
        box_lwh = bboxes["lwh"][frame_idx]  # (N, 4)
        category = bboxes["category"][frame_idx]  # (N, 1)
        valid_mask = box_lwh[:, 2] > 0  # (N,) h>0
        iou = None
        if "iou" in bboxes.keys():
            iou = bboxes["iou"][frame_idx][valid_mask]
        box_3d, box_3d_center, bev_box = self.extract_corner(bboxes, intrinsic, frame_idx, valid_mask)
        category = category[valid_mask, 0].astype(np.int32)
        draw_image = self.draw_cuboid(image, box_3d, box_3d_center, category, bboxes["xyz"][frame_idx,:,2][valid_mask], p_fp_color)
        bev_view = self.draw_bev(bev_image, bev_box, category, gt_pr_color, iou)
        # cv2.imshow("test", draw_image)
        # cv2.waitKey(100)
        return draw_image, bev_view

    def extract_corner(self, bboxes, intrinsic, frame_idx, valid_mask):
        valid_xyz = bboxes["xyz"][frame_idx][valid_mask]
        valid_lwh = bboxes["lwh"][frame_idx][valid_mask]
        valid_theta = bboxes["theta"][frame_idx][valid_mask]
        box_3d = list()
        box_3d_center = list()
        bev_box = list()
        for n in range(valid_xyz.shape[0]):
            theta = valid_theta[n]
            z = valid_xyz[n, 0]
            y = - valid_xyz[n, 2]
            x = - valid_xyz[n, 1]

            h = valid_lwh[n, 2]
            w = valid_lwh[n, 1]
            l = valid_lwh[n, 0]
            center = valid_xyz[n]
            corner = list()
            bev_coner = list()
            for i in [1, -1]:
                for j in [1, -1]:
                    for k in [0, 1]:
                        point = np.copy(center)
                        point[0] = x + i * w / 2 * np.cos(-theta + np.pi / 2) + (j * i) * l / 2 * np.cos(-theta)
                        point[2] = z + i * w / 2 * np.sin(-theta + np.pi / 2) + (j * i) * l / 2 * np.sin(-theta)
                        point[1] = y - k * h
                        bev_coner.append(point)

                        point = np.append(point, 1)
                        point = np.dot(intrinsic, point)
                        point = point[:2] / point[2]
                        corner.append(point)
            bev_box.append(np.stack(bev_coner, axis=0))
            box_3d.append(corner)
            center_point = np.dot(intrinsic, np.append(center, 1))
            center_point = center_point[:2] / center_point[2]
            box_3d_center.append(center_point)
        return box_3d, box_3d_center, bev_box

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
            depth = f"{valid_depth[i]:.2f}"
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

    def draw_bev(self, image, bev_box, category, color, iou):
        """
        :param corner_3d: tilted_points(yxz)
        :return:
        """
        for i in range(len(bev_box)):
            image_x = (image.shape[1] / 2) - (-bev_box[i][:, 0] / 0.05)
            image_y = (image.shape[0]) - (bev_box[i][:, 2] / 0.05)
            pixels = np.stack([image_x, image_y], axis=1)
            xmin = np.min(pixels[:, 0]).astype(int)
            ymin = np.min(pixels[:, 1]).astype(int)
            xmax = np.max(pixels[:, 0]).astype(int)
            ymax = np.max(pixels[:, 1]).astype(int)
            image = self.draw_rotated_box(image, pixels, color)
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            annotation = "dontcare" if category[i] < 0 else f"{self.categories[category[i]]}"
            cv2.putText(image, annotation, (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
            if iou is not None:
                cv2.putText(image, f"{iou[i]:.3f}", (xmax, ymax), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return image

    def draw_rotated_box(self, img, corners, color):
        """
        corners :
        """
        corner_idxs = [(1, 2), (5, 6), (3, 4), (7, 1)]
        for corner_idx in corner_idxs:
            cv2.line(img,
                     (int(corners[corner_idx[0], 0]),
                      int(corners[corner_idx[0], 1])),
                     (int(corners[corner_idx[1], 0]),
                      int(corners[corner_idx[1], 1])),
                     color, 2)
        return img
