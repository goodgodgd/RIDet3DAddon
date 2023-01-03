import os
import os.path as op
import numpy as np
import cv2

import RIDet3DAddon.config as cfg
from RIDet3DAddon.tflow.log.metric import split_true_false, split_tp_fp_fn_3d
import utils.tflow.util_function as uf


class VisualLog:
    def __init__(self, ckpt_path, epoch):
        self.grtr_log_keys = cfg.Train.LOG_KEYS
        self.pred_log_keys = cfg.Train.LOG_KEYS
        self.vlog_path = op.join(ckpt_path, "vlog2d", f"ep{epoch:02d}")
        self.visual_heatmap_path = op.join(ckpt_path, "heatmap", f"ep{epoch:02d}") if cfg.Log.VISUAL_HEATMAP else None
        self.front_view_path = op.join(ckpt_path, "vlog3d", "front", f"ep{epoch:02d}")
        self.bev_view_path = op.join(ckpt_path, "vlog3d", "bev", f"ep{epoch:02d}")
        if not op.isdir(self.front_view_path):
            os.makedirs(self.front_view_path)
            os.makedirs(self.bev_view_path)
        if not op.isdir(self.vlog_path):
            os.makedirs(self.vlog_path)
        if not op.isdir(self.visual_heatmap_path):
            os.makedirs(self.visual_heatmap_path)
        self.categories = {i: ctgr_name for i, ctgr_name in enumerate(cfg.Dataloader.CATEGORY_NAMES["category"])}

    def __call__(self, step, grtr, pred, splits):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'bboxes': {'yxhw': (B,N,4), ...}, 'feature_l': {'yxhw': (B,HWA,4), ...}, ...}
        """
        self.visual(step, grtr, pred, splits)

    def visual(self, step, grtr, pred, splits):
        # splits = {}
        # grtr_bbox_augmented = self.exapand_grtr_bbox(grtr, pred)
        # splits["2d"] = split_true_false(grtr_bbox_augmented["inst2d"], pred["inst2d"], cfg.Validation.TP_IOU_THRESH)
        # splits["3d"] = split_tp_fp_fn_3d(grtr_bbox_augmented, pred, cfg.Validation.TP_IOU_THRESH)
        batch = splits["2d"]["grtr_tp"]["yxhw"].shape[0]

        for i in range(batch):
            # grtr_log_keys = ["pred_object", "pred_ctgr_prob", "pred_score", "distance"]
            image_grtr = uf.to_uint8_image(grtr["image"][i]).numpy()
            bev_image = np.zeros((800, 500, 3), dtype=np.uint8)
            self.draw_2d_boxes(image_grtr, splits["2d"], grtr, pred, i, step, batch)
            self.draw_3d_boxes(image_grtr, splits["3d"], bev_image, grtr, pred, i, step, batch)

    # def exapand_grtr_bbox(self, grtr, pred):
    #     grtr_boxes_3d = self.merge_scale_hwa(grtr["feat3d"])
    #     grtr_boxes_2d = self.merge_scale_hwa(grtr["feat2d"])
    #     pred_boxes_2d = self.merge_scale_hwa(pred["feat2d"])
    #
    #     best_probs = np.max(pred_boxes_2d["category"], axis=-1, keepdims=True)
    #     grtr_boxes_2d["pred_ctgr_prob"] = best_probs
    #     grtr_boxes_2d["pred_object"] = pred_boxes_2d["object"]
    #     grtr_boxes_2d["pred_score"] = best_probs * pred_boxes_2d["object"]
    #
    #     grtr_boxes_3d["pred_ctgr_prob"] = best_probs
    #     grtr_boxes_3d["pred_object"] = pred_boxes_2d["object"]
    #     grtr_boxes_3d["pred_score"] = best_probs * pred_boxes_2d["object"]
    #
    #     batch, _, __ = grtr["inst2d"]["yxhw"].shape
    #     # numbox = cfg.Validation.MAX_BOX
    #     numbox = grtr["inst2d"]["yxhw"].shape[1]
    #     objectness = grtr_boxes_2d["object"]
    #     for key in grtr_boxes_3d:
    #         features = []
    #         for frame_idx in range(batch):
    #             valid_mask = objectness[frame_idx, :, 0].astype(np.bool)
    #             feature = grtr_boxes_3d[key]
    #             feature = feature[frame_idx, valid_mask, :]
    #             feature = np.pad(feature, [(0, numbox - feature.shape[0]), (0, 0)])
    #             features.append(feature)
    #
    #         features = np.stack(features, axis=0)
    #         grtr_boxes_3d[key] = features
    #
    #     for key in grtr_boxes_2d:
    #         features = []
    #         for frame_idx in range(batch):
    #             valid_mask = objectness[frame_idx, :, 0].astype(np.bool)
    #             feature = grtr_boxes_2d[key]
    #             feature = feature[frame_idx, valid_mask, :]
    #             feature = np.pad(feature, [(0, numbox - feature.shape[0]), (0, 0)])
    #             features.append(feature)
    #
    #         features = np.stack(features, axis=0)
    #         grtr_boxes_2d[key] = features
    #     return {"inst2d": grtr_boxes_2d, "inst3d": grtr_boxes_3d}
    #
    # def merge_scale_hwa(self, features):
    #     stacked_feat = {}
    #     slice_keys = list(features.keys())  # ['yxhw', 'object', 'category']
    #     for key in slice_keys:
    #         if key == "merged":
    #             continue
    #         # list of (batch, HWA in scale, dim)
    #         # scaled_preds = [features[scale_name][key] for scale_name in range(len(features))]
    #         scaled_preds = np.concatenate(features[key], axis=1)  # (batch, N, dim)
    #         stacked_feat[key] = scaled_preds
    #     return stacked_feat

    def draw_2d_boxes(self, image, splits, grtr, pred, i, step, batch):
        image_grtr = image.copy()
        image_grtr = self.imple_2d_draw_box(image_grtr, splits["grtr_tp"], i, self.grtr_log_keys, (0, 255, 0))
        image_grtr = self.imple_2d_draw_box(image_grtr, splits["grtr_fn"], i, self.grtr_log_keys, (0, 0, 255))

        image_pred = image.copy()
        image_pred = self.imple_2d_draw_box(image_pred, splits["pred_tp"], i, [], (0, 255, 0))
        cv2.imshow("test", image_pred)
        image_pred = self.imple_2d_draw_box(image_pred, splits["pred_fp"], i, [], (0, 0, 255))
        cv2.imshow("fp", image_pred)

        vlog_image = np.concatenate([image_pred, image_grtr], axis=0)
        if self.visual_heatmap_path:
            image_zero = np.zeros_like(image[..., :-1])
            self.draw_box_heatmap(grtr, pred, image_zero, i, step, batch)
        if step % 50 == 10:
            cv2.imshow("detection_result", vlog_image)
            cv2.waitKey(10)
        filename = op.join(self.vlog_path, f"{step * batch + i:05d}.jpg")
        cv2.imwrite(filename, vlog_image)

    def draw_3d_boxes(self, image, splits, bev_image, grtr, pred, i, step, batch):
        image_grtr = image.copy()
        image_grtr, bev_image = self.imple_3d_draw_boxes(image_grtr, bev_image, splits["grtr_tp"], i,
                                                         self.grtr_log_keys, (0, 255, 0), (255, 0, 0),
                                                         grtr["intrinsic"])
        image_grtr, bev_image = self.imple_3d_draw_boxes(image_grtr, bev_image, splits["grtr_fn"], i,
                                                         self.grtr_log_keys, (0, 0, 255), (255, 0, 0),
                                                         grtr["intrinsic"])
        image_pred = image.copy()
        image_pred, bev_image = self.imple_3d_draw_boxes(image_pred, bev_image, splits["pred_tp"], i, [], (0, 255, 0),
                                                         (0, 0, 255), grtr["intrinsic"])
        image_pred, bev_image = self.imple_3d_draw_boxes(image_pred, bev_image, splits["pred_fp"], i, [], (0, 0, 255),
                                                         (0, 0, 255), grtr["intrinsic"])
        front_view = np.concatenate([image_pred, image_grtr], axis=0)
        if step % 50 == 10:
            cv2.imshow("front_view_result", front_view)
            cv2.imshow("bev_view_result", bev_image)
            cv2.waitKey(10)
        front_filename = op.join(self.front_view_path, f"{step * batch + i:05d}.jpg")
        bev_filename = op.join(self.bev_view_path, f"{step * batch + i:05d}.jpg")
        cv2.imwrite(front_filename, front_view)
        cv2.imwrite(bev_filename, bev_image)

    def imple_2d_draw_box(self, image, bboxes, frame_idx, log_keys, color):
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

    def draw_box_heatmap(self, grtr, pred, image_zero, i, step, batch):
        box_heatmap = list()
        for scale in range(len(cfg.ModelOutput.FEATURE_SCALES)):
            feat_shape = pred["feat2d"]["merged"][scale].shape[1:4]
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
        if feature_image.shape[-1] == 1:
            feature_image = cv2.cvtColor(feature_image, cv2.COLOR_GRAY2BGR)
        feature_image = cv2.resize(feature_image, org_img.shape[::-1][1:], interpolation=cv2.INTER_NEAREST)
        feature_image = org_img + feature_image
        feature_image[-1, :] = [255, 255, 255]
        feature_image[:, -1] = [255, 255, 255]
        return feature_image

    def imple_3d_draw_boxes(self, image, bev_image, bboxes, frame_idx, log_keys, tp_fp_color, gt_pr_color, intrinsic):
        valid_mask = bboxes["hwl"][frame_idx][:, 0] > 0  # (N,) h>0
        box_ctgr = bboxes["category"][frame_idx][valid_mask, 0].astype(np.int32)  # (N, 1)
        occluded = bboxes["occluded"][frame_idx][valid_mask, 0].astype(np.int32)  # (N, 1)
        yx = bboxes["yx"][frame_idx][valid_mask]
        z = bboxes["z"][frame_idx][valid_mask]
        hwl = bboxes["hwl"][frame_idx][valid_mask]
        theta = bboxes["theta"][frame_idx][valid_mask]
        iou = None
        if "iou" in bboxes.keys():
            iou = bboxes["iou"][frame_idx][valid_mask]

        # box_ctgr = box_ctgr[valid_mask, 0].astype(np.int32)
        proj_box, proj_box_center, bev_box = self.extract_corner(intrinsic[frame_idx], yx, z, hwl, theta)

        front_view = self.draw_cuboid(image, proj_box, proj_box_center, box_ctgr, occluded, z, tp_fp_color)

        bev_view = self.draw_bev(bev_image, bev_box, box_ctgr, gt_pr_color, iou)

        return front_view, bev_view

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
            annotation = "dontcare" if category[i] < 0 else f"{self.categories[category[i]]}"
            occlude = occluded[i]
            depth = f"{z[..., 0][i]:.2f}"
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

    def draw_bev(self, image, bev_box, category, color, iou):
        """
        :param corner_3d: tilted_points(yxz)
        :return:
        """
        for i in range(len(bev_box)):
            # image_x = (image.shape[1] / 2) - (-bev_box[i][:, 0] /0.05)
            # image_y = (image.shape[0]) - (bev_box[i][:, 1] /0.05)
            image_x = (image.shape[1] / 2) - (-bev_box[i][:, 0] / 0.1)
            image_y = (image.shape[0]) - (bev_box[i][:, 2] / 0.1)
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
                cv2.putText(image, f"{iou[i][0]:.3f}", (xmax, ymax), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
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

    # def draw_rotated_box(self, img, corners, color):
    #     """
    #     corners :
    #     """
    #     corner_idxs = [(0, 1), (4, 5), (0, 4), (1, 5)]
    #     for corner_idx in corner_idxs:
    #         cv2.line(img,
    #                  (int(corners[corner_idx[0], 0]),
    #                   int(corners[corner_idx[0], 1])),
    #                  (int(corners[corner_idx[1], 0]),
    #                   int(corners[corner_idx[1], 1])),
    #                  color, 2)
    #     return img
