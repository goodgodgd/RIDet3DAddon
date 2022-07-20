import numpy as np

import config as cfg
import utils.tflow.util_function as uf
import RIDet3DAddon.tflow.utils.util_function as uf3d


class FeatureMapDistributer:
    def __init__(self, ditrib_policy, anchors_per_scale):
        self.ditrib_policy = eval(ditrib_policy)(anchors_per_scale)

    def create(self, bboxes2d, bboxes3d, feat_sizes):
        bbox2d_map, bbox3d_map, bbox2d_logit, bbox3d_logit = self.ditrib_policy(bboxes2d, bboxes3d, feat_sizes)
        return bbox2d_map, bbox3d_map, bbox2d_logit, bbox3d_logit


class ObjectDistribPolicy:
    def __init__(self, anchors_per_scale):
        self.feat_order = cfg.ModelOutput.FEATURE_SCALES
        self.anchor_ratio = np.concatenate([anchor for anchor in anchors_per_scale])
        self.num_anchor = len(self.anchor_ratio) // len(self.feat_order)


class SinglePositivePolicy(ObjectDistribPolicy):
    def __init__(self, anchors_per_scale):
        super().__init__(anchors_per_scale)

    def __call__(self, bboxes, feat_sizes):
        """
        :param bboxes: bounding boxes in image ratio (0~1) [cy, cx, h, w, obj, major_category, minor_category, depth] (N, 8)
        :param anchors: anchors in image ratio (0~1) (9, 2)
        :param feat_sizes: feature map sizes for 3 feature maps
        :return:
        """

        boxes_hw = bboxes[:, np.newaxis, 2:4]  # (N, 1, 8)
        anchors_hw = self.anchor_ratio[np.newaxis, :, :]  # (1, 9, 2)
        inter_hw = np.minimum(boxes_hw, anchors_hw)  # (N, 9, 2)
        inter_area = inter_hw[:, :, 0] * inter_hw[:, :, 1]  # (N, 9)
        union_area = boxes_hw[:, :, 0] * boxes_hw[:, :, 1] + anchors_hw[:, :, 0] * anchors_hw[:, :, 1] - inter_area
        iou = inter_area / union_area
        best_anchor_indices = np.argmax(iou, axis=1)

        gt_features = [np.zeros((feat_shape[0], feat_shape[1], self.num_anchor, bboxes.shape[-1]), dtype=np.float32)
                       for feat_shape in feat_sizes]

        # TODO split anchor indices by scales, create each feature map by single operation
        for anchor_index, bbox in zip(best_anchor_indices, bboxes):
            scale_index = anchor_index // self.num_anchor
            anchor_index_in_scale = anchor_index % self.num_anchor
            feat_map = gt_features[scale_index]
            # bbox: [y, x, h, w, category]
            grid_yx = (bbox[:2] * feat_sizes[scale_index]).astype(np.int32)
            assert (grid_yx >= 0).all() and (grid_yx < feat_sizes[scale_index]).all()
            # # bbox: [y, x, h, w, 1, major_category, minor_category, depth]
            feat_map[grid_yx[0], grid_yx[1], anchor_index_in_scale] = bbox
            gt_features[scale_index] = feat_map
        return gt_features


class MultiPositivePolicy(ObjectDistribPolicy):
    def __init__(self, anchors_per_scale):
        super().__init__(anchors_per_scale)
        self.iou_threshold = cfg.FeatureDistribPolicy.IOU_THRESH
        self.image_shape = cfg.Datasets.DATASET_CONFIGS.INPUT_RESOLUTION
        self.strides = cfg.ModelOutput.FEATURE_SCALES

# TODO 1. make anchor map 2. anchor and gt bbox match 3. iou threshold
    def __call__(self, bboxes, feat_sizes):

        anchor_map = self.make_anchor_map(feat_sizes)
        gt_features = self.make_bbox_map(bboxes, anchor_map, feat_sizes)
        return gt_features

    def make_anchor_map(self, feat_sizes):
        anchors = []
        for scale, (feat_size, stride) in enumerate(zip(feat_sizes, self.feat_order)):
            ry = (np.arange(0, feat_size[0]) + 0.5) * stride / self.image_shape[0]
            rx = (np.arange(0, feat_size[1]) + 0.5) * stride / self.image_shape[1]
            ry, rx = np.meshgrid(ry, rx)

            grid_map = np.vstack((ry.ravel(), rx.ravel(), np.zeros(feat_size[0] * feat_size[1]), np.zeros(feat_size[0] * feat_size[1]))).transpose()
            anchor_ratio = np.concatenate([np.zeros((3, 2)), self.anchor_ratio[scale * 3:(scale+1) * 3]], axis=1)
            anchor_map = (anchor_ratio.reshape((1, self.num_anchor, 4))
                          + grid_map.reshape((1, grid_map.shape[0], 4)).transpose((1, 0, 2)))
            anchor_map = anchor_map.reshape((self.num_anchor * grid_map.shape[0], 4))
            anchors.append(anchor_map)
        anchors = np.concatenate(anchors, axis=0)
        return anchors

    def make_bbox_map(self, bboxes, anchor_map, feat_sizes):

        bboxes_tlbr = uf.convert_box_format_yxhw_to_tlbr(bboxes)
        anchor_tlbr = uf.convert_box_format_yxhw_to_tlbr(anchor_map)
        bboxes_area = (bboxes_tlbr[:, 2] - bboxes_tlbr[:, 0]) * (bboxes_tlbr[:, 3] - bboxes_tlbr[:, 1])
        anchor_area = (anchor_tlbr[:, 2] - anchor_tlbr[:, 0]) * (anchor_tlbr[:, 3] - anchor_tlbr[:, 1])
        width_height = np.minimum(bboxes_tlbr[:, np.newaxis, 2:4], anchor_tlbr[np.newaxis, :, 2:]) - \
                       np.maximum(bboxes_tlbr[:, np.newaxis, :2], anchor_tlbr[np.newaxis, :, :2])
        width_height = np.clip(width_height, 0, 1)
        intersection = np.prod(width_height, axis=-1)
        iou = np.where(intersection > 0, intersection / (bboxes_area[:, np.newaxis] + anchor_area[np.newaxis, :] - intersection), np.zeros(1))
        max_iou = np.amax(iou, axis=0)
        max_gt_idx = np.argmax(iou, axis=1)
        max_idx = np.argmax(iou, axis=0)
        positive = max_iou > self.iou_threshold[0]
        negative = max_iou < self.iou_threshold[1]

        max_match = np.zeros((anchor_tlbr.shape[0], bboxes_tlbr.shape[-1]))
        max_match[max_gt_idx] = bboxes
        max_match = max_match * (~positive[:, np.newaxis])
        iou_match = bboxes[max_idx, ...] * positive[:, np.newaxis]
        gt_match = iou_match + max_match

        gt_features = []
        start_channel = 0
        for scale in feat_sizes:
            last_channel = start_channel + scale[0] * scale[1] * self.num_anchor
            gt_feat = gt_match[start_channel: last_channel, ...].astype(np.float32)
            start_channel = last_channel
            gt_features.append(gt_feat.reshape(scale[0], scale[1], self.num_anchor, -1))
        return gt_features


class OTAPolicy(ObjectDistribPolicy):
    def __init__(self, anchors_per_scale, center_radius=cfg.FeatureDistribPolicy.CENTER_RADIUS,
                 resolution=cfg.Datasets.DATASET_CONFIGS.INPUT_RESOLUTION):
        super().__init__(anchors_per_scale)
        self.center_radius = center_radius
        self.resolution = resolution

    def __call__(self, bboxes2d, bboxes3d, feat_sizes):
        valid_bboxes2d = bboxes2d[bboxes2d[..., 2] > 0]
        valid_bboxes3d = bboxes3d[bboxes2d[..., 2] > 0]
        logit_2d_box = self.de_sigmoid(valid_bboxes2d)
        logit_3d_box = self.de_sigmoid(valid_bboxes3d)
        obj_logit = self.de_sigmoid(0.8)
        bboxes_pixel = uf3d.convert_box_scale_01_to_pixel(valid_bboxes2d)
        under_sizes, upper_sizes = self.scale_limit()
        box_scales = self.find_box_scales(bboxes_pixel, under_sizes, upper_sizes)
        box_grid_coords = self.find_box_grid_coordinates(bboxes_pixel)
        gt_2d_features = [np.zeros((feat_shape[0], feat_shape[1], 1, bboxes2d.shape[-1]), dtype=np.float32)
                          for feat_shape in feat_sizes]
        gt_2d_logit = [np.zeros((feat_shape[0], feat_shape[1], 1, bboxes2d.shape[-1]), dtype=np.float32)
                       for feat_shape in feat_sizes]
        gt_3d_features = [np.zeros((feat_shape[0], feat_shape[1], 1, bboxes3d.shape[-1]), dtype=np.float32)
                          for feat_shape in feat_sizes]
        gt_3d_logit = [np.zeros((feat_shape[0], feat_shape[1], 1, bboxes3d.shape[-1]), dtype=np.float32)
                       for feat_shape in feat_sizes]

        for i, grid_size in enumerate(self.feat_order):
            feat2d_map = gt_2d_features[i]
            feat2d_logit = feat2d_map.copy()
            multi2d_map = feat2d_map.copy()
            multi2d_logit = feat2d_map.copy()

            feat3d_map = gt_3d_features[i]
            feat3d_logit = feat3d_map.copy()
            multi3d_map = feat3d_map.copy()
            multi3d_logit = feat3d_map.copy()

            box_scale = box_scales[..., i]
            box_grid_coord = box_grid_coords[..., i, :]
            valid_grid_yx = box_grid_coord[box_scale].astype(np.int)

            feat2d_map[valid_grid_yx[:, 0], valid_grid_yx[:, 1], ...] = valid_bboxes2d[box_scale][..., np.newaxis, :]
            feat2d_logit[valid_grid_yx[:, 0], valid_grid_yx[:, 1], ...] = logit_2d_box[box_scale][..., np.newaxis, :]

            feat3d_map[valid_grid_yx[:, 0], valid_grid_yx[:, 1], ...] = valid_bboxes3d[box_scale][..., np.newaxis, :]
            feat3d_logit[valid_grid_yx[:, 0], valid_grid_yx[:, 1], ...] = logit_3d_box[box_scale][..., np.newaxis, :]

            valid_mask = self.create_multi_mask(feat_sizes[i], valid_bboxes2d, box_scale)

            multi2d_map[..., 4] = multi2d_map[..., 4] + valid_mask[..., np.newaxis] * 0.8
            multi2d_logit[..., 4] = multi2d_logit[..., 4] + valid_mask[..., np.newaxis] * obj_logit
            multi2d_logit[valid_grid_yx[:, 0], valid_grid_yx[:, 1], ...] = logit_2d_box[box_scale][..., np.newaxis, :]
            multi3d_map[..., 7] = multi3d_map[..., 7] + valid_mask[..., np.newaxis] * 0.8
            multi3d_logit[..., 7] = multi3d_logit[..., 7] + valid_mask[..., np.newaxis] * obj_logit
            multi3d_logit[valid_grid_yx[:, 0], valid_grid_yx[:, 1], ...] = logit_3d_box[box_scale][..., np.newaxis, :]
            feat2d_map = np.maximum(multi2d_map, feat2d_map)
            feat3d_map = np.maximum(multi3d_map, feat3d_map)
            gt_2d_features[i] = feat2d_map
            gt_3d_features[i] = feat3d_map
            gt_2d_logit[i] = multi2d_logit
            gt_3d_logit[i] = multi3d_logit
        return gt_2d_features, gt_3d_features, gt_2d_logit, gt_3d_logit

    def find_box_scales(self, bboxes_pixel, under_sizes, upper_sizes):
        diag_length = np.linalg.norm(bboxes_pixel[..., 2:4], axis=-1)
        bbox_scales = list()
        for under, upper in zip(under_sizes, upper_sizes):
            bbox_scales.append(np.logical_and(diag_length > under, diag_length < upper))
        bbox_scales = np.stack(bbox_scales, axis=-1)
        for box_index in range(len(bbox_scales)):
            assert (np.any(bbox_scales[box_index])) or (not np.all(bbox_scales[box_index])), \
                f"to check box scales. {bbox_scales[box_index]}"
        return bbox_scales

    def find_box_grid_coordinates(self, bboxes_pixel):
        # [B, N, 1, C] / [3, 1] = [B, N, 3, C]
        grid_coord = bboxes_pixel[..., np.newaxis, :2] // np.array(self.feat_order)[..., np.newaxis]
        return grid_coord

    def scale_limit(self):
        under_sizes = list()
        upper_sizes = list()
        for feat_size in self.feat_order:
            under_sizes.append(feat_size * np.sqrt(2) / 2)
            upper_sizes.append(feat_size * np.sqrt(2) * 3)
        # no upper bound for large scale
        upper_sizes[-1] = 10000
        return under_sizes, upper_sizes

    def check_scale(self, bbox, under_sizes, upper_sizes):
        bbox_size = (bbox[2]**2 + bbox[3]**2)**(1/2)
        bbox_scales = list()
        if bbox_size < upper_sizes[0]:
            bbox_scales.append(0)
        if under_sizes[1] < bbox_size < upper_sizes[1]:
            bbox_scales.append(1)
        if under_sizes[2] < bbox_size:
            bbox_scales.append(2)
        return bbox_scales

    def create_multi_mask(self, feat_size, box_2d, box_scale):
        norm_radius = self.center_radius / feat_size
        rx = np.arange(0, feat_size[1])
        ry = np.arange(0, feat_size[0])
        x_grid, y_grid = np.meshgrid(rx, ry)
        y_grid_center = ((y_grid + 0.5) / feat_size[0])[np.newaxis, ...]
        x_grid_center = ((x_grid + 0.5) / feat_size[1])[np.newaxis, ...]
        valid_bbox_mask = self.bbox_mask(box_2d[box_scale], y_grid_center, x_grid_center, False)
        valid_center_mask = self.center_mask(box_2d[box_scale], y_grid_center, x_grid_center, norm_radius)
        valid_mask = valid_bbox_mask & valid_center_mask
        return valid_mask

    def bbox_mask(self, bbox, y_grid_center, x_grid_center, use_half):
        """
        :param bbox: instance GT bbox (n, c)
        :param y_grid_center: (num_gt, h*w)
        :param x_grid_center: (num_gt, h*w)
        :return: (num_box, num_anchor), (num_anchor)
        """
        if use_half:
            bbox[..., 2:4] = bbox[..., 2:4] * 0.5
        bbox_tlbr = uf.convert_box_format_yxhw_to_tlbr(bbox)

        delta_box_t = y_grid_center - bbox_tlbr[..., np.newaxis, :1]
        delta_box_l = x_grid_center - bbox_tlbr[..., np.newaxis, 1:2]
        delta_box_b = bbox_tlbr[..., np.newaxis, 2:3] - y_grid_center
        delta_box_r = bbox_tlbr[..., np.newaxis, 3:4] - x_grid_center
        # bbox_deltas: (num_box, num_anchor, 4)
        bbox_deltas = np.stack([delta_box_t, delta_box_l, delta_box_b, delta_box_r], axis=-1)
        in_valid_bboxes = np.min(bbox_deltas, axis=-1) > 0
        in_valid_all_bboxes = np.sum(in_valid_bboxes, axis=0) > 0
        return in_valid_all_bboxes

    def center_mask(self, bboxes, y_grid_center, x_grid_center, normalize_radius):
        """
        :param bboxes: instance GT bbox (num_gt, c)
        :param y_grid_center: (num_gt, h*w)
        :param x_grid_center: (num_gt, h*w)
        :param normalize_radius:
        :return: (num_box, num_anchor), (num_anchor)
        """
        center_t = bboxes[..., :1] - normalize_radius[0]
        center_l = bboxes[..., 1:2] - normalize_radius[1]
        center_b = bboxes[..., :1] + normalize_radius[0]
        center_r = bboxes[..., 1:2] + normalize_radius[1]

        delta_center_t = y_grid_center - center_t[..., np.newaxis, :]
        delta_center_l = x_grid_center - center_l[..., np.newaxis, :]
        delta_center_b = center_b[..., np.newaxis, :] - y_grid_center
        delta_center_r = center_r[..., np.newaxis, :] - x_grid_center
        # center_deltas: (num_gt, h*w, 4)
        center_deltas = np.stack([delta_center_t, delta_center_l, delta_center_b, delta_center_r], axis=-1)
        in_valid_center_bboxes = np.min(center_deltas, axis=-1) > 0
        in_valid_all_center = np.sum(in_valid_center_bboxes, axis=0) > 0
        return in_valid_all_center

    def de_sigmoid(self, x, eps=1e-7):
        x = np.maximum(x, eps)
        x = np.minimum(x, 1 / eps)
        x = x / (1 - x)
        x = np.log(x)
        return x
