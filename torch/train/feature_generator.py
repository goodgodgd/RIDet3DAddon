import numpy as np

import config as cfg
import utils.framework.util_function as uf
import RIDet3DAddon.torch.utils.util_function as uf3d
import RIDet3DAddon.torch.model.decoder_2d as decoder2d
import RIDet3DAddon.torch.model.decoder_3d as decoder3d
import RIDet3DAddon.torch.config as cfg3d


class FeatureMapDistributer:
    def __init__(self, ditrib_policy, imshape, anchors_per_scale):
        self.ditrib_policy = eval(ditrib_policy)(imshape, anchors_per_scale)

    def __call__(self, features):
        for key in ["inst2d", "inst3d"]:
            features[key] = uf3d.merge_and_slice_features(features[key], True, key)
        features = uf3d.convert_tensor_to_numpy(features)
        features = self.ditrib_policy(features)
        return features

    def slice_and_merge(self, features):
        slice_feat = uf.merge_and_slice_features(features, True, "feat")
        merge_features = {key: list() for key in slice_feat.keys()}
        for key, feat in slice_feat.items():
            if key is "merged":
                merge_features[key] = feat
                continue
            for dict_per_feat in feat:
                merge_features[key].append(uf.merge_dim_hwa(dict_per_feat))
        return merge_features


class ObjectDistribPolicy:
    def __init__(self, imshape, anchors_per_scale):
        self.feat_scales = cfg.ModelOutput.FEATURE_SCALES
        self.feat_shapes = [np.array(imshape[:2]) // scale for scale in self.feat_scales]
        self.anchor_ratio = np.concatenate([anchor for anchor in anchors_per_scale])
        self.num_anchor_per_scale = len(self.anchor_ratio) // len(self.feat_scales)

    def feature_merge(self, features):
        fkey = [key for key in features.keys() if key.startswith("feat")]
        for key in fkey:
            slice_feat = features[key]
            merge_features = {key: list() for key in slice_feat.keys()}
            for slice_key, feat in slice_feat.items():
                if key is "whole":
                    merge_features[slice_key] = feat
                    continue
                for dict_per_feat in feat:
                    merge_features[slice_key].append(uf.merge_dim_hwa(dict_per_feat))
            features[key] = merge_features
        return features

    def slice_and_merge(self, features):
        merge_features = {key: list() for key in features.keys() }
        for key, feat in features.items():
            if key is "whole":
                merge_features[key] = feat
                continue
            for dict_per_feat in feat:
                merge_features[key].append(uf.merge_dim_hwa(dict_per_feat))
        return merge_features


class SinglePositivePolicy(ObjectDistribPolicy):
    def __init__(self, imshape, anchors_per_scale):
        super().__init__(imshape, anchors_per_scale)
        self.decoder2d = decoder2d.FeatureDecoder(anchors_per_scale)
        self.decoder3d = decoder3d.FeatureDecoder3D(anchors_per_scale)

    def __call__(self, box_features):
        """
            :param features: {inst_box: {"yxhw": ..., "object": ..., ...}, inst_dc{...}, ...}
            :return:
            """
        bboxes = box_features["inst2d"]["yxhw"]
        boxes_hw = bboxes[..., np.newaxis, 2:4]  # (B, N, 1, 2)
        anchors_hw = self.anchor_ratio[np.newaxis, np.newaxis, :, :]  # (1, 1, 9, 2)
        inter_hw = np.minimum(boxes_hw, anchors_hw)  # (B, N, 9, 2)
        inter_area = inter_hw[..., 0] * inter_hw[..., 1]  # (B, N, 9)
        union_area = boxes_hw[..., 0] * boxes_hw[..., 1] + anchors_hw[..., 0] * anchors_hw[..., 1] - inter_area
        iou = inter_area / union_area
        best_anchor_indices = np.argmax(iou, axis=-1)
        out_2d_features = [np.zeros((bboxes.shape[0], box_features["inst2d"]["merged"].shape[-1] + 1,
                                     self.num_anchor_per_scale, feat_shape[0], feat_shape[1]), dtype=np.float32) for
                           feat_shape in self.feat_shapes]
        out_3d_features = [np.zeros((bboxes.shape[0], box_features["inst3d"]["merged"].shape[-1] + 1,
                                     self.num_anchor_per_scale, feat_shape[0], feat_shape[1]), dtype=np.float32) for
                           feat_shape in self.feat_shapes]
        batch_2d_features = [np.zeros((bboxes.shape[0], box_features["inst2d"]["merged"].shape[-1],
                                       self.num_anchor_per_scale, feat_shape[0], feat_shape[1]), dtype=np.float32) for
                             feat_shape in self.feat_shapes]
        batch_3d_features = [np.zeros((bboxes.shape[0], box_features["inst3d"]["merged"].shape[-1],
                                       self.num_anchor_per_scale, feat_shape[0], feat_shape[1]), dtype=np.float32) for
                             feat_shape in self.feat_shapes]
        anchor_map = [
            np.ones((bboxes.shape[0], 1, self.num_anchor_per_scale, feat_shape[0], feat_shape[1]), dtype=np.float32) * (
                -1) for feat_shape in self.feat_shapes]
        for batch in range(bboxes.shape[0]):
            for anchor_index, box2d, box3d in zip(best_anchor_indices[batch], box_features["inst2d"]["merged"][batch],
                                                  box_features["inst3d"]["merged"][batch]):
                if np.all(box2d == 0):
                    break
                scale_index = anchor_index // self.num_anchor_per_scale
                anchor_index_in_scale = anchor_index % self.num_anchor_per_scale
                feat_2d_map = batch_2d_features[scale_index]
                feat_3d_map = batch_3d_features[scale_index]
                anchor_scale_map = anchor_map[scale_index]
                # bbox: [y, x, h, w, category]
                grid_yx = (box2d[:2] * self.feat_shapes[scale_index]).astype(np.int32)
                assert (grid_yx >= 0).all() and (grid_yx < self.feat_shapes[scale_index]).all()
                # # bbox: [y, x, h, w, 1, major_category, minor_category, depth]
                feat_2d_map[batch, :, anchor_index_in_scale, grid_yx[0], grid_yx[1]] = box2d
                feat_3d_map[batch, :, anchor_index_in_scale, grid_yx[0], grid_yx[1]] = box3d
                anchor_scale_map[batch, 0, anchor_index_in_scale, grid_yx[0], grid_yx[1]] = anchor_index
                out_2d_features[scale_index] = np.concatenate([feat_2d_map, anchor_scale_map], axis=1)
                out_3d_features[scale_index] = np.concatenate([feat_3d_map, anchor_scale_map], axis=1)

        box_features["feat2d"] = out_2d_features
        box_features["feat3d"] = out_3d_features
        for key in ["feat2d", "feat3d"]:
            box_features[key] = uf3d.merge_and_slice_features(box_features[key], True, key)
        box_features["feat2d_logit"] = self.decoder2d.inverse(box_features["feat2d"])
        box_features["feat3d_logit"] = self.decoder3d.inverse(box_features["feat3d"], box_features["feat2d"]['yxhw'])
        box_features = self.feature_merge(box_features)
        return box_features


class MultiPositiveGenerator:
    def __init__(self, imshape, center_radius=cfg3d.FeatureDistribPolicy.CENTER_RADIUS,
                 multi_positive_weight=cfg3d.FeatureDistribPolicy.MULTI_POSITIVE_WIEGHT):
        self.feat_scales = cfg3d.ModelOutput.FEATURE_SCALES
        self.feat_shapes = [np.array(imshape[:2]) // scale for scale in self.feat_scales]
        self.box_size_range_per_scale = self.get_box_size_ranges()
        self.center_radius = center_radius
        self.multi_positive_weight = multi_positive_weight

    def get_box_size_ranges(self):
        size_ranges = []
        for feat_size in self.feat_scales:
            size_ranges.append([feat_size * np.sqrt(2) / 2, feat_size * np.sqrt(2) * 3])
        size_ranges = np.array(size_ranges)
        # no upper bound for large scale
        size_ranges[0, 0] = 0
        size_ranges[-1, 1] = 100000
        return size_ranges
    
    def __call__(self, features):
        inst2d = features["inst2d"]
        inst3d = features["inst3d"]
        box2d = features["inst2d"]["yxhw"]
        grid_yx, belong_to_scale = self.to_grid_over_scales(box2d)
        features["feat2d"] = self.create_featmap_single_positive(inst2d, grid_yx, belong_to_scale)
        features["feat3d"] = self.create_featmap_single_positive(inst3d, grid_yx, belong_to_scale)

        for key in ["feat2d", "feat3d"]:
            features[key] = uf3d.merge_and_slice_features(features[key], True, key)
        features["feat2d"]["mp_object"] = self.multi_positive_objectness(box2d, belong_to_scale,
                                                                         features["feat2d"]["object"])
        return features

    def to_grid_over_scales(self, box2d):
        grid_yx, belong_to_scale = [], []
        for i, s in enumerate(self.feat_scales):
            grid_yx_in_scale, belong_in_scale = self.to_grid(box2d, i)
            grid_yx.append(grid_yx_in_scale)
            belong_to_scale.append(belong_in_scale)
        return grid_yx, belong_to_scale

    def to_grid(self, box2d, scale_index):
        box2d_pixel = uf3d.convert_box_scale_01_to_pixel(box2d)
        diag_length = np.linalg.norm(box2d_pixel[..., 2:4], axis=-1)
        box_range = self.box_size_range_per_scale[scale_index]
        belong_to_scale = np.logical_and(diag_length > box_range[0], diag_length < box_range[1])
        box_grid_yx = box2d_pixel[..., :2] // np.array(self.feat_scales[scale_index])
        return box_grid_yx, belong_to_scale

    def create_featmap_single_positive(self, inst, grid_yx, belong_to_scale):
        feats_over_scale = []
        for i, s in enumerate(self.feat_scales):
            feats_in_batch = []
            for b in range(inst["merged"].shape[0]):
                # print("batch:", b, "scale:", s)
                feat_map = self.create_featmap(self.feat_shapes[i], inst["merged"][b], grid_yx[i][b], belong_to_scale[i][b])
                feats_in_batch.append(feat_map)
            feats_over_scale.append(np.stack(feats_in_batch, axis=0))
        return feats_over_scale

    def create_featmap(self, feat_shapes, instances, grid_yx, valid):

        valid_grid_yx = grid_yx[valid].astype(np.int)
        gt_features = np.zeros((instances.shape[-1], 1, feat_shapes[0], feat_shapes[1]), dtype=np.float32)
        instances = instances[valid]
        instances = np.transpose(instances)
        # print('valid_grid_yx', valid_grid_yx)
        gt_features[:, :, valid_grid_yx[:, 0], valid_grid_yx[:, 1]] = instances[:, np.newaxis, ...]
        return gt_features

    def multi_positive_objectness(self, box2d, validity, single_positive_map):
        mp_object_over_scale = []
        for i, s in enumerate(self.feat_scales):
            grid_center_yx = self.make_grid_center(self.feat_shapes[i])
            positive_tlbr_from_box = self.get_positive_range_from_box(box2d, validity[i])
            positive_tlbr_from_radius = self.get_positive_range_from_radius(box2d, self.center_radius, i, validity[i])
            object_from_box = self.get_positive_map_in_boxes(positive_tlbr_from_box, grid_center_yx)
            object_from_radius = self.get_positive_map_in_boxes(positive_tlbr_from_radius, grid_center_yx)
            mp_object = self.merge_positive_maps(single_positive_map[i], object_from_box, object_from_radius)
            mp_object_over_scale.append(mp_object)
        return mp_object_over_scale
    
    def make_grid_center(self, feat_map_size):
        rx = np.arange(0, feat_map_size[1])
        ry = np.arange(0, feat_map_size[0])
        x_grid, y_grid = np.meshgrid(rx, ry)
        y_grid_center = ((y_grid + 0.5) / feat_map_size[0])
        x_grid_center = ((x_grid + 0.5) / feat_map_size[1])
        # (h, w, 2)
        grid_center_yx = np.stack([y_grid_center, x_grid_center], axis=-1)
        return grid_center_yx
    
    def get_positive_range_from_box(self, box2d_in_scale, validity):
        half_box = np.concatenate([box2d_in_scale[..., :2], box2d_in_scale[..., 2:4] * 0.5], axis=-1)
        box_tlbr = uf.convert_box_format_yxhw_to_tlbr(half_box)
        box_tlbr *= validity[..., np.newaxis]
        return box_tlbr
    
    def get_positive_range_from_radius(self, box2d_in_scale, center_radius, scale_index, validity):
        norm_radius = np.array(center_radius) / self.feat_shapes[scale_index]
        positive_t = box2d_in_scale[..., :1] - norm_radius[0]
        positive_l = box2d_in_scale[..., 1:2] - norm_radius[1]
        positive_b = box2d_in_scale[..., :1] + norm_radius[0]
        positive_r = box2d_in_scale[..., 1:2] + norm_radius[1]
        positive_tlbr = np.concatenate([positive_t, positive_l, positive_b, positive_r], axis=-1)
        positive_tlbr *= validity[..., np.newaxis]
        return positive_tlbr
    
    def get_positive_map_in_boxes(self, positive_tlbr, grid_center_yx):
        """
        :param positive_tlbr: (B, N, 4) 
        :param grid_center_yx: (H, W, 2)
        :return: 
        """
        grid_center_yx = grid_center_yx[np.newaxis, ...]    # (1,H,W,2)
        positive_tlbr = positive_tlbr[:, np.newaxis, np.newaxis, ...]   # (B,1,1,N,4)
        # (1,H,W,1) - (B,1,1,N) -> (B,H,W,N)
        delta_t = grid_center_yx[..., 0:1] - positive_tlbr[..., 0]
        delta_l = grid_center_yx[..., 1:2] - positive_tlbr[..., 1]
        delta_b = positive_tlbr[..., 2] - grid_center_yx[..., 0:1]
        delta_r = positive_tlbr[..., 3] - grid_center_yx[..., 1:2]
        # (B, H, W, N, 4)
        tblr_grid_x_box = np.stack([delta_t, delta_l, delta_b, delta_r], axis=-1)
        # (B, H, W, N)
        positive_mask = np.all(tblr_grid_x_box > 0, axis=-1)
        # (B, H, W)
        positive_mask = np.any(positive_mask > 0, axis=-1)
        return positive_mask

    def merge_positive_maps(self, single_positive_map, object_from_box, object_from_radius):
        multi_positive_map = uf.convert_to_tensor(np.logical_and(object_from_box, object_from_radius), "int32")
        output_map = single_positive_map + (1 - single_positive_map) * multi_positive_map[:, np.newaxis, np.newaxis,...] \
                     * self.multi_positive_weight
        return output_map


class MultiPositivePolicy(ObjectDistribPolicy):
    def __init__(self, imshape, anchors_per_scale, center_radius=cfg.FeatureDistribPolicy.CENTER_RADIUS,
                 resolution=cfg.Datasets.DATASET_CONFIG.INPUT_RESOLUTION):
        super().__init__(imshape, anchors_per_scale)
        self.center_radius = center_radius
        self.resolution = resolution
        self.generate_feature_maps = MultiPositiveGenerator(imshape)
        self.decoder2d = decoder2d.FeatureDecoder(anchors_per_scale)
        self.decoder3d = decoder3d.FeatureDecoder3D(anchors_per_scale)

    def __call__(self, features):
        """
        :param features["inst2d"]: (B, N, 6), np.array, [yxhw, objectness, category]
        :param features["inst3d"]: (B, N, 9), np.array, [yxhwl, z, theta, objecntess, category]
        :return: 
        """
        features = self.generate_feature_maps(features)
        features["feat2d_logit"] = self.decoder2d.inverse(features["feat2d"])
        features["feat3d_logit"] = self.decoder3d.inverse(features["feat3d"], features["feat2d"])
        features = self.feature_merge(features)
        return features



