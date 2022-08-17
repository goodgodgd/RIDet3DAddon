import utils.tflow.util_function as uf
from RIDet3DAddon.tflow.train.loss_pool import *
import RIDet3DAddon.tflow.utils.util_function as uf3d
import RIDet3DAddon.config as cfg3d


class IntegratedLoss:
    def __init__(self, loss_weights, valid_category):
        self.loss_names = [key for key in loss_weights.keys()]
        self.use_ignore_mask = cfg3d.Train.IGNORE_MASK
        self.loss_weights = loss_weights
        self.iou_aware = cfg3d.ModelOutput.IOU_AWARE
        self.num_scale = len(cfg3d.ModelOutput.FEATURE_SCALES)
        # self.valid_category: binary mask of categories, (1, 1, K)
        self.valid_category = uf.convert_to_tensor(valid_category, 'float32')
        self.scaled_loss_objects = self.create_scale_loss_objects(loss_weights)

    def create_scale_loss_objects(self, loss_weights):
        loss_objects = dict()
        for loss_name, values in loss_weights.items():
            loss_objects[loss_name] = eval(values[1])(*values[2:])
        return loss_objects

    def __call__(self, features, predictions):
        # grtr_slices = uf3d.merge_and_slice_features(features, True)
        # pred_slices = uf3d.merge_and_slice_features(predictions, False)
        # features = self.merge_hwa_features(features)
        # predictions = self.merge_hwa_features(predictions)
        total_loss = 0
        loss_by_type = {loss_name: 0 for loss_name in self.loss_weights}
        for scale in range(self.num_scale):
            auxi = self.prepare_box_auxiliary_data(features["feat2d"], features["inst2d"],
                                                   predictions["feat2d"], scale)
            for loss_name, loss_object in self.scaled_loss_objects.items():
                loss_map_suffix = loss_name + "_map"
                if loss_map_suffix not in loss_by_type:
                    loss_by_type[loss_map_suffix] = []

                scalar_loss, loss_map = loss_object(features, predictions, auxi, scale)
                weight = self.loss_weights[loss_name][0][scale]
                total_loss += scalar_loss * weight
                loss_by_type[loss_name] += scalar_loss
                loss_by_type[loss_map_suffix].append(loss_map)

            # for loss_name, loss_object in self.scaled_loss_3d_objects.items():
            #     loss_map_suffix = loss_name + "_map"
            #     if loss_map_suffix not in loss_by_type:
            #         loss_by_type[loss_map_suffix] = []
            #
            #     scalar_loss, loss_map = loss_object(features["feat3d"], predictions["feat3d"], auxi, scale)
            #     weight = self.loss_3d_weights[loss_name][0][scale]
            #     total_loss += scalar_loss * weight
            #     loss_by_type[loss_name] += scalar_loss
            #     loss_by_type[loss_map_suffix].append(loss_map)

        return total_loss, loss_by_type

    def prepare_box_auxiliary_data(self, grtr_feat, grtr_boxes, pred_feat, scale):
        auxiliary = dict()
        # As object_count is used as a denominator, it must NOT be 0.
        auxiliary["object_count"] = uf.maximum(uf.reduce_sum(grtr_feat["object"][scale]), 1)
        auxiliary["valid_category"] = self.valid_category
        auxiliary["ignore_mask"] = self.get_ignore_mask(grtr_boxes, pred_feat, scale)
        return auxiliary

    def prepare_lane_auxiliary_data(self, grtr, pred):
        auxiliary = dict()
        # As object_count is used as a denominator, it must NOT be 0.
        auxiliary["object_count"] = uf.maximum(uf.reduce_sum(grtr["object"]), 1)
        auxiliary["valid_category"] = self.valid_category
        return auxiliary

    def get_ignore_mask(self, grtr, pred, scale):
        if not self.use_ignore_mask:
            return 1
        # b, h, w, a, _ = pred["yxhw"][scale].shape
        # merged_pred = uf3d.merge_dim_hwa(pred["yxhw"][scale])
        iou = uf.compute_iou_general(pred["yxhw"][scale], grtr["yxhw"])
        best_iou = uf.reduce_max(iou, axis=-1)
        ignore_mask = uf.cast(best_iou < 0.65, dtype='float32')
        # ignore_mask = tf.reshape(ignore_mask, (b, h, w, a))
        return ignore_mask

    def merge_hwa_features(self, features):
        for key in features.keys():
            if "feat" in key:
                for sub_key, sub_value in features[key].items():
                    for scale_index, scale_value in enumerate(sub_value):
                        features[key][sub_key][scale_index] = uf3d.merge_dim_hwa(scale_value)
        return features
