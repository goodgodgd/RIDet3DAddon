import time

from RIDet3DAddon.torch.train.loss_pool import *
import utils.framework.util_function as uf
import RIDet3DAddon.torch.utils.util_function as uf3d
import RIDet3DAddon.torch.config as cfg3d


class IntegratedLoss:
    def __init__(self, loss_weights, valid_category):
        self.loss_names = [key for key in loss_weights.keys()]
        self.use_ignore_mask = cfg.Train.IGNORE_MASK
        self.loss_weights = loss_weights
        self.iou_aware = cfg.ModelOutput.IOU_AWARE
        self.num_scale = len(cfg.ModelOutput.FEATURE_SCALES)
        # self.valid_category: binary mask of categories, (1, 1, K)
        self.valid_category = uf.convert_to_tensor(valid_category, 'float32')
        self.scaled_loss_objects = self.create_scale_loss_objects(loss_weights)

    def create_scale_loss_objects(self, loss_weights):
        loss_objects = dict()
        for loss_name, values in loss_weights.items():
            loss_objects[loss_name] = eval(values[1])(*values[2:])
        return loss_objects

    def __call__(self, features, predictions):
        start_time = time.time()
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
                loss_by_type[loss_name] += scalar_loss * weight
                loss_by_type[loss_map_suffix].append(loss_map)
        return total_loss, loss_by_type

    def prepare_box_auxiliary_data(self, grtr_feat, grtr_boxes, pred_feat, scale):
        auxiliary = dict()
        # As object_count is used as a denominator, it must NOT be 0.
        auxiliary["object_count"] = uf.maximum(uf.reduce_sum(grtr_feat["object"][scale], 1), torch.tensor(1))
        auxiliary["valid_category"] = self.valid_category
        auxiliary["ignore_mask"] = self.get_ignore_mask(grtr_boxes, pred_feat, scale)

        return auxiliary

    def get_ignore_mask(self, grtr, pred, scale):
        if not self.use_ignore_mask:
            return 1
        iou = uf.compute_iou_general(pred["yxhw"][scale], grtr["yxhw"])
        best_iou_values, best_iou_indices = uf.reduce_max(iou, dim=-1)
        ignore_mask = best_iou_values < 0.65
        return ignore_mask
