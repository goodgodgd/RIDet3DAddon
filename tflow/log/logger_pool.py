import numpy as np

import RIDet3DAddon.config as cfg
import utils.tflow.util_function as uf


class LogBase:
    def __init__(self, logkey):
        self.logkey = logkey
        self.num_categs = cfg.ModelOutput.PRED_MAIN_COMPOSITION["category"]

    def compute_mean(self, feature, mask, valid_num):
        if mask is None:
            return np.sum(feature, dtype=np.float32) / valid_num
        return np.sum(feature * mask[..., 0], dtype=np.float32) / valid_num

    def compute_sum(self, feature, mask):
        return np.sum(feature * mask, dtype=np.float32)

    def one_hot(self, grtr_category, category_shape):
        one_hot_data = np.eye(category_shape, dtype=np.float32)[grtr_category[..., 0].astype(np.int32)]
        return one_hot_data


class LogMeanLoss(LogBase):
    def __call__(self, grtr, pred, loss):
        # mean over all feature maps
        return loss[self.logkey]


class LogMeanDetailLoss(LogBase):
    def __call__(self, grtr, pred, loss):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch*HW, A, 4), 'object': ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch*HW, A, 4), 'object': ..., 'category', ...}
        :param return:
        """
        # divide into categories and anchors
        grtr_ctgr_mask = self.one_hot(grtr["feat2d"]["category"], self.num_categs)    # (b, h*w, anchor, 1) -> (b, h*w, anchor, category)
        grtr_tgarget_mask = grtr_ctgr_mask * grtr["feat2d"]["object"]
        nonan = (1 - grtr["feat2d"]["object"]) * 1e-12
        valid_num = np.sum(grtr_tgarget_mask + nonan, axis=(0, 1), dtype=np.float32)    # (anchor, category)
        loss_per_ctgr = np.expand_dims(loss[self.logkey], axis=-1) * grtr_tgarget_mask  # (b, h*w, anchor, category)
        return np.reshape(np.sum(loss_per_ctgr, axis=(0, 1), dtype=np.float32) / valid_num, -1) .tolist()   # (anchor * category)


class LogPositiveObj(LogBase):
    def __call__(self, grtr, pred, loss):
        pos_obj = 0
        for scale in range(len(grtr["feat2d"]["object"])):
            pos_obj_sc = self.compute_pos_obj(grtr["feat2d"]["object"][scale], pred["feat2d"]["object"][scale])
            pos_obj += pos_obj_sc
        return pos_obj / len(grtr["feat2d"]["object"])

    def compute_pos_obj(self, grtr_obj_mask, pred_obj_prob):
        obj_num = np.maximum(np.sum(grtr_obj_mask, dtype=np.float32), 1)
        pos_obj = np.sum(grtr_obj_mask * pred_obj_prob, dtype=np.float32) / obj_num
        return pos_obj


class LogNegativeObj(LogBase):
    def __call__(self, grtr, pred, loss):
        neg_obj = 0
        for scale in range(len(grtr["feat2d"]["object"])):
            neg_obj_sc = self.compute_neg_obj(grtr["feat2d"]["object"][scale], pred["feat2d"]["object"][scale])
            neg_obj += neg_obj_sc
        return neg_obj / len(grtr["feat2d"]["object"])

    def compute_neg_obj(self, grtr_obj_mask, pred_obj_prob):
        # average top 50 negative objectness probabilities per frame
        neg_obj_map = (1. - grtr_obj_mask) * pred_obj_prob
        batch, hwa, _ = neg_obj_map.shape
        neg_obj_map = np.reshape(neg_obj_map, (batch, hwa))
        neg_obj_map = np.sort(neg_obj_map, axis=1)[:, ::-1]
        neg_obj_map = neg_obj_map[:, :20]
        neg_obj = np.mean(neg_obj_map, dtype=np.float32)
        return neg_obj


class LogPositiveDetailObj(LogBase):
    def __call__(self, grtr, pred, loss):
        grtr_ctgr_mask = self.one_hot(grtr["feat2d"]["category"], self.num_categs)
        grtr_target_mask = grtr_ctgr_mask * grtr["feat2d"]["object"]
        nonan = (1 - grtr["feat2d"]["object"]) * 1e-12
        pos_obj_num = np.sum(grtr_target_mask + nonan, axis=(0, 1), dtype=np.float32)
        pos_obj = np.sum(grtr_target_mask * pred["feat2d"]["object"], axis=(0, 1), dtype=np.float32) / pos_obj_num
        return np.reshape(pos_obj, -1).tolist()


class LogNegativeDetailObj(LogBase):
    def __call__(self, grtr, pred, loss):
        best_pred_ctgr = np.expand_dims(np.argmax(pred["feat2d"]["category"], axis=-1), axis=-1)
        target_ctgr_mask = self.one_hot(best_pred_ctgr, self.num_categs)
        neg_obj_mask = 1. - grtr["feat2d"]["object"]
        neg_obj_map = neg_obj_mask * pred["feat2d"]["object"] * target_ctgr_mask
        batch, hw, a, channel = grtr["feat2d"]["object"].shape
        neg_obj_list = []
        for i in range(self.num_categs):
            neg_obj_per_ctgr = neg_obj_map[..., i]
            neg_obj_per_ctgr = np.reshape(neg_obj_per_ctgr, (batch*hw, a))
            neg_obj_per_ctgr = np.sort(neg_obj_per_ctgr, axis=0)[::-1]
            neg_obj_list.append(neg_obj_per_ctgr)
        neg_obj_map = np.stack(neg_obj_list, axis=-1)
        neg_obj_map = neg_obj_map[:(10*batch), :, :]
        neg_obj = np.mean(neg_obj_map, axis=0, dtype=np.float32)
        return np.reshape(neg_obj, -1).tolist()


class LogIouMean(LogBase):
    def __call__(self, grtr, pred, loss):
        grtr_ctgr_mask = self.one_hot(grtr["feat2d"]["category"], self.num_categs)
        grtr_obj_mask = grtr_ctgr_mask * grtr["feat2d"]["object"]
        grtr_yxhw = np.expand_dims(grtr["feat2d"]["yxhw"], axis=-2) * np.expand_dims(grtr_obj_mask, axis=-1)  # (b, h*w, anchor,category,yxhw)
        pred_yxhw = np.expand_dims(pred["feat2d"]["yxhw"], axis=-2) * np.expand_dims(pred["feat2d"]["category"], axis=-1)
        iou = uf.compute_iou_aligned(grtr_yxhw, pred_yxhw).numpy()
        iou = iou * grtr_obj_mask
        nonan = (1 - grtr["feat2d"]["object"]) * 1e-12
        valid_num = np.sum(grtr_obj_mask + nonan, axis=(0, 1), dtype=np.float32)
        iou = np.sum(iou, axis=(0, 1), dtype=np.float32) / valid_num
        return np.reshape(iou, -1).tolist()


class LogBoxYX(LogBase):
    def __call__(self, grtr, pred, loss):
        grtr_ctgr_mask = self.one_hot(grtr["feat2d"]["category"], self.num_categs)
        grtr_obj_mask = grtr_ctgr_mask * grtr["feat2d"]["object"]
        grtr_yxhw = np.expand_dims(grtr["feat2d"]["yxhw"], axis=-2) * np.expand_dims(grtr_obj_mask, axis=-1)  # (b, h*w, anchor,category,yxhw)
        pred_yxhw = np.expand_dims(pred["feat2d"]["yxhw"], axis=-2) * np.expand_dims(pred["feat2d"]["category"], axis=-1)
        yx_diff = (np.abs(grtr_yxhw[..., :2] - pred_yxhw[..., :2], dtype=np.float32)) / (grtr_yxhw[..., :2] + 1e-12)
        yx_sum = np.sum(yx_diff, axis=-1, dtype=np.float32) * grtr_obj_mask     # (b*h*w, anchor, category)
        nonan = (1 - grtr["feat2d"]["object"]) * 1e-12
        valid_num = np.sum(grtr_obj_mask + nonan, axis=(0, 1), dtype=np.float32)
        yx_mean = np.sum(yx_sum, axis=(0, 1), dtype=np.float32) / valid_num
        return np.reshape(yx_mean, -1).tolist()


class LogBoxHW(LogBase):
    def __call__(self, grtr, pred, loss):
        grtr_ctgr_mask = self.one_hot(grtr["feat2d"]["category"], self.num_categs)
        grtr_obj_mask = grtr_ctgr_mask * grtr["feat2d"]["object"]
        grtr_yxhw = np.expand_dims(grtr["feat2d"]["yxhw"], axis=-2) * np.expand_dims(grtr_obj_mask, axis=-1)  # (b, h*w, anchor,category,yxhw)
        pred_yxhw = np.expand_dims(pred["feat2d"]["yxhw"], axis=-2) * np.expand_dims(pred["feat2d"]["category"], axis=-1)
        hw_diff = (np.abs(grtr_yxhw[..., 2:] - pred_yxhw[..., 2:], dtype=np.float32)) / (grtr_yxhw[..., 2:] + 1e-12)
        hw_sum = np.sum(hw_diff, axis=-1, dtype=np.float32) * grtr_obj_mask
        nonan = (1 - grtr["feat2d"]["object"]) * 1e-12
        valid_num = np.sum(grtr_obj_mask + nonan, axis=(0, 1), dtype=np.float32)
        hw_mean = np.sum(hw_sum, axis=(0, 1), dtype=np.float32) / valid_num
        return np.reshape(hw_mean, -1).tolist()


class LogTrueClass(LogBase):
    def __call__(self, grtr, pred, loss):
        grtr_ctgr_mask = self.one_hot(grtr["feat2d"]["category"], self.num_categs)
        grtr_target_mask = grtr_ctgr_mask * grtr["feat2d"]["object"]
        # nonan = (1 - grtr["object"]) * 1e-12
        # valid_num = np.sum(grtr_target_mask + nonan, axis=0, dtype=np.float32)
        category_prob = grtr_target_mask * pred["feat2d"]["category"]
        true_class = np.max(category_prob, axis=(0, 1))
        # true_class = np.max(category_prob, axis=0) / valid_num
        return np.reshape(true_class, -1).tolist()


class LogFalseClass(LogBase):
    def __call__(self, grtr, pred, loss):
        grtr_ctgr_mask = self.one_hot(grtr["feat2d"]["category"], self.num_categs)
        false_ctgr_mask = (1. - grtr_ctgr_mask)
        false_prob = false_ctgr_mask * pred["feat2d"]["category"] * grtr["feat2d"]["object"]
        false_class = np.max(false_prob, axis=(0, 1))
        return np.reshape(false_class, -1).tolist()


class LogBoxYXZ(LogBase):
    def __call__(self, grtr, pred, loss):
        grtr_ctgr_mask = self.one_hot(grtr["feat3d"]["category"], self.num_categs)
        grtr_obj_mask = grtr_ctgr_mask * grtr["feat2d"]["object"]
        grtr_yxz = np.expand_dims(grtr["feat3d"]["yxz"], axis=-2) * np.expand_dims(grtr_obj_mask,
                                                                           axis=-1)  # (b, h*w, anchor,category,yxz)
        pred_yxz = np.expand_dims(pred["feat3d"]["yxz"], axis=-2) * np.expand_dims(pred["feat3d"]["category"], axis=-1)
        yxz_diff = (np.abs(grtr_yxz - pred_yxz, dtype=np.float32)) / (grtr_yxz + 1e-12) * grtr_obj_mask[..., np.newaxis]
        yxz_sum = np.sum(yxz_diff, axis=-1, dtype=np.float32)        # (b*h*w, anchor, category)
        nonan = (1 - grtr["feat2d"]["object"]) * 1e-12
        valid_num = np.sum(grtr_obj_mask + nonan, axis=(0, 1), dtype=np.float32)
        yxz_mean = np.sum(yxz_sum, axis=(0, 1), dtype=np.float32) / valid_num
        return np.reshape(yxz_mean, -1).tolist()


class LogBoxHWL(LogBase):
    def __call__(self, grtr, pred, loss):
        grtr_ctgr_mask = self.one_hot(grtr["feat3d"]["category"], self.num_categs)
        grtr_obj_mask = grtr_ctgr_mask * grtr["feat2d"]["object"]
        grtr_hwl = np.expand_dims(grtr["feat3d"]["hwl"], axis=-2) * np.expand_dims(grtr_obj_mask,
                                                                           axis=-1)  # (b, h*w, anchor,category,yxhw)
        pred_hwl = np.expand_dims(pred["feat3d"]["hwl"], axis=-2) * np.expand_dims(pred["feat3d"]["category"], axis=-1)

        hwl_diff = (np.abs(grtr_hwl - pred_hwl, dtype=np.float32)) / (grtr_hwl + 1e-12) * np.expand_dims(pred["feat3d"]["category"], axis=-1)
        hwl_sum = np.sum(hwl_diff, axis=-1, dtype=np.float32)        # (b*h*w, anchor, category)
        nonan = (1 - grtr["feat2d"]["object"]) * 1e-12
        valid_num = np.sum(grtr_obj_mask + nonan, axis=(0, 1), dtype=np.float32)
        hwl_mean = np.sum(hwl_sum, axis=(0, 1), dtype=np.float32) / valid_num
        return np.reshape(hwl_mean, -1).tolist()




