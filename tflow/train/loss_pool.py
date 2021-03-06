import tensorflow as tf
import numpy as np

import utils.util_function as uf


class LossBase:
    def __call__(self, grtr, pred, auxi, scale):
        dummy_large = tf.reduce_mean(tf.square(pred["feature_l"]))
        dummy_medium = tf.reduce_mean(tf.square(pred["feature_m"]))
        dummy_small = tf.reduce_mean(tf.square(pred["feature_s"]))
        dummy_total = dummy_large + dummy_medium + dummy_small
        return dummy_total, {"dummy_large": dummy_large, "dummy_medium": dummy_medium, "dummy_small": dummy_small}


class CiouLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: complete-iou loss (batch, HWA)
        """
        # object_mask: (batch, HWA, 1)
        object_mask = tf.cast(grtr["object"][scale] == 1, dtype=tf.float32)
        ciou_loss = self.compute_ciou(grtr["yxhw"][scale], pred["yxhw"][scale]) * object_mask[..., 0]
        # sum over object-containing grid cells
        scalar_loss = tf.reduce_sum(ciou_loss)
        return scalar_loss, ciou_loss

    def compute_ciou(self, grtr_yxhw, pred_yxhw):
        """
        :param grtr_yxhw: (batch, HWA, 4)
        :param pred_yxhw: (batch, HWA, 4)
        :return: ciou loss (batch, HWA)
        """
        grtr_tlbr = uf.convert_box_format_yxhw_to_tlbr(grtr_yxhw)
        pred_tlbr = uf.convert_box_format_yxhw_to_tlbr(pred_yxhw)
        # iou: (batch, HWA)
        iou = uf.compute_iou_aligned(grtr_yxhw, pred_yxhw, grtr_tlbr, pred_tlbr)
        cbox_tl = tf.minimum(grtr_tlbr[..., :2], pred_tlbr[..., :2])
        cbox_br = tf.maximum(grtr_tlbr[..., 2:], pred_tlbr[..., 2:])
        cbox_hw = cbox_br - cbox_tl
        c = tf.reduce_sum(cbox_hw * cbox_hw, axis=-1)
        center_diff = grtr_yxhw[..., :2] - pred_yxhw[..., :2]
        u = tf.reduce_sum(center_diff * center_diff, axis=-1)
        # NOTE: divide_no_nan results in nan gradient
        # d = tf.math.divide_no_nan(u, c)
        d = u / (c + 1.0e-5)

        # grtr_hw_ratio = tf.math.divide_no_nan(grtr_yxhw[..., 2], grtr_yxhw[..., 3])
        # pred_hw_ratio = tf.math.divide_no_nan(pred_yxhw[..., 2], pred_yxhw[..., 3])
        grtr_hw_ratio = grtr_yxhw[..., 3] / (grtr_yxhw[..., 2] + 1.0e-5)
        pred_hw_ratio = pred_yxhw[..., 3] / (pred_yxhw[..., 2] + 1.0e-5)
        coeff = tf.convert_to_tensor(4.0 / (np.pi * np.pi), dtype=tf.float32)
        v = coeff * tf.pow((tf.atan(grtr_hw_ratio) - tf.atan(pred_hw_ratio)), 2)
        alpha = v / (1 - iou + v)
        penalty = d + alpha * v
        loss = 1 - iou + penalty
        return loss


class IouLoss(LossBase):
    def __init__(self, use_l1=False):
        self.use_l1 = use_l1

    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: complete-iou loss (batch, HWA)
        """
        # object_mask: (batch, HWA, 1)
        object_mask = tf.cast(grtr["object"][scale] == 1, dtype=tf.float32)
        iou_loss = self.compute_iou(grtr["yxhw"][scale], pred["yxhw"][scale]) * object_mask[..., 0]
        # sum over object-containing grid cells
        scalar_loss = tf.reduce_sum(iou_loss)
        if self.use_l1:
            scalar_l1, l1_loss = IoUL1smooth()(grtr, pred, auxi, scale)
            scalar_loss += scalar_l1
            iou_loss += l1_loss
        return scalar_loss, iou_loss

    def compute_iou(self, grtr_yxhw, pred_yxhw):
        grtr_tlbr = uf.convert_box_format_yxhw_to_tlbr(grtr_yxhw)
        pred_tlbr = uf.convert_box_format_yxhw_to_tlbr(pred_yxhw)
        iou = uf.compute_iou_aligned(grtr_yxhw, pred_yxhw, grtr_tlbr, pred_tlbr)
        iou_loss = 1 - iou
        return iou_loss


class IoUL1smooth(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        object_mask = tf.cast(grtr["object"][scale] == 1, dtype=tf.float32)
        # grtr_tlbr = uf.convert_box_format_yxhw_to_tlbr(grtr["yxhw"][scale]) * object_mask
        # pred_tlbr = uf.convert_box_format_yxhw_to_tlbr(pred["yxhw"][scale]) * object_mask

        huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)  # (batch, HWA)
        l1_loss = huber_loss(grtr["yxhw"][scale], pred["yxhw"][scale]) * object_mask[..., 0]
        scalar_loss = tf.reduce_sum(l1_loss)
        return scalar_loss, l1_loss


class BoxObjectnessLoss(LossBase):
    def __init__(self, pos_weight, neg_weight):
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: objectness loss (batch, HWA)
        """
        grtr_obj = grtr["object"][scale]
        pred_obj = pred["object"][scale]
        ignore_mask = auxi["ignore_mask"]
        conf_focal = tf.pow(grtr_obj - pred_obj, 2)
        obj_loss = tf.keras.losses.binary_crossentropy(grtr_obj, pred_obj, label_smoothing=0.04) * conf_focal[..., 0]
        obj_positive = obj_loss * grtr_obj[..., 0] * self.pos_weight
        obj_negative = obj_loss * (1 - grtr_obj[..., 0]) * ignore_mask * self.neg_weight
        scalar_loss = tf.reduce_sum(obj_positive) + tf.reduce_sum(obj_negative)
        return scalar_loss, obj_loss


class CategoryLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        pass

    def compute_category_loss(self, grtr, pred, mask):
        """
        :param grtr: (batch, HWA, 1)
        :param pred: (batch, HWA, K)
        :param mask: (batch, HWA, K or 1)
        :return: (batch, HWA)
        """
        num_cate = pred.shape[-1]
        grtr_label = tf.cast(grtr, dtype=tf.int32)
        grtr_onehot = tf.one_hot(grtr_label[..., 0], depth=num_cate, axis=-1)[..., tf.newaxis]
        # category_loss: (batch, HWA, K)
        category_loss = tf.losses.binary_crossentropy(grtr_onehot, pred[..., tf.newaxis], label_smoothing=0.04)
        loss_map = category_loss * mask
        loss_map = tf.reduce_sum(loss_map, axis=-1)
        loss = tf.reduce_sum(loss_map)
        return loss, loss_map


class MajorCategoryLoss(CategoryLoss):
    def __call__(self, grtr, pred, auxi, scale):
        """
        :param grtr: GT feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param pred: pred. feature map slices of some scale, {'yxhw': (batch, HWA, 4), 'object', ..., 'category', ...}
        :param auxi: auxiliary data
        :return: category loss (batch, HWA, K)
        """
        object_mask, valid_category = tf.cast(grtr["object"][scale] == 1, dtype=tf.float32), auxi["valid_category"]
        scalar_loss, category_loss = self.compute_category_loss(grtr["category"][scale], pred["category"][scale],
                                                                object_mask * valid_category)
        return scalar_loss, category_loss


class Box3DLoss(LossBase):
    def __call__(self, grtr, pred, auxi, scale):
        object_mask = tf.cast(grtr["object"][scale] == 1, dtype=tf.float32)
        pred = self.box_preprocess(grtr, pred, scale)
        huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        box_3d_loss = huber_loss(grtr["yxhwl"][scale], pred["yxhwl"][scale]) * object_mask
        scalar_loss = tf.reduce_sum(box_3d_loss)
        return scalar_loss, box_3d_loss

    def box_preprocess(self, grtr, pred, scale):
        # TODO gt logit
        yxh = pred["yxhwl"][scale][..., :3]
        w = pred["yxhwl"][scale][..., 3:4]
        l = pred["yxhwl"][scale][..., 4:5]
        yxhlw = tf.concat([yxh, l, w], axis=-1)
        M = tf.abs(pred["theta"][scale] - grtr["theta"][scale]) < (np.pi / 4)
        pred["yxhwl"][scale][..., 3:5] = M * pred["yxhwl"][scale][..., 3:5] + (1 - M) * yxhlw[..., 3:5]
        return pred


class ThetaLoss(LossBase):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, grtr, pred, auxi, scale):
        object_mask = tf.cast(grtr["object"][scale] == 1, dtype=tf.float32)
        c_theta = tf.math.cos(grtr["theta"][scale] - pred["theta"][scale]) * 4
        theta_loss = 2 - (1/(2*self.beta) * tf.math.log((self.alpha+c_theta)/(self.alpha-c_theta))) * object_mask[..., 0]
        scalar_loss = tf.reduce_sum(theta_loss)
        return scalar_loss, theta_loss
