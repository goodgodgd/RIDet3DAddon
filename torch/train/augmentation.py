# import cv2
# import torch
# import numpy as np
# import torchvision
#
# import RIDet3DAddon.torch.config as cfg3d
# import config as cfg3d
#
#
# class Augmentation:
#     def __init__(self, augmentation=cfg3d.Train.AUGMENT_PROBS):
#         self.augmentation = augmentation
#
#     def __call__(self, features):
#         for key, val in self.augmentation.items():
#             val.update({"image_shape": features["image"].shape})
#             class_name = eval(f"{key}")(**val)
#             features["image"] = class_name.apply_image(features["image"])
#             features["inst2d"] = class_name.apply_box2d(features["inst2d"])
#
#
# class Flip:
#     def __init__(self, **kwargs):
#         self.direction = kwargs["direction"]
#         self.im_shape = kwargs["image_shape"]
#         self.device = cfg3d.Train.DEVICE
#
#     def apply_image(self, image):
#         flip_image = torch.flip(image, dims=[self.direction])
#         aug_image = torch.concat([image, flip_image], dim=0)
#         return aug_image
#
#     def apply_box2d(self, box):
#         box_remind = box[..., 4:]
#         valid_mask = box[..., 4:5] > 0
#
#         test = torch.tensor([2 - self.direction, self.direction - 1], dtype=torch.float32, device=self.device)
#         box_flip_ =  box[..., 0:1] - test
#         # if self.direction == 2:
#         #     box_flip_ = torch.stack([box[..., 0], 1 - box[..., 1], box[..., 2], box[..., 3]], dim=-1) * valid_mask
#         # else:
#         #     box_flip_ = torch.stack([1 - box[..., 0], box[..., 1], box[..., 2], box[..., 3]], dim=-1) * valid_mask
#         box_flip = torch.concat([box_flip_, box_remind], dim=-1)
#         aug_box = torch.concat([box, box_flip], dim=0)
#         aug_box_np = aug_box.detach().cpu().numpy()
#         return aug_box


import albumentations as A
import torch
import numpy as np

import RIDet3DAddon.torch.config as cfg3d
import config as cfg
import utils.framework.util_function as uf


def augmentation_factory(augment_probs=None):
    if augment_probs:
        augmenters = []
        for key, prob in augment_probs.items():
            if key == "ColorJitter":
                augmenters.append(A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0, p=prob))
            elif key == "Flip":
                augmenters.append(A.HorizontalFlip(p=prob, always_apply=True))
            elif key == "CropResize":
                augmenters.append(A.OneOf([A.RandomSizedCrop((160, 320), 320, 1024, w2h_ratio=2.5, p=prob),
                                  A.RandomScale((-0.5, -0.5), p=prob)], p=1.0))
            elif key == "Blur":
                augmenters.append(A.Blur(p=prob))
        # augmenters.append(A.PadIfNeeded(320, 1024, border_mode=0, value=0, always_apply=True))
        aug_func = A.ReplayCompose(augmenters, bbox_params=A.BboxParams(format="yolo", min_visibility=0.5,
                                                                  label_fields=["remainder"]))
        # "remainder" mean object, class, minor_class, distance. except bbox coord
        total_augment = TotalAugment(aug_func)
    else:
        total_augment = None
    return total_augment


class TotalAugment:
    def __init__(self, augment_objects=None):
        self.augment_objects = augment_objects
        self.max_bbox = cfg.Dataloader.MAX_BBOX_PER_IMAGE

    def __call__(self, features):
        total_image = []
        total_bboxes = []
        total_inst3d = []
        batch_size = features["image"].shape[0]
        for i in range(batch_size):
            image = features["image"][i]
            bboxes = features["inst2d"][i]
            inst3d = features["inst3d"][i]
            raw_data = self.preprocess(image, bboxes, inst3d)
            # raw_data["imshape"] = features["image_shape"][i]
            # raw_data["intrinsic"] = features["intrinsic"][i].cpu().detach().numpy()
            aug_data = self.transformation(raw_data)
            aug_data = self.post_process(aug_data)
            total_image.extend([image[np.newaxis, ...], aug_data["image"]])
            total_bboxes.extend([bboxes[np.newaxis, ...], aug_data["total_bboxes"]])
            total_inst3d.extend([inst3d[np.newaxis, ...], aug_data["inst3d"]])
        features["image"] = torch.concat(total_image, dim=0)
        features["inst2d"] = torch.concat(total_bboxes, dim=0)
        features["inst3d"] = torch.concat(total_inst3d, dim=0)
        # features["intrinsic"] = uf.convert_to_tensor(np.repeat(features["intrinsic"], 2, axis=0), dtype="float32")
        # features["image_shape"] = uf.convert_to_tensor(np.repeat(features["image_shape"], 2, axis=0), dtype="float32")
        return features

    def preprocess(self, image, bboxes, inst3d):
        data = {"image": image.cpu().detach().numpy()}
        yxhw = uf.convert_tensor_to_numpy(bboxes[:, :4])
        remainder = uf.convert_tensor_to_numpy(bboxes[:, 4:])
        inst3d = uf.convert_tensor_to_numpy(inst3d)
        valid_mask = yxhw[:, 2] > 0
        yxhw = yxhw[valid_mask, :]

        data["inst3d"] = inst3d[valid_mask, :]
        data["remainder"] = remainder[valid_mask, :]
        data["xywh"] = self.convert_coord(yxhw)
        return data

    def convert_coord(self, coord):
        if len(coord) < 1:
            convert_coord = np.array([[0, 0, 0, 0]], dtype=np.float32)
        else:
            convert_coord = np.array([coord[:, 1], coord[:, 0], coord[:, 3], coord[:, 2]], dtype=np.float32).T

        return convert_coord

    def transformation(self, data):
        if not np.all(data["xywh"]):
            aug_data = dict()
            aug_data["image"] = data["image"]
            aug_data["bboxes"] = data["xywh"]
            aug_data["remainder"] = data["remainder"]
            aug_data["inst3d"] = data["inst3d"]
        else:
            augment_data = self.augment_objects(image=data["image"], bboxes=data["xywh"], remainder=data["remainder"])
            aug_data = {key: np.array(val) if isinstance(val, list) else val for key, val in augment_data.items()}
            aug_data["inst3d"] = data["inst3d"]
            # is_crop = augment_data["replay"]["transforms"][1]["transforms"][0]["applied"]
            # is_resize = augment_data["replay"]["transforms"][1]["transforms"][1]["applied"]
            is_flip = augment_data["replay"]["transforms"][0]["applied"]
            # if is_crop:
            #     h_ratio = augment_data["replay"]["transforms"][1]["transforms"][0]["params"]["crop_height"] / data["imshape"][0]
            #     w_ratio = augment_data["replay"]["transforms"][1]["transforms"][0]["params"]["crop_width"] / data["imshape"][1]
            #     proj_box = self.project_to_image(data["inst3d"][..., :2], data["inst3d"][..., 2:3], data["intrinsic"])
            #     proj_box = proj_box[..., :2] / np.array([h_ratio, w_ratio])
            # elif is_resize:
            #     ratio = augment_data["replay"]["transforms"][1]["transforms"][1]["params"]["scale"]
            if is_flip:
                flip_theta = np.pi - data["inst3d"][..., -2]
                for i, flip_data in enumerate(flip_theta):
                    # check range
                    if flip_data > np.pi:
                        flip_theta[i] += 2 * np.pi
                    if flip_data < np.pi:
                        flip_theta[i] -= 2 * np.pi
                aug_data["inst3d"][..., -2] = flip_theta
                # aug_data["inst3d"][..., 1] = -data["inst3d"][..., 1]
                aug_data["inst3d"] = self.flip_inst3d(data["inst3d"])
            aug_data["bboxes"] = self.convert_coord(aug_data["bboxes"])
        return aug_data

    def flip_inst3d(self, box_xyz):
        box_xyz[:, 1] = box_xyz[:, 1] * -1
        box_xyz[:, 6] = box_xyz[:, 6] * -1
        return box_xyz

    def convert_list_to_numpy(self, data):
        convert_data = np.asarray(data, dtype=np.float32)
        return convert_data

    def post_process(self, aug_data):
        aug_data["image"] = uf.convert_to_tensor(aug_data["image"], dtype="float32")[None, ...]
        if aug_data["bboxes"].shape[0] < self.max_bbox and len(aug_data["bboxes"]) != 0:
            bboxes = aug_data["bboxes"]
            inst3d = aug_data["inst3d"]
            remainder = aug_data["remainder"]
            aug_data["bboxes"] = np.zeros((self.max_bbox, aug_data["bboxes"].shape[1]), dtype=np.float32)
            aug_data["remainder"] = np.zeros((self.max_bbox, aug_data["remainder"].shape[1]), dtype=np.float32)
            aug_data["bboxes"][:bboxes.shape[0]] = bboxes
            aug_data["remainder"][:remainder.shape[0]] = remainder
            aug_data["inst3d"] = np.zeros((self.max_bbox, aug_data["inst3d"].shape[1]), dtype=np.float32)
            aug_data["inst3d"][:inst3d.shape[0]] = inst3d

        bbox_total_labels = np.concatenate([aug_data["bboxes"], aug_data["remainder"]], axis=-1)
        aug_data["total_bboxes"] = uf.convert_to_tensor(bbox_total_labels, dtype="float32")[None, ...]
        aug_data["inst3d"] = uf.convert_to_tensor(aug_data["inst3d"], dtype="float32")[None, ...]
        return aug_data

