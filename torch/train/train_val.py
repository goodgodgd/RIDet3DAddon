import time
import cv2
import numpy as np
import torch
from timeit import default_timer as timer

import config as cfg
import utils.framework.util_function as uf
from RIDet3DAddon.torch.log.logger import Logger
import RIDet3DAddon.torch.config as cfg3d
import RIDet3DAddon.torch.log.visual_log as vl3d
import RIDet3DAddon.torch.dataloader.data_util as du3d


class TrainValBase:
    def __init__(self, model, loss_object, optimizer, epoch_steps, feature_creator, ckpt_path):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps
        self.ckpt_path = ckpt_path
        self.is_train = True
        self.device = cfg.Train.DEVICE
        self.feat_scales = cfg.ModelOutput.FEATURE_SCALES
        self.feature_creator = feature_creator
        num_channels = len(cfg3d.Train.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg3d.Train.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg3d.Train.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def run_epoch(self, dataset, epoch,  visual_log=False, exhaustive_log=False, val_only=False):
        #  dataset, scheduler, epoch=0, visual_log=False, exhaustive_log=False, val_only=False
        self.mode_set()
        logger = Logger(visual_log, exhaustive_log, self.loss_object.loss_names, self.ckpt_path, epoch, self.is_train,
                        val_only)
        epoch_start = timer()
        train_loader_iter = iter(dataset)
        steps = len(train_loader_iter)
        for step in range(steps):
            start = timer()
            features_ = self.to_device(next(train_loader_iter))

            features = features_.copy()
            frame_names = features['frame_names']
            del features["frame_names"]
            prediction, total_loss, loss_by_type, new_features = self.run_batch(features)
            start_log = timer()
            new_features["frame_names"] = frame_names
            logger.log_batch_result(step, new_features, prediction, total_loss, loss_by_type)
            print_loss = ""
            for key in loss_by_type.keys():
                if not "map" in key:
                    print_loss = print_loss+ f"{key}={loss_by_type[key]:.3f}, "
            # logger.append_batch_result(step, features, prediction, total_loss, loss_by_type)
            uf.print_progress(f"training {step}/{self.epoch_steps} steps, {epoch} epoch "
                              f"time={timer() - start:.3f}, "
                              f"loss={total_loss:.3f}, " +
                              print_loss

                              )

            if step >= self.epoch_steps:
                break
            # if step >= 10:
            #     break
            features["frame_names"] = frame_names

        logger.finalize(epoch_start)

    def to_device(self, features):
        for key in features:
            if isinstance(features[key], torch.Tensor):
                features[key] = features[key].to(device=self.device)
            elif isinstance(features[key], list):
                data = list()
                for feature in features[key]:
                    if isinstance(feature, torch.Tensor):
                        feature = feature.to(device=self.device)
                    data.append(feature)
                features[key] = data
            elif isinstance(features[key], dict):
                features[key] = self.to_device(features[key])
        return features

    def run_step(self, features):
        raise NotImplementedError()

    def mode_set(self):
        raise NotImplementedError()

    def permute_channel(self, features):  ## image

        image = self.normalizer(features.permute(0, 3, 1, 2).to(self.device))
        return image


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, optimizer, epoch_steps, feature_creator, ckpt_path):
        super().__init__(model, loss_object, optimizer, epoch_steps, feature_creator, ckpt_path)
        self.split = "train"
        self.is_train = True
        # self.visual_log2d = vl3d.VisualLog2d(ckpt_path,epoch_steps)

    def run_batch(self, features):
        features = self.feature_creator(features)


        # for i in range(features["image"].shape[0]):
        #     inst_image = features["image"][i].copy()
        #     inst_image = self.visual_log2d.draw_boxes(inst_image, features["inst2d"], i, (255, 0, 0))
        #     # image = self.visual_log2d.draw_lanes(image, features["inst_lane"], i, (255, 0, 255))
        #     feat = []
        #     for scale in range(3):
        #         feature = features["feat2d"]["whole"][scale][i]
        #         test = feature[4,...] > 0
        #         feature = feature[:,feature[4, ...] > 0]
        #         feat.append(feature)
        #     feat_boxes = np.concatenate(feat, axis=-1)
        #     feat_boxes = uf.convert_box_format_yxhw_to_tlbr(feat_boxes.T)
        #     if len(feat_boxes) == 0:
        #         feat_boxes = np.array([[0, 0, 0, 0]], dtype=np.float32)
        #     feat_image = features["image"][i].copy()
        #     feat_image = du3d.draw_box(feat_image, feat_boxes)
        #     total_image = np.concatenate([feat_image, inst_image], axis=0)
        #     cv2.imshow("test", total_image)
        #     cv2.waitKey(0)

        return self.run_step(features)

    def run_step(self, features):
        features = uf.convert_to_tensor(features, "float32", True)
        input_image = self.permute_channel(features['image'])
        prediction = self.model(input_image)
        total_loss, loss_by_type = self.loss_object(features, prediction)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return prediction, total_loss, loss_by_type, features

    def mode_set(self):
        self.model.train()


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, epoch_steps, feature_creator, ckpt_path):
        super().__init__(model, loss_object, None, epoch_steps, feature_creator,
                         ckpt_path)

        self.is_train = False

    def run_batch(self, features):
        features = self.feature_creator(features)

        return self.run_step(features)

    def run_step(self, features):
        features = uf.convert_to_tensor(features, "float32", True)
        input_image = self.permute_channel(features['image'])
        prediction = self.model(input_image)
        total_loss, loss_by_type = self.loss_object(features, prediction)
        return prediction, total_loss, loss_by_type, features

    def mode_set(self):
        self.model.eval()


