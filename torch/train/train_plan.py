import os
import os.path as op
import torch
import numpy as np
import pandas as pd

import RIDet3DAddon.torch.config_dir.util_config as uc3d
import RIDet3DAddon.torch.config as cfg3d
from RIDet3DAddon.torch.dataloader.dataset_reader import DatasetReader
from RIDet3DAddon.torch.model.model_factory import ModelFactory
from RIDet3DAddon.torch.train.loss_factory import IntegratedLoss
import RIDet3DAddon.torch.train.train_val as tv
import train.framework.train_util as tu
from RIDet3DAddon.torch.train.feature_generator import FeatureMapDistributer
from RIDet3DAddon.torch.train.augmentation import augmentation_factory
from utils.snapshot_code import CodeSnapshot
import config as cfg


def train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, model_save):
    batch_size, data_batch_size, anchors = cfg.Train.BATCH_SIZE, cfg.Train.DATA_BATCH_SIZE, cfg.AnchorGeneration.ANCHORS
    data_path, ckpt_path = cfg3d.Paths.DATAPATH, op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    valid_category, start_epoch = prepare_train(dataset_name, ckpt_path)

    if end_epoch <= start_epoch:
        print(f"!! end_epoch {end_epoch} <= start_epoch {start_epoch}, no need to train")
        return

    dataset_train, train_steps, imshape, anchors_per_scale = \
        get_dataset(data_path, dataset_name, True, data_batch_size, "train", anchors)
    dataset_val, val_steps, _, _ = get_dataset(data_path, dataset_name, False, batch_size, "val", anchors)

    model, loss_object, optimizer = create_training_parts(batch_size, imshape, anchors_per_scale, ckpt_path,
                                                          learning_rate, loss_weights, valid_category)
    augmentation = augmentation_factory(cfg3d.Train.AUGMENT_PROBS)
    feature_creator = FeatureMapDistributer(cfg3d.FeatureDistribPolicy.POLICY_NAME, imshape, anchors_per_scale)
    trainer = tv.ModelTrainer(model, loss_object, optimizer, train_steps, feature_creator, augmentation, ckpt_path)
    validater = tv.ModelValidater(model, loss_object, val_steps, feature_creator, ckpt_path)

    for epoch in range(start_epoch, end_epoch):
        print(f"========== Start dataset : {dataset_name} epoch: {epoch + 1}/{end_epoch} ==========")
        detail_log = (epoch in cfg.Train.DETAIL_LOG_EPOCHS)
        trainer.run_epoch(dataset_train, epoch)
        validater.run_epoch(dataset_val, epoch, detail_log, detail_log)
        save_model_ckpt(ckpt_path, model)
    if model_save:
        save_model_ckpt(ckpt_path, model, f"ep{end_epoch:02d}")


def prepare_train(dataset_name, ckpt_path):
    valid_category = uc3d.get_valid_category_mask(dataset_name)
    start_epoch = read_previous_epoch(ckpt_path)
    CodeSnapshot(ckpt_path, start_epoch)()
    return valid_category, start_epoch


def create_training_parts(batch_size, imshape, anchors_per_scale, ckpt_path, learning_rate,
                          loss_weights, valid_category, weight_suffix='latest'):
    model = ModelFactory(batch_size, imshape, anchors_per_scale).get_model()
    model = try_load_weights(ckpt_path, model, weight_suffix)
    loss_object = IntegratedLoss(loss_weights, valid_category)
    optimizer = build_optimizer(model, learning_rate)
    return model, loss_object, optimizer


def save_model_ckpt(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.pt")
    if not op.isdir(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    print("=== save model:", ckpt_file)
    torch.save(model.state_dict(), ckpt_file)
    # model.save('./my_model')


def get_dataset(datapath, dataset_name, shuffle, batch_size, split, anchors):
    # data_split_path = op.join(datapath, f"{dataset_name}_{split}")
    reader = DatasetReader(dataset_name, datapath, split, shuffle, batch_size, 1)
    dataset = reader.get_dataset()
    frames = reader.get_total_frames()
    image_shape = cfg3d.Datasets.DATASET_CONFIG.INPUT_RESOLUTION
    # anchor sizes per scale in pixel
    anchors_per_scale = np.array([anchor / np.array([image_shape[:2]]) for anchor in anchors], dtype=np.float32)
    print(f"[get_dataset] dataset={dataset_name}, image shape={image_shape}, "
          f"frames={frames},\n\tanchors={anchors_per_scale}")
    return dataset, frames // batch_size, image_shape, anchors_per_scale


def try_load_weights(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.pt")
    if op.isfile(ckpt_file):
        print(f"===== Load weights from checkpoint: {ckpt_file}")
        model = tu.load_weights(model, ckpt_file)
    else:
        print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
    return model


def read_previous_epoch(ckpt_path):
    filename = op.join(ckpt_path, 'history.csv')
    if op.isfile(filename):
        history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        if history.empty:
            print("[read_previous_epoch] EMPTY history:", history)
            return 0

        epochs = history['epoch'].tolist()
        epochs.sort()
        prev_epoch = epochs[-1]
        print(f"[read_previous_epoch] start from epoch {prev_epoch + 1}")
        return prev_epoch + 1
    else:
        print(f"[read_previous_epoch] NO history in {filename}")
        return 0


def build_optimizer(model: torch.nn.Module, learning_rate) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = [{"params": [], "lr": 0, "weight_decay": 0}]
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = learning_rate
        weight_decay = cfg3d.Optimizer.WEIGHT_DECAY
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg3d.Optimizer.WEIGHT_DECAY_NORM
        elif key.endswith(".bias"):
            lr = learning_rate
            weight_decay = cfg3d.Optimizer.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=cfg3d.Optimizer.MOMENTUM)
    return optimizer