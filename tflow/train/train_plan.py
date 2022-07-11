import os
import os.path as op
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

import config_dir.util_config as uc
import config as cfg
from dataloader.dataset_reader import DatasetReader
from model.model_factory import ModelFactory
from train.loss_factory import IntegratedLoss
import train.train_val as tv
import train.train_util as tu
from train.augmentation import augmentation_factory
from train.feature_generator import FeatureMapDistributer
from utils.snapshot_code import CodeSnapshot
from utils.strategy import DistributionStrategy, StrategyScope
import train.train_scheduler as ts


@StrategyScope
def train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, lr_hold):
    batch_size, anchors = cfg.Train.BATCH_SIZE, cfg.AnchorGeneration.ANCHORS
    data_path, ckpt_path = cfg.Paths.DATAPATH, op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)

    valid_category, start_epoch, augmenter, train_batch_size = prepare_train(dataset_name, ckpt_path, batch_size)
    strategy, val_batch_size, trainer_class = check_strategy(batch_size)

    if end_epoch <= start_epoch:
        print(f"!! end_epoch {end_epoch} <= start_epoch {start_epoch}, no need to train")
        return

    dataset_train, train_steps, imshape, anchors_per_scale = \
        get_dataset(data_path, dataset_name, True, train_batch_size, "train", anchors)
    dataset_val, val_steps, _, _ = get_dataset(data_path, dataset_name, False, val_batch_size, "val", anchors)

    model, loss_object, optimizer = create_training_parts(batch_size, imshape, anchors_per_scale, ckpt_path,
                                                          learning_rate, loss_weights, valid_category)
    feature_creator = FeatureMapDistributer(cfg.FeatureDistribPolicy.POLICY_NAME, anchors_per_scale)
    lrs = ts.Scheduler(learning_rate, cfg.Scheduler.CYCLE_STEPS, train_steps, ckpt_path,
                       warmup_epoch=cfg.Scheduler.WARMUP_EPOCH)

    trainer = trainer_class(model, loss_object, augmenter, optimizer, train_steps, feature_creator,
                            anchors_per_scale, strategy, ckpt_path)
    validater = tv.ModelValidater(model, loss_object, val_steps, feature_creator, anchors_per_scale, ckpt_path)

    for epoch in range(start_epoch, end_epoch):
        # dataset_train.shuffle(buffer_size=200)
        lrs.set_scheduler(lr_hold, epoch)
        print(f"========== Start dataset : {dataset_name} epoch: {epoch + 1}/{end_epoch} ==========")
        detail_log = (epoch in cfg.Train.DETAIL_LOG_EPOCHS)
        trainer.run_epoch(dataset_train, lrs, epoch)
        validater.run_epoch(dataset_val, None, epoch, detail_log, detail_log)
        save_model_ckpt(ckpt_path, model)
    save_model_ckpt(ckpt_path, model, f"ep{end_epoch:02d}")


def prepare_train(dataset_name, ckpt_path, batch_size):
    valid_category = uc.get_valid_category_mask(dataset_name)
    start_epoch = read_previous_epoch(ckpt_path)
    CodeSnapshot(ckpt_path, start_epoch)()
    augmenter = augmentation_factory(cfg.Train.AUGMENT_PROBS)
    if augmenter:
        train_batch_size = batch_size // 2
    else:
        train_batch_size = batch_size
    return valid_category, start_epoch, augmenter, train_batch_size


def check_strategy(batch_size):
    strategy = DistributionStrategy.get_strategy()
    if cfg.Train.MODE == "distribute":
        val_batch_size, trainer_class = (batch_size//strategy.num_replicas_in_sync, tv.ModelDistribTrainer)
    else:
        val_batch_size, trainer_class = (batch_size, tv.ModelTrainer)
    return strategy, val_batch_size, trainer_class


@StrategyScope
def create_training_parts(batch_size, imshape, anchors_per_scale, ckpt_path, learning_rate,
                          loss_weights, valid_category, weight_suffix='latest'):
    model = ModelFactory(batch_size, imshape, anchors_per_scale).get_model()
    model = try_load_weights(ckpt_path, model, weight_suffix)
    loss_object = IntegratedLoss(loss_weights, valid_category)
    optimizer = tf.optimizers.Adam(lr=learning_rate)
    if cfg.Train.USE_EMA:
        optimizer = tfa.optimizers.MovingAverage(optimizer, decay=cfg.Train.EMA_DECAY)
    # TODO if ppyolo maybe use SGD weight_decay = 5e-4, initial_weight=0.01
    # optimizer = tfa.optimizers.SGDW(weight_decay=5e-4, learning_rate=learning_rate, momentum=0.937)
    # optimizer = tf.optimizers.SGD(lr=learning_rate, momentum=0.937)
    return model, loss_object, optimizer


def save_model_ckpt(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
    if not op.isdir(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    print("=== save model:", ckpt_file)
    model.save_weights(ckpt_file)
    # model.save('./my_model')


def get_dataset(data_path, dataset_name, shuffle, batch_size, split, anchors):
    data_split_path = op.join(data_path, f"{dataset_name}_{split}")
    reader = DatasetReader(data_split_path, shuffle, batch_size, 1)
    dataset = reader.get_dataset()
    frames = reader.get_total_frames()
    dataset_cfg = reader.get_dataset_config()
    image_shape = dataset_cfg["image"]["shape"]
    # anchor sizes per scale in pixel
    if not anchors.all():
        anchors_per_scale = np.array([[[1, 1]], [[1, 1]], [[1, 1]]], dtype=np.float32)
    else:
        anchors_per_scale = np.array([anchor / np.array([image_shape[:2]]) for anchor in anchors], dtype=np.float32)
    print(f"[get_dataset] dataset={dataset_name}, image shape={image_shape}, "
          f"frames={frames},\n\tanchors={anchors_per_scale}")
    return dataset, frames // batch_size, image_shape, anchors_per_scale


def try_load_weights(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
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
