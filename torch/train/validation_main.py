
import os.path as op
import numpy as np
import pandas as pd

import RIDet3DAddon.torch.config as cfg3d
from RIDet3DAddon.torch.dataloader.dataset_reader import DatasetReader
from RIDet3DAddon.torch.model.model_factory import ModelFactory
from RIDet3DAddon.torch.train.loss_factory import IntegratedLoss
import RIDet3DAddon.torch.train.train_val as tv
import train.framework.train_util as tu
from RIDet3DAddon.torch.train.feature_generator import FeatureMapDistributer
import RIDet3DAddon.torch.config_dir.util_config as uc3d
import utils.framework.util_function as uf

import config as cfg


def validate_main():
    uf.set_gpu_configs()
    ckpt_path = op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    latest_epoch = read_previous_epoch(ckpt_path)
    val_epoch = cfg.Validation.VAL_EPOCH
    weight_suffix = val_epoch if isinstance(val_epoch, str) else f"ep{val_epoch:02d}"
    target_epoch = latest_epoch if isinstance(val_epoch, str) else val_epoch
    start_epoch = 0

    for dataset_name, epochs, learning_rate, loss_weights, model_save in cfg3d.Train.TRAINING_PLAN:
        print("Start")
        if start_epoch <= target_epoch <= start_epoch + target_epoch:
            analyze_performance(dataset_name, loss_weights, weight_suffix, latest_epoch)
            start_epoch += epochs


def analyze_performance(dataset_name, loss_weights, weight_suffix, latest_epoch):
    batch_size, train_mode, anchors = cfg.Train.BATCH_SIZE, cfg.Train.MODE, cfg.AnchorGeneration.ANCHORS
    datapath, ckpt_path = cfg3d.Paths.DATAPATH, op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    valid_category = uc3d.get_valid_category_mask(dataset_name)

    dataset_val, val_steps, imshape, anchors_per_scale \
        = get_dataset(datapath, dataset_name, False, batch_size, "val", anchors)
    feature_creator = FeatureMapDistributer(cfg3d.FeatureDistribPolicy.POLICY_NAME, imshape, anchors_per_scale)

    model = ModelFactory(batch_size, imshape, anchors_per_scale, training=None).get_model()
    model = try_load_weights(ckpt_path, model, weight_suffix)
    loss_object = IntegratedLoss(loss_weights, valid_category)
    validater = tv.ModelValidater(model, loss_object, val_steps, feature_creator, ckpt_path)

    print(f"========== Start analyze_performance with {dataset_name} epoch: {weight_suffix} ==========")
    validater.run_epoch(dataset_val, latest_epoch, True, True, val_only=True)


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


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    validate_main()
