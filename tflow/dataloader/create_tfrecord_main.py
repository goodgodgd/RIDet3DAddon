import os.path as op
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import config_dir.config_generator as cg
import config as cfg
from dataloader.dataset_writer import TfrecordMaker


def create_tfrecords():
    datasets = cfg.Dataloader.DATASETS_FOR_TFRECORD
    for dataset, splits in datasets.items():
        for split in splits:
            dataset_cfg = cg.set_dataset_and_get_config(dataset)
            tfrpath = op.join(cfg.Paths.DATAPATH, f"{dataset}_{split}")
            if op.isdir(tfrpath):
                print("[convert_to_tfrecords] dataloader already created in", op.basename(tfrpath))
                continue

            tfrmaker = TfrecordMaker(dataset_cfg, split, tfrpath, cfg.Dataloader.SHARD_SIZE)
            tfrmaker.make()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    create_tfrecords()
