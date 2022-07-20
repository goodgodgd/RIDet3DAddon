import settings
from RIDet3DAddon.tflow.train.train_plan import train_by_plan
import numpy as np
import utils.tflow.util_function as uf

import RIDet3DAddon.config as cfg3d


def train_main():
    uf.set_gpu_configs()
    end_epoch = 0
    for dataset_name, epochs, learning_rate, loss_weights, lr_hold in cfg3d.Train.TRAINING_PLAN:
        end_epoch += epochs
        train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, lr_hold)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    train_main()
