import numpy as np
import os.path as op
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import config as cfg


class Scheduler:
    def __init__(self, base_learning_rate, cycle_steps, epoch_steps, ckpt_path, min_learning_rate=1e-10, cycle_ratio=1.,
                 height_ratio=1., warmup_epoch=0):
        self.learning_rate = base_learning_rate
        self.min_learning_rate = min_learning_rate / base_learning_rate
        self.cycle_steps = cycle_steps
        self.epoch_steps = epoch_steps
        self.ckpt_path = ckpt_path
        self.log_file = op.join(ckpt_path, "lrs.csv")
        self.cycle_ratio = cycle_ratio
        self.height_ratio = height_ratio
        self.warmup_epoch = warmup_epoch
        self.cur_epoch = 0
        self.lr_scheduler = ConstLRScheduler(self.learning_rate*0.01)
        self.lr_log = self.load_lr_data()

    def load_lr_data(self):
        if op.isfile(self.log_file):
            lr_data = pd.read_csv(self.log_file, encoding='utf-8', converters={'step': lambda c: int(c)})
        else:
            lr_data = pd.DataFrame()
        return lr_data

    def set_scheduler(self, learning_rate_hold, cur_epoch):
        self.cur_epoch = cur_epoch
        if self.warmup_epoch and self.warmup_epoch > cur_epoch:
            print("Warmup Learning Scheduler")
            self.lr_scheduler = ConstLRScheduler(self.learning_rate*0.01)
        elif learning_rate_hold:
            print("Hold Learning Rate")
            self.lr_scheduler = ConstLRScheduler(self.learning_rate)
        else:
            print("Cosine Learning Scheduler")
            self.lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(self.learning_rate, self.cycle_steps,
                                                                                  self.cycle_ratio, self.height_ratio,
                                                                                  self.min_learning_rate)

    def __call__(self, step):
        global_step = (self.cur_epoch - self.warmup_epoch) * self.epoch_steps + step
        lr = self.lr_scheduler(global_step)
        lr_value = {"lr": lr}
        self.lr_log = self.lr_log.append(lr_value, ignore_index=True)
        return lr

    def save_log(self):
        self.lr_log.to_csv(self.log_file, encoding='utf-8', index=False,)
        self.draw()

    def draw(self):
        plt.plot(self.lr_log)
        plt.title('Linear learning rate')
        plt.xlabel('step')
        plt.yscale("log", base=10)
        plt.ylabel('learning_rate')
        plt.grid()
        plt.savefig(op.join(self.ckpt_path, "lrs_fig.png"))
        plt.clf()


class ConstLRScheduler:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, step):
        return self.lr


# ==================================================


def main():
    end_epoch = 0
    start_epoch = 0
    data_step = 500
    cycle = 1000
    data = []
    plans = [(10, 0.0001, False),
             # (15, 0.00001, True),
             (10, 0.00001, False),
             # (10, 0.000001, False),
             (10, 0.000001, False)]
    for num, (epochs, lr, hold) in enumerate(plans):
        end_epoch += epochs
        lrs = Scheduler(lr, cycle, data_step, ".", warmup_epoch=5, cycle_ratio=1., height_ratio=1.)
        for i in range(start_epoch, end_epoch):
            lrs.set_scheduler(hold, i)
            for step in range(data_step):
                learning_rate = lrs(step)
                data.append(learning_rate)
        lrs.draw()
        start_epoch += epochs

    plt.plot(data)
    plt.title('Warmup_Cosine learning rate')
    plt.xlabel('step')
    plt.ylabel('learning_rate')
    plt.grid()
    plt.savefig(f"./lrs.png")



if __name__ == "__main__":
    main()
