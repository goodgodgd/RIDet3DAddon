import config as cfg
import os.path as op
from dataloader.dataset_reader import DatasetReader
import numpy as np
from matplotlib import pyplot as plt


def get_dataset(split):
    reader = DatasetReader(op.join(cfg.Paths.DATAPATH, split))
    dataset = reader.get_dataset()
    tfr_cfg = reader.get_tfr_config()
    image_shape = tfr_cfg["image"]["shape"]
    return dataset, image_shape


def plot_hws(dataset):
    total_hs = []
    total_ws = []
    for feature in dataset:
        height = feature["bboxes"][..., 2].numpy()
        width = feature["bboxes"][..., 3].numpy()
        center_x = feature["bboxes"][..., 1].numpy()
        center_y = feature["bboxes"][..., 0].numpy()
        xmin = np.maximum(center_x - width / 2, 0)
        xmax = np.minimum(center_x + width / 2, 1)
        ymin = np.maximum(center_y - height / 2, 0)
        ymax = np.minimum(center_y + height / 2, 1)

        height = ymax - ymin
        width = xmax - xmin
        height = height[height > 0] * 512
        width = width[width > 0] * 1280

        total_hs.append(height)
        total_ws.append(width)
    total_h = np.concatenate(total_hs, axis=0)
    total_w = np.concatenate(total_ws, axis=0)
    return total_h, total_w


def draw_plot(h, w, anchors):
    plt.scatter(w, h, s=1, c=[0, 1, 0])
    color_list = ["red", "blue", "brown", "navy", "peru"]
    colors = []
    for i in range(len(cfg.ModelOutput.FEATURE_SCALES)):
        colors.append(color_list[i])
    for anchor, color in zip(anchors, colors):
        plt.scatter(anchor[:, 1], anchor[:, 0], s=1, color=color)
    plt.legend(("data_box", "anchor_3", "anchor_4", "anchor_5"))
    plt.xlabel("width")
    plt.ylabel("height")
    plt.savefig('./scatter_hw.png', dpi=200)


def log_tfrecord():
    target_tfrs = ("kitti_train", "kitti_val")
    datasets = []
    for tfr in target_tfrs:
        dataset, image_shape = get_dataset(tfr)
        datasets.append(dataset)

    anchor = np.array([[[43.0, 51.0], [123.0, 53.0], [77.0, 128.0]], [[51.0, 323.0], [254.0, 113.0], [165.0, 229.0]], [[85.0, 692.0], [92.0, 1079.0], [282.0, 395.0]]])
    dataset_h = []
    dataset_w = []
    for dataset in datasets:
        data_h, data_w = plot_hws(dataset)
        dataset_h.append(data_h)
        dataset_w.append(data_w)
    dataset_h = np.concatenate(dataset_h, axis=0)
    dataset_w = np.concatenate(dataset_w, axis=0)
    draw_plot(dataset_h, dataset_w, anchor)


if __name__ == "__main__":
    log_tfrecord()

