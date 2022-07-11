import os.path as op
import json
import tensorflow as tf
import cv2

import config as cfg
import utils.util_function as uf


class DatasetReader:
    def __init__(self, datapath, shuffle=False, batch_size=cfg.Train.BATCH_SIZE, epochs=1):
        self.datapath = datapath
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs
        self.config = self.read_dataset_config(datapath)
        self.features_dict = self.get_features(self.config)

    def read_dataset_config(self, datapath):
        with open(op.join(datapath, "tfr_config.txt"), "r") as fr:
            config = json.load(fr)
            for key, properties in config.items():
                if not isinstance(properties, dict):
                    continue
                # convert parse types in string to real type
                properties["parse_type"] = eval(properties["parse_type"])
                properties["decode_type"] = eval(properties["decode_type"])
        return config

    def get_features(self, config):
        features_dict = {}
        for key, properties in config.items():
            if isinstance(properties, dict):
                default = "" if properties["parse_type"] is tf.string else 0
                features_dict[key] = tf.io.FixedLenFeature((), properties["parse_type"], default_value=default)
        return features_dict

    def get_dataset(self):
        """
        :return features: {"image": ..., "bboxes": ...}
            image: (batch, height, width, 3)
            bbox: [y1, x1, y2, x2, category] (batch, grid_height, grid_width, 5)
        """
        file_pattern = f"{self.datapath}/*.tfrecord"
        filenames = tf.io.gfile.glob(file_pattern)
        filenames.sort()
        print("[tfrecord reader] pattern:", file_pattern, "\nfiles:", filenames)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_example)
        return self.dataset_process(dataset)

    def parse_example(self, example):
        parsed = tf.io.parse_single_example(example, self.features_dict)
        decoded = {}
        for key, properties in self.config.items():
            if isinstance(properties, dict):
                decoded[key] = tf.io.decode_raw(parsed[key], properties["decode_type"])
                decoded[key] = tf.reshape(decoded[key], shape=properties["shape"])
        # uint8 image -> image float (-1 ~ 1)
        decoded["image"] = uf.to_float_image(decoded["image"])
        return decoded

    def dataset_process(self, dataset):
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=200)
            print("[dataset] dataset shuffled")
        print(f"[dataset] num epochs={self.epochs}, batch size={self.batch_size}")
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        return dataset

    def get_total_frames(self):
        return self.config["length"]

    def get_dataset_config(self):
        return self.config


# ==================================================
import numpy as np
import dataloader.data_util as tu


def test_read_dataset():
    print("===== start test_read_dataset")
    dataset = DatasetReader(op.join(cfg.Paths.DATAPATH, "kitti_train")).get_dataset()
    for i, x in enumerate(dataset):

        continue
        image = uf.to_uint8_image(x['image'])
        image = image[0].numpy()
        bboxes = x['bboxes'][0].numpy()
        image = tu.draw_boxes(image, bboxes, cfg.Dataloader.MAJOR_CATE)
        cv2.imshow("image with boxes", image)

        features = []
        for feat_name in cfg.ModelOutput.FEATURE_ORDER:
            feature = x[feat_name][0].numpy()
            feature = feature[feature[..., 4] > 0]
            features.append(feature)
        feat_boxes = np.concatenate(features, axis=0)
        # image = tu.draw_boxes(image, feat_boxes, cfg.Dataloader.MAJOR_CATE)
        cv2.imshow("image with feature bboxes", image)
        key = cv2.waitKey()
        if key == ord('q'):
            break

    print("!!! test_read_dataset passed")


def test_load_dataset():
    print("===== start test_read_dataset")
    dataset = DatasetReader(op.join(cfg.Paths.DATAPATH, "kitti_train")).get_dataset()
    for i, x in enumerate(dataset):
        print(f"=== index: {i}, image={x['image'].shape}, bbox={x['bboxes'].shape}"
              f", feature_l={x['feature_l'].shape}, feature_s={x['feature_s'].shape}")
        print("dontcare", x["dontcare"][0, :10])


if __name__ == "__main__":
    # test_read_dataset()
    test_load_dataset()
