import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataloader.readers.kitti_reader import KittiReader
import dataloader.framework.data_util as du
from RIDet3DAddon.torch.dataloader.example_maker import ExampleMaker
import RIDet3DAddon.torch.config_dir.util_config as uc3d
import RIDet3DAddon.torch.config as cfg3d
import config as cfg

from RIDet3DAddon.torch.dataloader.readers.kitti_bev_reader import KittiBevReader


class DatasetAdapter(Dataset, ExampleMaker):
    def __init__(self, data_reader, dataset_cfg, split,
                 feat_scales=cfg.ModelOutput.FEATURE_SCALES,
                 category_names=cfg3d.Dataloader.CATEGORY_NAMES,
                 max_bbox=cfg.Dataloader.MAX_BBOX_PER_IMAGE,
                 max_dontcare=cfg.Dataloader.MAX_DONT_PER_IMAGE

                 ):
        Dataset.__init__(self)
        ExampleMaker.__init__(self, data_reader, dataset_cfg, split, feat_scales,
                              category_names, max_bbox, max_dontcare)

    def __len__(self):
        return self.data_reader.num_frames()

    def __getitem__(self, index):
        return self.get_example(index)


class DatasetReader:
    def __init__(self, ds_name, data_path, split, shuffle=False, batch_size=cfg.Train.BATCH_SIZE, epochs=1):
        self.data_reader = self.reader_factory(ds_name, data_path, split)
        self.data_path = data_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs

    def get_total_frames(self):
        return len(self.data_reader)

    def get_dataset(self):
        data_loader = DataLoader(dataset=self.data_reader, shuffle=self.shuffle, batch_size=self.batch_size,
                                 drop_last=True, num_workers=0)
        return data_loader

    def reader_factory(self, ds_name, data_path, split):
        if ds_name == "kitti":
            data_reader = KittiReader(data_path, split, cfg.Datasets.Kitti)
        elif ds_name == "kittibev":
            data_reader = KittiBevReader(data_path, split, cfg3d.Datasets.Kittibev)
        else:
            data_reader = None
        dataset_cfg = uc3d.set_dataset_and_get_config(ds_name)
        data_reader = DatasetAdapter(data_reader, dataset_cfg, split)
        return data_reader


def test_read_dataset():
    path = "/media/dolphin/intHDD/kitti_detection/data_object_image_2/training/image_2"
    reader = DatasetReader("kittibev", path, "train", False, 2, 1)
    dataset = reader.get_dataset()
    dataset_cfg = cfg.Datasets.Kitti
    for i, features in enumerate(dataset):
        print('features', type(features))
        # for key, val in features.items():
        image = features["image"].detach().numpy().astype(np.uint8)[0]
        boxes2d = features["bbox2d"].detach().numpy()[0]
        print(image.shape)
        print(boxes2d.shape)
        # boxes_3d = uf.convert_box_format_yxhw_to_tlbr(boxes_3d[:, :4])
        boxed_image = du.draw_boxes(image, boxes2d, dataset_cfg.CATEGORIES_TO_USE)
        cv2.imshow("KITTI", boxed_image)
        key = cv2.waitKey(0)
        # image_i = image.copy()
        # box_image_3d = du.draw_boxes(image_i, boxes_3d, category_names=None)
        # cv2.imshow('img', box_image)
        # cv2.imshow('img_2', box_image_3d)
        # cv2.waitKey()


if __name__ == '__main__':
    test_read_dataset()
