import numpy as np

import dataloader.framework.data_util as tu

import RIDet3DAddon.torch.dataloader.preprocess as pr
import config as cfg
import RIDet3DAddon.torch.config as cfg3d


class ExampleMaker:
    def __init__(self, data_reader, dataset_cfg, split,
                 feat_scales=cfg.ModelOutput.FEATURE_SCALES,
                 category_names=cfg3d.Dataloader.CATEGORY_NAMES,
                 max_bbox=cfg.Dataloader.MAX_BBOX_PER_IMAGE,
                 max_dontcare=cfg.Dataloader.MAX_DONT_PER_IMAGE):
        self.data_reader = data_reader
        self.feat_scales = feat_scales
        self.category_names = category_names
        self.anchors_lane = cfg.Dataloader.ANCHORS_LANE
        self.lane_detect_rows = cfg.Dataloader.LANE_DETECT_ROWS
        self.max_bbox = max_bbox
        self.preprocess_example = pr.ExamplePreprocess(target_hw=dataset_cfg.INPUT_RESOLUTION,
                                                       dataset_cfg=dataset_cfg,
                                                       max_bbox=max_bbox,
                                                       max_dontcare=max_dontcare,
                                                       min_pix=cfg3d.Dataloader.MIN_PIX[split],
                                                       category_names=category_names
                                                       )

    def get_example(self, index):
        example = dict()
        example["image"] = self.data_reader.get_image(index)
        raw_hw_shape = example["image"].shape[:2]
        box2d, categories = self.data_reader.get_2d_box(index, raw_hw_shape)
        box3d, _ = self.data_reader.get_3d_box(index)

        example["inst2d"], example["inst3d"], example["dontcare"] = self.merge_box_and_category(box2d, box3d, categories)
        example = self.preprocess_example(example)
        example["frame_names"] = self.data_reader.frame_names[index]
        # if index % 100 == 10:
        #     self.show_example(example)
        return example

    def extract_bbox(self, example):
        scales = [key for key in example if "feature" in key]
        # merge pred features over scales
        total_features = []
        for scale_name in scales:
            height, width, anchor, channel = example[scale_name].shape
            merged_features = np.reshape(example[scale_name], (height * width * anchor, channel))
            total_features.append(merged_features)

        total_features = np.concatenate(total_features, axis=0)  # (batch, N, dim)
        total_features = total_features[total_features[..., 4] > 0]
        if total_features.size != 0:
            num_box = total_features.shape[0]
            pad_num = np.maximum(self.max_bbox - num_box, 0)
            zero_pad = np.zeros((pad_num, total_features.shape[-1]), dtype=np.float32)
            example["bboxes"] = np.concatenate([total_features[:self.max_bbox, :], zero_pad])
        return example

    def merge_box_and_category(self, bboxes2d, bboxes3d, categories):
        reamapped_categories = []
        for category_str in categories:
            if category_str in self.category_names["category"]:
                major_index = self.category_names["category"].index(category_str)
            elif category_str in self.category_names["dont"]:
                major_index = -1
            else:
                major_index = -2
            reamapped_categories.append(major_index)
        reamapped_categories = np.array(reamapped_categories)[..., np.newaxis]
        # bbox: yxhw, obj, ctgr (6)
        bboxes2d = np.concatenate([bboxes2d, reamapped_categories], axis=-1)
        bboxes3d = np.concatenate([bboxes3d, reamapped_categories], axis=-1)
        dontcare = bboxes2d[bboxes2d[..., -1] == -1]
        bboxes2d = bboxes2d[bboxes2d[..., -1] >= 0]
        bboxes3d = bboxes3d[bboxes3d[..., -1] >= 0]
        return bboxes2d, bboxes3d, dontcare


    def show_example(self, example):
        image = tu.draw_boxes(example["image"], example["inst2d"], self.category_names)
        if self.include_lane:
            image = tu.draw_lanes(image, example["lanes"], self.category_names)

        # cv2.imshow("image with feature bboxes", image)
        # cv2.waitKey(100)

