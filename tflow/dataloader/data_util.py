import numpy as np
import cv2
import tensorflow as tf
import pandas as pd

import utils.tflow.util_function as uf


class TfrSerializer:
    def __call__(self, raw_example):
        features = self.convert_to_feature(raw_example)
        # wrap the data as TensorFlow Features.
        features = tf.train.Features(feature=features)
        # wrap again as a TensorFlow Example.
        tf_example = tf.train.Example(features=features)
        # serialize the data.
        serialized = tf_example.SerializeToString()
        return serialized

    def convert_to_feature(self, raw_example):
        features = dict()
        for key, value in raw_example.items():
            if value is None:
                continue
            elif isinstance(value, np.ndarray):
                # method 1: encode into raw bytes - fast but losing shape, 2 seconds to make training dataset
                value = value.tobytes()
                # method 2: encode into png format - slow but keeping shape, 10 seconds to make training dataset
                # value = tf.io.encode_png(value)
                # value = value.numpy()  # BytesList won't unpack a tf.string from an EagerTensor.
                features[key] = self._bytes_feature(value)
            elif isinstance(value, str):
                value = bytes(value, 'utf-8')
                features[key] = self._bytes_feature(value)
            elif isinstance(value, int):
                features[key] = self._int64_feature(value)
            elif isinstance(value, float):
                features[key] = self._float_feature(value)
            else:
                assert 0, f"[convert_to_feature] Wrong data type: {type(value)}"
        return features

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def inspect_properties(example):
    config = dict()
    for key, value in example.items():
        if value is not None:
            config[key] = read_data_config(key, value)
    return config


def read_data_config(key, value):
    parse_type = ""
    decode_type = ""
    shape = ()
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            decode_type = "tf.uint8"
        elif value.dtype == np.int32:
            decode_type = "tf.int32"
        elif value.dtype == np.float32:
            decode_type = "tf.float32"
        else:
            assert 0, f"[read_data_config] Wrong numpy type: {value.dtype}, key={key}"
        parse_type = "tf.string"
        shape = list(value.shape)
    elif isinstance(value, int):
        parse_type = "tf.int64"
        shape = None
    elif isinstance(value, str):
        parse_type = "tf.string"
        shape = None
    else:
        assert 0, f"[read_data_config] Wrong type: {type(value)}, key={key}"

    return {"parse_type": parse_type, "decode_type": decode_type, "shape": shape}


def draw_boxes(image, bboxes, category_names, locations=None, box_format="yxhw"):
    """
    :param grid_feats:
    :param image: (height, width, 3), np.uint8
    :param bboxes: (N, 6), np.float32 (0~1) or np.int32 (pixel scale)
    :param category_names: list of category names
    :param box_format: "yxhw": [y, x, h, w, category] or "2pt": [y1, x1, y2, x2, category]
    """
    image = image.copy()
    bboxes = bboxes.copy()
    if np.max(bboxes[:, :4]) <= 1:
        height, width = image.shape[:2]
        bboxes[:, :4] *= np.array([[height, width, height, width]], np.float32)
    if box_format == "yxhw":
        bboxes = uf.convert_box_format_yxhw_to_tlbr(bboxes)
    bboxes = bboxes[bboxes[:, 2] > 0, :]

    for i, bbox in enumerate(bboxes):
        pt1, pt2 = (bbox[1].astype(np.int32), bbox[0].astype(np.int32)), (bbox[3].astype(np.int32), bbox[2].astype(np.int32))
        category_index = int(bbox[-1])
        image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), thickness=2)
        image = cv2.putText(image, f"{i}{category_names['category'][category_index]}", pt1, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

    if locations is not None:
        for i, location in enumerate(locations):
            y, x = location
            image = cv2.circle(image, (int(x), int(y)), 4, (255, 0, 255), -1)
    return image


def check_locations_in_grid(locations, grid_ratio):
    grid_shape = locations.shape[:2]
    rows = np.arange(0, grid_shape[1], 1) + 0.5
    cols = np.arange(0, grid_shape[0], 1) + 0.5
    cols = cols.T
    grid_x, grid_y = np.meshgrid(rows, cols)
    grid = np.stack([grid_y, grid_x], axis=-1)
    grid = np.repeat(np.expand_dims(grid, 2), 3, axis=2)

    feat_location = (grid / np.array(grid_shape)) * locations
    feat_location_pixs = feat_location * np.array(grid_shape) * grid_ratio
    feat_location_pixs = feat_location_pixs[feat_location_pixs[..., 0] > 0]
    return feat_location_pixs


def depth_map_to_point_cloud(depth_map, intrinsic):
    # make sure center point is in depth center area, in order to check image scale
    assert np.abs(depth_map.shape[0] - intrinsic[1, 2] * 2) / depth_map.shape[0] < 0.5, \
        f"depth height={depth_map.shape[0]}, cy={intrinsic[1, 2]}"
    u_grid, v_grid = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
    if depth_map.size > 1e6:
        # if point cloud is too lage, use only quarter of depths
        depth_map[0:-1:2, :] = 0.
        depth_map[:, 0:-1:2] = 0.
    Z = depth_map.reshape(-1)
    # X = (u - cx) / fx * Z
    X = (u_grid.reshape(-1) - intrinsic[0, 2]) / intrinsic[0, 0] * Z
    # Y = (v - cy) / fy * Z
    Y = (v_grid.reshape(-1) - intrinsic[1, 2]) / intrinsic[1, 1] * Z
    points = np.stack([X, Y, Z], axis=1)
    points = points[Z > 0.1]
    return points


def point_cloud_to_depth_map(src_pcd, intrinsic, imshape):
    """
    :param src_pcd: source point cloud [N, 3] (X=right, Y=down, Z=front)
    :param intrinsic: [3, 3]
    :param imshape: height and width of output depth map
    :return: depth map
    """
    # print("[pcd2depth]", src_pcd.shape, intrinsic.shape, imshape)
    points = src_pcd[src_pcd[:, 2] > 1.].T  # [3, N]
    # project to camera, pixels: [3, N]
    pixels = np.dot(intrinsic, points) / points[2:3]
    assert np.isclose(pixels[2], 1.).all()
    # remove pixels out of image plane
    valid_mask = (pixels[0]>=0) & (pixels[0]<imshape[1]-1) & (pixels[1]>=0) & (pixels[1]<imshape[0]-1)
    pixels = pixels[:, valid_mask]
    points = points[:, valid_mask]
    # verify pixel-point relationship
    leftup = points[:, (pixels[1] > intrinsic[1, 2]-20) & (pixels[1] < intrinsic[1, 2]-10) & (pixels[0] < 50)]
    righdw = points[:, (pixels[1] > intrinsic[1, 2]+30) & (pixels[1] < intrinsic[1, 2]+40) & (pixels[0] > imshape[1]-50)]
    if leftup.size > 0: assert (np.mean(leftup[:2], axis=1) < 0).all(), f"{leftup}"
    if righdw.size > 0: assert (np.mean(righdw[:2], axis=1) > 0).all(), f"{righdw}"
    # quarter pixels around `pixels`
    data = np.stack([np.floor(pixels[0]), np.floor(pixels[1]), np.ceil(pixels[0]), np.ceil(pixels[1])], axis=1)
    quart_pixels = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2'])
    quart_pixels = quart_pixels.astype(int)
    quarter_columns = [['x1', 'y1'], ['x1', 'y2'], ['x2', 'y1'], ['x2', 'y2']]
    depthmap = np.zeros(imshape, dtype=np.float32)
    weightmap = np.zeros(imshape, dtype=np.float32)
    flpixels = pixels[:2]

    for quarter_col in quarter_columns:

        qtpixels = quart_pixels.loc[:, quarter_col]
        qtpixels = qtpixels.rename(columns={quarter_col[0]: 'col', quarter_col[1]: 'row'})
        # diff = (1-abs(x-xn), 1-abs(y-yn)) [N, 2]
        diff = 1 - np.abs(flpixels.T - qtpixels.values)
        # weights = (1-abs(x-xn)) * (1-abs(y-yn)) [N]
        weights = diff[:, 0] * diff[:, 1]

        step = 0
        while (len(qtpixels.index) > 0) and (step < 5):
            step += 1
            step_pixels = qtpixels.drop_duplicates(keep='first')
            rows = step_pixels['row'].values
            cols = step_pixels['col'].values
            inds = step_pixels.index.values
            depthmap[rows, cols] += points[2, inds] * weights[inds]
            weightmap[rows, cols] += weights[inds]
            qtpixels = qtpixels[~qtpixels.index.isin(step_pixels.index)]

    depthmap[depthmap > 0] = depthmap[depthmap > 0] / weightmap[depthmap > 0]
    depthmap[weightmap < 0.5] = 0
    return depthmap.astype(np.float32)