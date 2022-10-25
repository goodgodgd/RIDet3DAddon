import numpy as np
import cv2
import open3d as o3d

import RIDet3DAddon.torch.config as cfg
from dataloader.readers.kitti_reader import KittiReader
import utils.framework.util_function as uf


class KittiBevReader(KittiReader):
    def __init__(self, drive_path, split, dataset_cfg=None):
        super(KittiBevReader, self).__init__(drive_path, split, dataset_cfg)
        self.cell_size = self.dataset_cfg.CELL_SIZE
        self.xy_range = self.dataset_cfg.POINT_XY_RANGE
        self.img_shape = self.dataset_cfg.INPUT_RESOLUTION  # height, width
        self.tilt_angle = self.dataset_cfg.TILT_ANGLE

    # def get_bev_box(self, index):
    def extract_2d_box(self, line):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]

        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None, None
        if category_name in self.dataset_cfg.CATEGORY_REMAP:
            category_name = self.dataset_cfg.CATEGORY_REMAP[category_name]
        y = round(float(raw_label[1]))
        x = round(float(raw_label[2]))
        l = round(float(raw_label[3]))
        w = round(float(raw_label[4]))
        bbox = np.array([y, x, l, w, 1], dtype=np.int32)
        return bbox, category_name

    def extract_3d_box(self, line):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]

        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None, None
        if category_name in self.dataset_cfg.CATEGORY_REMAP:
            category_name = self.dataset_cfg.CATEGORY_REMAP[category_name]


        tilt_y = float(raw_label[5])
        tilt_x = float(raw_label[6])
        height = float(raw_label[7])

        w = float(raw_label[8])
        l = float(raw_label[9])
        h = float(raw_label[10])
        # x = float(raw_label[11])
        # y = float(raw_label[12])
        # z = float(raw_label[13])
        x = float(raw_label[13])
        y = - float(raw_label[11])
        z = - float(raw_label[12])

        rotation_y = float(raw_label[14])

        bbox3d = np.array(
            [x, y, z, l, w, h, rotation_y, ],
            dtype=np.float32)
        return bbox3d, category_name


def drow_box(img, bbox):
    # bbox = bbox.detach().numpy()
    print(bbox)

    x0 = int(bbox[1])
    x1 = int(bbox[3])
    y0 = int(bbox[0])
    y1 = int(bbox[2])
    img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
    return img


def draw_rotated_box(img, corners):
    """
    corners :
    """
    color = (255, 255, 255)

    corner_idxs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for corner_idx in corner_idxs:
        cv2.line(img,
                 (int(corners[corner_idx[0], 0]),
                  int(corners[corner_idx[0], 1])),
                 (int(corners[corner_idx[1], 0]),
                  int(corners[corner_idx[1], 1])),
                 color, 2)
    return img


def show_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=6)
    o3d.visualization.draw_geometries([pcd, mesh_frame],
                                      zoom=0.3412,
                                      front=[0.0, -0.0, 1.0],
                                      lookat=[0.0, 0.0, 0.0],
                                      up=[-0.0694, -0.9768, 0.2024],
                                      point_show_normal=False)


def test_kitti_bev():
    # path = "/media/dolphin/intHDD/kim_result/deg_0/image_2"
    path = cfg.Paths.DATAPATH
    # drive_path, split, dataset_cfg=None, cell_size=0.05, grid_shape=500, tbev_pose=0
    kitti_bev = KittiBevReader(path, "train", dataset_cfg=cfg.Datasets.Kittibev)
    num_frame = kitti_bev.num_frames()
    for i in range(num_frame):
        image = kitti_bev.get_image(i)
        bev_box, _ = kitti_bev.get_2d_box(i)
        bev_box = uf.convert_box_format_yxhw_to_tlbr(bev_box)
        print(bev_box.shape)
        if bev_box is not False:
            print(image.shape)
            for dd in bev_box:
                image = drow_box(image, dd)
            print(image.shape)
            cv2.imshow("i", image)
            cv2.waitKey(100)


if __name__ == '__main__':
    test_kitti_bev()
