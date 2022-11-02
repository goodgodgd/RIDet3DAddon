import open3d as o3d
import os.path as op
import numpy as np
from glob import glob
import cv2

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

    def init_drive(self, drive_path, split):
        frame_names = glob(op.join(drive_path, f"{split}/image/*.png"))
        frame_names.sort()
        # frame_names = frame_names[1652:]

        print("[KittiReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def extract_2d_box(self, line):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]

        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None, None
        if category_name in self.dataset_cfg.CATEGORY_REMAP:
            category_name = self.dataset_cfg.CATEGORY_REMAP[category_name]
        y = round(float(raw_label[4]))
        x = round(float(raw_label[5]))
        l = round(float(raw_label[6]))
        w = round(float(raw_label[7]))

        tlbr = uf.convert_box_format_yxhw_to_tlbr(np.array([[y, x, l, w, 1]]))
        vaild = tlbr[:,:4] <3
        tlbr[:,:4] = tlbr[:,:4] + (vaild*3)
        bbox = uf.convert_box_format_tlbr_to_yxhw(tlbr)[0]

        return bbox.astype(np.int32), category_name

    def extract_3d_box(self, line):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]

        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None, None
        if category_name in self.dataset_cfg.CATEGORY_REMAP:
            category_name = self.dataset_cfg.CATEGORY_REMAP[category_name]

        lx = float(raw_label[8])
        ly = float(raw_label[9])
        lz = float(raw_label[10])

        h = float(raw_label[11])
        w = float(raw_label[12])
        l = float(raw_label[13])

        cx = float(raw_label[14])
        cy = float(raw_label[15])
        cz = float(raw_label[16])

        rotation_y = float(raw_label[17])

        bbox3d = np.array([lx, ly, lz, h, w, l, cx, cy, cz, rotation_y], dtype=np.float32)
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
