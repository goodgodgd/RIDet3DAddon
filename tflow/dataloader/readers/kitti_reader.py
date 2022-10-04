import os.path as op
import numpy as np
from glob import glob
import cv2

from dataloader.readers.reader_base import DatasetReaderBase, DriveManagerBase
import dataloader.data_util as tu
import utils.util_class as uc


class KittiDriveManager(DriveManagerBase):
    def __init__(self, datapath, split):
        super().__init__(datapath, split)

    def list_drive_paths(self):
        kitti_split = "training"    # if self.split == "train" else "testing"
        return [op.join(self.datapath, kitti_split, "image_2")]

    def get_drive_name(self, drive_index):
        return f"drive{drive_index:02d}"


class KittiReader(DatasetReaderBase):
    def __init__(self, drive_path, split, dataset_cfg):
        super().__init__(drive_path, split, dataset_cfg)

    def init_drive(self, drive_path, split):
        frame_names = glob(op.join(drive_path, "*.png"))
        frame_names.sort()
        # if split == "train":
        #     frame_names = frame_names[:-500]
        # else:
        #     frame_names = frame_names[-500:]
        print("[KittiReader.init_drive] # frames:", len(frame_names), "first:", frame_names[0])
        return frame_names

    def get_frame_name(self, index):
        return self.frame_names[index]

    def get_image(self, index):
        return cv2.imread(self.frame_names[index])

    def get_image_shape(self, index):
        return cv2.imread(self.frame_names[index]).shape[:2]

    def get_depth(self, index):
        image_file = self.frame_names[index]
        depth_file = image_file.replace("image_2", "depth_2").replace(".png", ".npy")
        depth = np.load(depth_file)
        depth = np.moveaxis(depth, [2, 3, 1, 0], [0, 1, 2, 3])[..., 0]
        return depth

    def get_intrinsic(self, index):
        calib_dict = dict()
        image_file = self.frame_names[index]
        calib_file = image_file.replace("image_2", "calib").replace(".png", ".txt")
        img_shape = self.get_image_shape(index)
        with open(calib_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                new_line = []
                line = line.split(" ")
                if len(line) == 1:
                    pass
                else:
                    line[0] = line[0].rstrip(":")
                    line[-1] = line[-1].rstrip("\n")
                    for a in line[1:]:
                        new_line.append(float(a))
                    calib_dict[line[0]] = new_line
            calib_dict["P2"] = np.reshape(np.array(calib_dict["P2"]), (3, 4))
            divider = np.array([[img_shape[1]], [img_shape[0]], [1]])
            calib_dict["P2"] /= divider

            return calib_dict["P2"].astype(np.float32)

    def get_bboxes(self, index, raw_hw_shape=None):
        """
        :return: bounding boxes in 'yxhw' format
        """
        image_file = self.frame_names[index]
        label_file = image_file.replace("image_2", "label_2").replace(".png", ".txt")

        bboxes2d = []
        bboxes3d = []
        categories = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                bbox2d, bbox3d, category = self.extract_box(line)
                if (bbox2d is not None) or (bbox3d is not None):
                    bboxes2d.append(bbox2d)
                    bboxes3d.append(bbox3d)
                    categories.append(category)
        if (not bboxes2d) or (not bboxes3d):
            raise uc.MyExceptionToCatch("[get_bboxes] empty boxes")
        bboxes2d = np.array(bboxes2d)
        bboxes3d = np.array(bboxes3d)
        return bboxes2d, bboxes3d, categories

    def extract_box(self, line):
        raw_label = line.strip("\n").split(" ")
        category_name = raw_label[0]
        if category_name not in self.dataset_cfg.CATEGORIES_TO_USE:
            return None, None, None
        y1 = round(float(raw_label[5]))
        x1 = round(float(raw_label[4]))
        y2 = round(float(raw_label[7]))
        x2 = round(float(raw_label[6]))

        h = raw_label[8]
        w = raw_label[9]
        l = raw_label[10]
        if (y2 - y1 < 1) or (x2 - x1 < 1):
            return None, None, None
        x3d = raw_label[11]
        y3d = raw_label[12]
        z = raw_label[13]
        theta = raw_label[14]
        bbox_2d = np.array([(y1 + y2) / 2, (x1 + x2) / 2, y2 - y1, x2 - x1, 1], dtype=np.float32)
        bbox_3d = np.array([y3d, x3d, z, h, w, l, theta], dtype=np.float32)
        return bbox_2d, bbox_3d, category_name

    def load_calib_data(self, file):
        calib_dict = {}
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                new_line = []
                line = line.split(" ")
                if len(line) == 1:
                    pass
                else:
                    line[0] = line[0].rstrip(":")
                    line[-1] = line[-1].rstrip("\n")
                    for a in line[1:]:
                        new_line.append(float(a))
                    calib_dict[line[0]] = new_line
        calib_dict["Tr_velo_to_cam"] = np.reshape(np.array(calib_dict["Tr_velo_to_cam"]), (3, 4))
        calib_dict["P0"] = np.reshape(np.array(calib_dict["P0"]), (3, 4))
        return calib_dict

    def load_velo_scan(self, file):
        """Load and parse a velodyne binary file."""
        scan = np.fromfile(file, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def get_point_cloud(self, velo_data, T2cam):
        velo_data[:, 3] = 1
        velo_in_camera = np.dot(T2cam, velo_data.T)
        velo_in_camera = velo_in_camera[:3].T
        # remove all velodyne points behind image plane
        velo_in_camera = velo_in_camera[velo_in_camera[:, 2] > 0]
        return velo_in_camera


def generate_depth_map(velo_data, T_cam_velo, K_cam, orig_shape, target_shape):
    # remove all velodyne points behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance(0)
    velo_data = velo_data[velo_data[:, 0] >= 0, :].T    # (N, 4) => (4, N)
    velo_data[3, :] = 1
    velo_in_camera = np.dot(T_cam_velo, velo_data)      # => (3, N)

    """ CAUTION!
    orig_shape, target_shape: (height, width) 
    velo_data[i, :] = (x, y, z)
    """
    targ_height, targ_width = target_shape
    orig_height, orig_width = orig_shape
    # rescale intrinsic parameters to target image shape
    K_prj = K_cam.copy()
    K_prj[0, :] *= (targ_width / orig_width)    # fx, cx *= target_width / orig_width
    K_prj[1, :] *= (targ_height / orig_height)  # fy, cy *= target_height / orig_height

    # project the points to the camera
    velo_pts_im = np.dot(K_prj, velo_in_camera[:3])         # => (3, N)
    velo_pts_im[:2] = velo_pts_im[:2] / velo_pts_im[2:3]    # (u, v, z)

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[0] = np.round(velo_pts_im[0]) - 1
    velo_pts_im[1] = np.round(velo_pts_im[1]) - 1
    valid_x = (velo_pts_im[0] >= 0) & (velo_pts_im[0] < targ_width)
    valid_y = (velo_pts_im[1] >= 0) & (velo_pts_im[1] < targ_height)
    velo_pts_im = velo_pts_im[:, valid_x & valid_y]

    # project to image
    depth = np.zeros(target_shape)
    depth[velo_pts_im[1].astype(np.int), velo_pts_im[0].astype(np.int)] = velo_pts_im[2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

    depth[depth < 0] = 0
    depth = depth[:, :, np.newaxis]
    # depth: (height, width, 1)
    return depth


def apply_color_map(depth):
    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    depth_view = (np.clip(depth, 0, 50.) / 50. * 255).astype(np.uint8)
    depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_VIRIDIS)
    depth_view[depth == 0, :] = (0, 0, 0)
    return depth_view


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


# ==================================================
import config as cfg


def test_kitti_depth():
    print("===== start test_kitti_reader")
    dataset_cfg = cfg.Datasets.Kitti
    drive_mngr = KittiDriveManager(dataset_cfg.PATH, "train")
    drive_paths = drive_mngr.get_drive_paths()
    reader = KittiReader(drive_paths[0], "train", dataset_cfg)
    max_list = []
    for i in range(reader.num_frames()):
        image = reader.get_image(i)
        # intrinsic = reader.get_intrinsic(i)
        # extrinsic = reader.get_stereo_extrinsic(i)
        depth = reader.get_depth(i, image.shape)
        depth_view = apply_color_map(depth)
        # depth = np.repeat(depth, 3, axis=-1)
        # print("depth:", depth.shape, depth.dtype)
        # max_list.append(depth.max())
        # print(depth.max())
        # depth = depth / 80.
        # depth = 1 - depth
        # depth[depth >= 1] = 0
        # depth[depth < 0] = 0
        # depth = depth * 255
        # depth = np.asarray(depth, dtype=np.uint8)

        # valid_depth = norm_depth[np.where(norm_depth > 0)]
        # valid_depth = (1 - valid_depth) * 255
        # valid_depth = np.asarray(valid_depth, dtype=np.uint8)
        # norm_depth[norm_depth > 0] = valid_depth
        print("image:", image.shape)
        total_image = np.concatenate([image, depth_view], axis=0)
        # cv2.imshow("kitti_image", depth_view)
        cv2.imshow("kitti_depth", total_image)
        # test = op.join("/home/eagle/mun_workspace/22_paper/kitti/training/gt_depth", op.split(reader.frame_names[i])[-1])
        # test = op.join("/home/eagle/mun_workspace/22_paper/kitti/testing/gt_depth_npy", op.split(reader.frame_names[i])[-1].replace(".png", ".npy"))
        # np.save(op.join("/home/eagle/mun_workspace/22_paper/kitti/testing/gt_depth_npy", op.split(reader.frame_names[i])[-1].replace(".png", ".npy")), depth)
        # cv2.imwrite(op.join("/home/eagle/mun_workspace/22_paper/kitti/training/gt_depth_image", op.split(reader.frame_names[i])[-1]), depth_view)
        key = cv2.waitKey()
        if key == ord('q'):
            break
    # print(max(max_list))


def test_kitti_reader():
    print("===== start test_kitti_reader")
    dataset_cfg = cfg.Datasets.Kitti
    drive_mngr = KittiDriveManager(dataset_cfg.PATH, "train")
    drive_paths = drive_mngr.get_drive_paths()
    reader = KittiReader(drive_paths[0], "train", dataset_cfg)
    for i in range(reader.num_frames()):
        image = reader.get_image(i)
        bboxes = reader.get_bboxes(i)
        print(f"frame {i}, bboxes:\n", bboxes)
        boxed_image = tu.draw_boxes(image, bboxes, dataset_cfg.CATEGORIES_TO_USE)
        cv2.imshow("kitti", boxed_image)
        key = cv2.waitKey()
        if key == ord('q'):
            break
    print("!!! test_kitti_reader passed")


if __name__ == "__main__":
    test_kitti_reader()
