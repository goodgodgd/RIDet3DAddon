import os
import numpy as np
import cv2
import open3d as o3d
import pandas as pd

import RIDet3DAddon.torch.config as cfg3d
import RIDet3DAddon.torch.config_dir.util_config as uc3d
from dataloader.readers.kitti_reader import KittiReader
import RIDet3DAddon.torch.dataloader.data_util as du
import utils.framework.util_function as uf


class KittiBevMaker(KittiReader):
    def __init__(self, drive_path, split, dataset_cfg=None):
        super(KittiBevMaker, self).__init__(drive_path, split, dataset_cfg)
        self.cell_size = self.dataset_cfg.CELL_SIZE
        self.xy_range = self.dataset_cfg.POINT_XY_RANGE
        self.img_shape = self.dataset_cfg.INPUT_RESOLUTION  # height, width
        self.tilt_angle = self.dataset_cfg.TILT_ANGLE

    def get_bev_image(self, index):
        image_file = self.frame_names[index]
        lidar_file = image_file.replace("image_2", "velodyne").replace(".png", ".bin")
        point_cloud = np.fromfile(lidar_file, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, 4))
        plane_model = self.get_ground_plane(point_cloud[:, :3])
        point_cloud_hrn = self.get_points_with_hrn(point_cloud, plane_model, self.tilt_angle)
        flpixels = self.pixel_coordinates(point_cloud_hrn, self.tilt_angle)
        result_depthmap = self.interpolation(flpixels, point_cloud_hrn, 3, 6)
        normal_bev = du.normalization(result_depthmap)
        result_bev = (normal_bev * 255).astype(np.uint8)
        return result_bev, point_cloud_hrn

    def get_bev_box(self, index):
        """
        return : annotation : {bev_y1, bev_x1, bev_y2, bev_x2, height,
                        dimensions_2, dimensions_3, dimensions_1,
                        location_1, location_2, rotation_y}
        """
        image_file = self.frame_names[index]
        lidar_file = image_file.replace("image_2", "velodyne").replace(".png", ".bin")
        point_cloud = np.fromfile(lidar_file, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, 4))

        # print("minmax", np.min(point_cloud, axis=0), np.max(point_cloud, axis=0))
        plane_model = self.get_ground_plane(point_cloud[:, :3])
        bbox3d, category_name = self.get_3d_box(index)
        calib = self.get_calibration(index)
        annotation = []
        ann_cata = []
        rotated_corners = []
        for box, cate in zip(bbox3d, category_name):
            ann, rotated_corner = self.convert_3d_to_bev(box, calib, plane_model)
            if np.max(ann) == 0:
                continue
            # box = np.delete(box, -2)
            # box = np.delete(box, 2)
            ann = np.concatenate([ann, box], axis=-1)
            annotation.append(ann)
            ann_cata.append(cate)
            rotated_corners.append(rotated_corner)

        if len(annotation) > 0:
            annotation = np.stack(annotation, axis=0)
            annotation = uf.convert_box_format_tlbr_to_yxhw(annotation)
            rotated_corners = np.stack(rotated_corners, axis=0)
            return annotation, ann_cata, rotated_corners
        return None, None, None

    def convert_3d_to_bev(self, box, calib, plane_model):
        pts_3d_ref = np.transpose(np.dot(np.linalg.inv(calib.R0), np.transpose(box[3:6][np.newaxis, ...])))
        n = pts_3d_ref.shape[0]
        pts_3d_ref = np.hstack((pts_3d_ref, np.ones((n, 1))))
        he = np.array([0, 0, plane_model[3]*3/2]).reshape([1, 3])
        centroid = np.dot(pts_3d_ref, np.transpose(calib.C2V)) + he

        corners = du.get_box3d_corner(box[0:3])

        R = du.create_rotation_matrix([box[-1], 0, 0])
        corners = np.dot(corners, R) + centroid
        corners = np.concatenate([corners, centroid], axis=0)
        mask = (corners[:, 0] > self.xy_range[0]) & \
               (corners[:, 0] < self.xy_range[1]) & \
               (corners[:, 1] > self.xy_range[2]) & \
               (corners[:, 1] < self.xy_range[3])
        corners = corners[mask, :]

        rotated_corners, normal_theta = self.get_rotation_and_normal_vector(corners, self.tilt_angle)
        height = self.cal_height(centroid, plane_model) * 2
        pixels = self.pixel_coordinates(rotated_corners[:, :2], self.tilt_angle)

        imshape = [self.img_shape[0], self.img_shape[1], 3]
        valid_mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < imshape[0] - 1) & \
                     (pixels[:, 1] >= 0) & (pixels[:, 1] < imshape[1] - 1)
        pixels = pixels[valid_mask, :]
        if pixels.size < 18:
            return np.array([0, 0, 0, 0, 0, 0, 0]), rotated_corners
            # return False
        xmin = np.min(pixels[:, 1])
        xmax = np.max(pixels[:, 1])
        ymin = np.min(pixels[:, 0])
        ymax = np.max(pixels[:, 0])
        height = height[0, 0]
        ann = np.array([ymin, xmin, ymax, xmax, rotated_corners[-1, 1],  rotated_corners[-1, 0], height])
        return ann, rotated_corners

    def get_ground_plane(self, points):
        points_vaild = (points[:, 2] < -1.0)  # & (points[:, 2] < -1.5)
        rote_pcd = o3d.geometry.PointCloud()
        rote_pcd.points = o3d.utility.Vector3dVector(points[points_vaild, :])
        plane_model, inliers = rote_pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=200)

        plane_model = np.array(plane_model)
        plane_model /= np.sum(np.power(plane_model[:3], 2))
        assert np.arccos(np.abs(plane_model[2])) < np.pi / 12
        return plane_model

    def get_points_with_hrn(self, point_cloud, plane_model, tilt_angle):
        mask = (point_cloud[:, 0] > self.xy_range[0]) & \
               (point_cloud[:, 0] < self.xy_range[1]) & \
               (point_cloud[:, 1] > self.xy_range[2]) & \
               (point_cloud[:, 1] < self.xy_range[3])
        point_cloud = point_cloud[mask, :]
        points = point_cloud[:, :3]
        lidar_height = plane_model[3]
        points[:, 2] += lidar_height
        
        tilted_points, normal_theta = self.get_rotation_and_normal_vector(points, tilt_angle)
        height = self.cal_height(points, tilt_angle)
        reflence = point_cloud[:, 3:4]

        # point_cloud_hrn shape : (N,6) :[tilted_points(x,y,z), points_z, reflence, normal_theta]
        point_cloud_hrn = np.concatenate([tilted_points, height, reflence, normal_theta], axis=1)
        return point_cloud_hrn

    def get_rotation_and_normal_vector(self, points, tilt_angle):
        """

        :param points: velodyne_points
        :param tilt_angle: axis-y rotation angle
        :return:
        tilted_points : transformation points with tilt_angle
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=np.pi, max_nn=5)
        pcd.estimate_normals(search_param)
        points_normals = np.asarray(pcd.normals)  #[:, 0:3:2]
        normal_theta = np.arctan2(points_normals[:, 0], points_normals[:, 2])
        normal_theta = normal_theta % (2 * np.pi)
        normal_theta = normal_theta[..., np.newaxis]
        pcd.rotate(pcd.get_rotation_matrix_from_xyz((0, tilt_angle, 0)), center=(0, 0, 0))
        tilted_points = np.asarray(pcd.points)[:, :3]
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=6)
        # o3d.visualization.draw_geometries([pcd, mesh_frame],
        #                                   zoom=0.3412,
        #                                   front=[0.0, -0.0, 1.0],
        #                                   lookat=[0.0, 0.0, 0.0],
        #                                   up=[-0.0694, -0.9768, 0.2024],
        #                                   point_show_normal=False)

        return tilted_points, normal_theta

    def cal_height(self, tilted_points, tilt_angle):
        height = tilted_points[:, 2] + tilted_points[:, 0] * np.tan(tilt_angle)
        height = height[..., np.newaxis]
        return height

    def pixel_coordinates(self, tilted_points, tilt_angle):
        """

        :param tilted_points: tilted_points(x,y)
        :param tilt_angle: rotation_y(default : 0)
        :return:
        """
        image_x = (self.img_shape[1] / 2) - (tilted_points[:, 1] / self.cell_size)
        image_y = (self.img_shape[0]) - (tilted_points[:, 0] / self.cell_size)
        pixels = np.stack([image_y, image_x], axis=1)
        return pixels

    def interpolation(self, pixels, points, start_idx, end_idx):
        """

        :param pixels: (N,2) float pixel coordinates (y, x)
        :param points: (N,6) [tilted_points(x,y,z), height, reflence, normal_theta]
        :return:
        """
        imshape = [self.img_shape[0], self.img_shape[1], end_idx - start_idx]
        valid_mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < imshape[0] - 1) & \
                     (pixels[:, 1] >= 0) & (pixels[:, 1] < imshape[1] - 1)
        points = points[valid_mask, :]
        pixels = pixels[valid_mask, :]
        pixels_df = pd.DataFrame(pixels.astype(int), columns=['y', 'x'])

        bev_img = np.zeros(imshape, dtype=np.float32)
        weights = np.zeros(imshape[:2], dtype=np.float32)
        step = 0
        while (len(pixels_df.index) > 0) and (step < 5):
            step += 1
            step_pixels = pixels_df.drop_duplicates(keep='first')
            rows = step_pixels['y'].values
            cols = step_pixels['x'].values
            inds = step_pixels.index.values
            bev_img[rows, cols, :] += points[inds, start_idx:end_idx]
            weights[rows, cols] += 1
            pixels_df = pixels_df[~pixels_df.index.isin(step_pixels.index)]

        bev_img /= weights[..., np.newaxis]
        return bev_img


def save_txt_dict(save_dirctory, dict, file_name):
    if not os.path.exists(save_dirctory):
        os.makedirs(save_dirctory)
    save_txt_file_name = os.path.join(save_dirctory, f'{file_name}.txt')

    if not os.path.exists(save_txt_file_name):
        with open(save_txt_file_name, "w") as f:
            for catagory, value in dict.items():
                data = f"{catagory}: {value}\n"
                f.write(data)


def save_txt(save_dirctory, lines, file_name):
    os.makedirs(save_dirctory, exist_ok=True)
    save_txt_file_name = os.path.join(save_dirctory, f'{file_name}.txt')
    # if not os.path.exists(save_txt_file_name):
    with open(save_txt_file_name, "w") as f:
        for line in lines:
            data = f"{' '.join(line)}\n"
            f.write(data)


def make_kitti_to_bev():
    path = cfg3d.Datasets.Kittibev.PATH
    tilt_angle = cfg3d.Datasets.Kittibev.TILT_ANGLE
    save_path = '/media/cheetah/IntHDD/kim_result'

    deg = int(tilt_angle * (180 / np.pi))
    print(deg)
    bev_make = KittiBevMaker(path, "all", cfg3d.Datasets.Kittibev)
    save_path_deg = os.path.join(save_path, 'deg_' + str(deg))
    save_image_path = save_path_deg + '/image_2'
    save_label_path = save_path_deg + '/label_2'

    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)
    num_frame = bev_make.num_frames()

    for i in range(num_frame):
        image_file = bev_make.frame_names[i]
        file_num = image_file.split('/')[-1].split('.')[0]
        image, _ = bev_make.get_bev_image(i)

        bev_box, category, _ = bev_make.get_bev_box(i)
        if bev_box is None:
            continue
        line = []
        for cate, box in zip(category, bev_box):
            box = box.astype(np.str)
            box = np.insert(box, 0, cate)
            line.append(box)
        save_image_file = os.path.join(save_image_path, f"{file_num}.png")
        bev_box = uf.convert_box_format_yxhw_to_tlbr(bev_box)
        image_view = du.draw_box(image, bev_box)
        cv2.imshow("ets",image_view)
        cv2.waitKey(100)
        cv2.imwrite(save_image_file, image)
        save_txt(save_label_path, line, file_num)
        uf.print_progress(f"{i}/{num_frame} deg {deg}")


if __name__ == '__main__':
    make_kitti_to_bev()
