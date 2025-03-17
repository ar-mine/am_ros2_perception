import math
import numpy as np
from PIL import Image
import scipy.io as scio
from scipy.spatial.transform import Rotation
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup
from typing import Optional

from .graspness_implementation.models.graspnet import GraspNet, pred_decode
from .graspness_implementation.dataset.graspnet_dataset import minkowski_collate_fn
from .graspness_implementation.utils.collision_detector import ModelFreeCollisionDetector
from .graspness_implementation.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image

class GraspPoseDetector:
    def __init__(self, checkpoint_path: str,
                 seed_feat_dim: int=512,
                 num_point: int=15000,
                 voxel_size: float=0.005,
                 collision_thresh: float=0.005,
                 voxel_size_cd: float=0.01):
        self._checkpoint_path = checkpoint_path
        self._seed_feat_dim = seed_feat_dim
        self._num_point = num_point
        self._voxel_size = voxel_size
        self._collision_thresh = collision_thresh
        self._voxel_size_cd = voxel_size_cd

        # Init Grasp net
        self._net = GraspNet(seed_feat_dim=self._seed_feat_dim, is_training=False)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._net.to(self._device)
        # Load checkpoint
        checkpoint = torch.load(self._checkpoint_path)
        self._net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (self._checkpoint_path, start_epoch))
        self._net.eval()

    def data_process(self, depth_image: np.ndarray,
                     intrinsic: np.ndarray,
                     resolution: np.ndarray,
                     factor_depth: float=1000.0,
                     rgb_image: Optional[np.ndarray]=None,
                     seg_image: Optional[np.ndarray]=None):

        camera = CameraInfo(resolution[0], resolution[1],
                            intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)
        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)

        # get valid points
        depth_mask = (depth_image > 0)
        if seg_image is not None:
            # workspace_mask = get_workspace_mask(cloud, seg_image, trans=trans, organized=True, outlier=0.02)
            workspace_mask = seg_image.astype(bool)
        else:
            workspace_mask = np.ones_like(depth_mask).astype(bool)
        mask = (depth_mask & workspace_mask)

        cloud_masked = cloud[mask]

        # sample points random
        if len(cloud_masked) >= self._num_point:
            idxs = np.random.choice(len(cloud_masked), self._num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self._num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        if rgb_image is not None:
            points_color = rgb_image.reshape(-1, 3)
            ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                        'points_color': points_color.astype(np.float32)/255.0,
                        'coors': cloud_sampled.astype(np.float32) / self._voxel_size,
                        'feats': np.ones_like(cloud_sampled).astype(np.float32),
                        'orignal_pd': cloud.reshape(-1, 3).astype(np.float32)
                        }
        else:
            ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                        'coors': cloud_sampled.astype(np.float32) / self._voxel_size,
                        'feats': np.ones_like(cloud_sampled).astype(np.float32),
                        'orignal_pd': cloud.reshape(-1, 3).astype(np.float32)
                        }
        return ret_dict

    def inference(self, data_input: dict,
                  enable_vis: bool=True,
                  enable_collision: bool=False,
                  enable_filter: bool=False):
        # Load data
        batch_data = minkowski_collate_fn([data_input])
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self._device)
            else:
                batch_data[key] = batch_data[key].to(self._device)
        # Forward pass
        with torch.no_grad():
            end_points = self._net(batch_data)
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)
        # Collision detection
        if enable_collision and self._collision_thresh > 0:
            cloud = data_input['orignal_pd']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self._voxel_size_cd)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self._collision_thresh)
            gg = gg[~collision_mask]
        # Angle range filter
        if enable_filter:
            filtered = filter_rotation_matrices(gg.rotation_matrices, rz_range=(-135, -45), rx_range=(-60, 60))
            gg = gg[filtered]

        gg = gg.nms()
        gg = gg.sort_by_score()
        # Whether visualize result
        if enable_vis:
            # pc = data_input['point_clouds']
            if len(gg) > 30:
                gg = gg[:30]
            # print(len(gg))
            # gg = gg[:2]
            # print(gg[1])
            grippers = gg.to_open3d_geometry_list()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(data_input['orignal_pd'])
            if 'points_color' in data_input.keys():
                cloud.colors = o3d.utility.Vector3dVector(data_input['points_color'])
            # 初始化 Open3D 可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(cloud)
            # 添加额外的几何体（例如 grippers）
            for geom in grippers:
                vis.add_geometry(geom)
            # extend_coordinates = visualize_region_of_rotation([0.03260839, -0.00954793, 0.377])
            # for geom in extend_coordinates:
            #     vis.add_geometry(geom)
            # 创建一个小坐标系，用于可视化姿态
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            vis.add_geometry(world_frame)
            # Load view
            param = o3d.io.read_pinhole_camera_parameters("/home/automan/ROS2/cobot_ws/src/am_ros2_perception/am_grasp/camera_params.json")
            vis.get_view_control().convert_from_pinhole_camera_parameters(param)
            # 先运行一次可视化窗口，用户可以调整视角
            vis.run()
            # # **保存当前视角**
            # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            # o3d.io.write_pinhole_camera_parameters("./camera_params.json", param)

        if len(gg):
            return gg.translations[0], gg.rotation_matrices[0]
        else:
            return None, None

# def visualize_region_of_rotation(translation, enable_world: bool=False):
#     # 定义旋转范围
#     rx_min, rx_max = 0.5, 1.0
#     ry_min, ry_max = 0.5, 1.0
#     # rz 这里示例固定为 0，你可根据需求设定
#     rz_fixed = 0.0
#
#     # 定义离散采样步长
#     num_samples = 5  # 每个轴采样 5 个点，你可增加或减少
#     rx_vals = np.linspace(rx_min, rx_max, num_samples)
#     ry_vals = np.linspace(ry_min, ry_max, num_samples)
#
#     # 用来保存所有要可视化的几何体
#     geometries = []
#
#     # 在这个示例中，我们使用一个坐标系 (TriangleMesh) 来表示每个 (rx, ry) 的姿态
#     for rx in rx_vals:
#         for ry in ry_vals:
#             R = euler_to_rotmat(rx, ry, rz_fixed)
#
#             # 创建一个小坐标系，用于可视化姿态
#             frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
#
#             # 应用旋转
#             frame.rotate(R, center=(0, 0, 0))
#             # 应用平移
#             frame.translate(translation)
#
#             geometries.append(frame)
#
#     # 可以额外添加一个大坐标系，用来对比全局坐标
#     if enable_world:
#         world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
#         geometries.append(world_frame)
#
#     return geometries


def filter_rotation_matrices(matrices, rx_range=None, ry_range=None, rz_range=None):
    """
    筛选出欧拉角中 ry（绕 Y 轴旋转角）在 (ry_min, ry_max) 范围内的旋转矩阵
    matrices: list of 3x3 numpy 数组，每个数组为一个旋转矩阵
    返回：满足条件的旋转矩阵列表
    """
    filtered = []
    for i in range(matrices.shape[0]):
        R = matrices[i]
        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        rx, ry, rz = euler
        # 检查 ry 是否在指定范围内
        if rz_range[0] < rz < rz_range[1] and rx_range[0] < rx < rx_range[1]:
            filtered.append(True)
        else:
            filtered.append(False)
    return np.array(filtered).astype(bool)


if __name__ == '__main__':
    grasp_net = GraspPoseDetector(
        checkpoint_path='/home/automan/ROS2/cobot_ws/src/am_ros2_perception/am_grasp/am_grasp/graspness_implementation/ckpts/minkuresunet_realsense.tar'
    )

    rgb = np.array(Image.open("/home/automan/Pictures/RGBD/00000_color.png"))
    depth = np.array(Image.open("/home/automan/Pictures/RGBD/00000_depth.png"))

    intrinsic = np.array([[616.0755615234375, 0.0, 335.7129211425781],
                          [0.0, 616.6409912109375, 234.61709594726562],
                          [0.0, 0.0, 1.0]])
    resolution = np.array([640.0, 480.0])
    data_dict = grasp_net.data_process(depth, intrinsic=intrinsic, resolution=resolution, rgb_image=rgb)

    grasp_net.inference(data_dict, enable_vis=True)
