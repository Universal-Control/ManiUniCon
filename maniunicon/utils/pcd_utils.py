import numpy as np
import open3d as o3d
import trimesh
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Union, Dict
from collections import OrderedDict
import cv2

from maniunicon.utils import math_utils

try:
    import pytorch3d.ops as torch3d_ops
except:
    print("pytorch3d not installed")

try:
    from openpoints.models.layers import furthest_point_sample
except:
    print("openpoints not installed")

BOUND = [0.2, 1.03, -1.2, 1.2, -0.3, 0.7]


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # adjust view point manually, press 'q' to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def visualize_and_capture(
    pos, color, camera_params_path, output_image_path, visualize=False
):
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if isinstance(color, torch.Tensor):
        color = color.detach().cpu().numpy()

    # create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)

    # normalize color to [0, 1], assuming input color range is 0-255
    if np.max(color) > 1.5:
        color = color.astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(color)

    # create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)

    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )

    # add geometry
    vis.add_geometry(pcd)
    vis.add_geometry(frame_base)

    # set view parameters
    params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
    view_ctl = vis.get_view_control()
    view_ctl.convert_from_pinhole_camera_parameters(params)

    # update rendering  and capture image
    vis.update_geometry(pcd)
    vis.update_geometry(frame_base)
    vis.poll_events()
    vis.update_renderer()

    # capture and save image (add delay to ensure rendering is complete)
    vis.capture_screen_image(output_image_path, do_render=True)

    if visualize:
        vis.run()

    # close window
    vis.destroy_window()


def hand_save_view_point():
    robot = RealRobot(camera_only=True)
    obs = robot.get_obs()
    pos = obs["pointcloud"]["pos"]
    color = obs["pointcloud"]["color"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.colors = o3d.utility.Vector3dVector(color)
    save_view_point(
        pcd, "/home/minghuan/ppt_learning/ppt_learning/utils/test/view_point.json"
    )
    print("=" * 80)
    print("Done")


def capture_pcd(
    obs,
    view_point="/home/nuc001/minghuan/ppt_learning/ppt_learning/utils/test/view_point.json",
    output_image="./general_capture",
    tag="general_capture",
    visualize=False,
):
    pos = obs["pointcloud"]["pos"]
    color = obs["pointcloud"]["color"]

    if len(pos.shape) > 2:
        pos = pos.reshape(-1, pos.shape[-2], pos.shape[-1])
        color = color.reshape(-1, color.shape[-2], color.shape[-1])
        for idx, p in enumerate(pos):
            visualize_and_capture(
                p,
                color[idx],
                view_point,
                f"{output_image}_{idx}.png",
                visualize=visualize,
            )
    else:
        visualize_and_capture(
            pos, color, view_point, f"{output_image}.png", visualize=visualize
        )
    print("=" * 80)
    print(f"Saved {tag} pic")


def uniform_sampling(points, npoints=1200):
    n = points.shape[0]
    index = np.arange(n)
    if n == 0:
        return np.zeros(npoints, dtype=np.int64)
    if index.shape[0] > npoints:
        np.random.shuffle(index)
        index = index[:npoints]
    elif index.shape[0] < npoints:
        num_repeat = npoints // index.shape[0]
        index = np.concatenate([index for i in range(num_repeat)])
        index = np.concatenate([index, index[: npoints - index.shape[0]]])
    return index


def pcd_filter_bound(cloud, bound=BOUND):
    # return (
    #     (pcd["pos"][..., 2] > eps)
    #     & (pcd["pos"][..., 1] < max_dis)
    #     & (pcd["pos"][..., 0] < max_dis)
    #     & (pcd["pos"][..., 2] < max_dis)
    # )
    if isinstance(cloud, dict):
        pc = cloud["pos"]  # (n, 3)
    else:
        assert isinstance(cloud, np.ndarray), f"{type(cloud)}"
        assert cloud.shape[1] == 3, f"{cloud.shape}"
        pc = cloud

    # remove robot table
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = np.nonzero(
        np.logical_and.reduce((within_bound_x, within_bound_y, within_bound_z))
    )[0]

    return within_bound


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def rand_dist(size, min=-1.0, max=1.0):
    return (max - min) * torch.rand(size) + min


def rand_discrete(size, min=0, max=1):
    if min == max:
        return torch.zeros(size)
    return torch.randint(min, max + 1, size)


def voxelize_point_cloud(points, voxel_size=0.1, colors=None):
    """
    Voxelize a single point cloud.

    Parameters:
    points (np.ndarray): Input point cloud as a (N, 3) array.
    colors (np.ndarray): Input point cloud color as a (N, 3) array.
    voxel_size (float): Size of the voxel grid.

    Returns:
    np.ndarray: Voxelized point cloud.
    """
    # Convert numpy array to open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Voxelize the point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size
    )

    # Extract voxel centers
    voxels = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    voxel_centers = voxels * voxel_size + voxel_grid.origin

    return voxel_grid, voxel_centers


def visualize_point_cloud(points):
    # Convert the points to Open3D's PointCloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the PointCloud
    o3d.visualization.draw_geometries([pcd])


def update_openpoint_cfgs(cfg):
    pass


def vis_attention(pc1, pc2, attention_score):
    """
    src_emb: B * 3 * N_src
    score: B * N_src * N_dst
    """
    # TODO(lirui): build interactive feature visualization of the attention
    # http://www.open3d.org/docs/0.9.0/tutorial/Advanced/interactive_visualization.html

    for idx, (tool_point, obj_point) in enumerate(zip(pc1, pc2)):
        tool_pcd = o3d.geometry.PointCloud()
        tool_pcd.points = o3d.utility.Vector3dVector(tool_point.detach().cpu().numpy())
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_point.detach().cpu().numpy())
        tool_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        obj_pcd.paint_uniform_color([0.0, 1.0, 0.0])

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(tool_pcd)
        vis.run()  # user picks points

        attn = attention_score[idx].detach().cpu().numpy()
        picked_points = vis.get_picked_points()
        print("picked_points:", picked_points)

        if len(picked_points) > 0:
            corrsponding_score = attn[picked_points[0]]
            colors = plt.cm.jet(corrsponding_score)[:, :3]
            obj_pcd.colors = o3d.utility.Vector3dVector(colors)

            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            vis.add_geometry(obj_pcd)
            vis.run()  # user picks points
            vis.destroy_window()


def dbscan_outlier_removal(pcd):  # (N, 3)
    clustering = DBSCAN(eps=0.1, min_samples=10).fit(pcd)
    labels = clustering.labels_
    print("Number of clusters: ", len(set(labels)))
    # max_label = max(set(labels), key=labels) # only keep the cluster with the most points

    return np.array(pcd)[labels != -1]


def dbscan_outlier_removal_idx(pcd, eps=0.1, min_samples=300):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pcd)
    labels = clustering.labels_
    # print("Number of clusters: ", len(set(labels)))
    non_outlier_indices = np.where(labels != -1)[0]

    return non_outlier_indices


def dbscan_outlier_removal_idx(pcd, eps=0.1, min_samples=300):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pcd)
    labels = clustering.labels_
    # print("Number of clusters: ", len(set(labels)))
    non_outlier_indices = np.where(labels != -1)[0]

    return non_outlier_indices


def open3d_pcd_outlier_removal(
    pointcloud, radius_nb_num=300, radius=0.08, std_nb_num=300, vis=False
):
    """N x 3 or N x 6"""
    print("running outlier removal")
    model_pcd = o3d.geometry.PointCloud()
    model_pcd.points = o3d.utility.Vector3dVector(pointcloud[..., :3])
    model_pcd.colors = o3d.utility.Vector3dVector(pointcloud[..., 3:])
    # prior: it's a single rigid object
    model_pcd.remove_duplicated_points()
    model_pcd.remove_non_finite_points()
    print("finished removing duplicated and non-finite points")

    cl, ind = model_pcd.remove_radius_outlier(
        nb_points=int(radius_nb_num), radius=radius
    )
    model_pcd.points = o3d.utility.Vector3dVector(np.array(model_pcd.points)[ind, :3])
    model_pcd.colors = o3d.utility.Vector3dVector(np.array(model_pcd.colors)[ind, :3])
    print("finished removing radius outliers")

    cl, ind = model_pcd.remove_statistical_outlier(
        nb_neighbors=std_nb_num, std_ratio=2.0
    )
    print("finished removing statistical outliers")
    if vis:
        display_inlier_outlier(model_pcd, ind)
    # return pointcloud[ind] # No remove, not sure why
    return np.array(model_pcd.select_by_index(ind).points), np.array(
        model_pcd.select_by_index(ind).colors
    )


def display_inlier_outlier(pcd, ind):
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)

    # Display inlier and outlier point clouds
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def add_crop_noise_to_points_data(point_cloud, radius_min=0.015, radius_max=0.04):
    """
    select an anchor point, and remove all points inside radius of size 0.02
    """
    # should be independent of tool or object
    CROP_REMAINING = np.random.uniform() < 0.5
    radius = radius_max
    if not CROP_REMAINING:
        radius = radius_min
        radius = np.random.uniform(0.005, radius)  # further reduce

    CROP_TOOL = True
    point_num = point_cloud.shape[0]
    point_cloud = (
        point_cloud[: point_num // 2] if CROP_TOOL else point_cloud[point_num // 2 :]
    )

    select_anchor_index = np.random.randint(point_num // 2)
    point_center = point_cloud[select_anchor_index]
    close_dist = np.linalg.norm(point_cloud[:, :3] - point_center[None, :3], axis=1)
    close_mask = close_dist < radius if CROP_REMAINING else close_dist >= radius
    if close_mask.sum() != 0 and close_mask.sum() != point_num // 2:
        # in case it crops out entire object or tool
        masked_point = point_cloud[close_mask]
        random_index = np.random.choice(masked_point.shape[0], (~close_mask).sum())
        masked_point = np.concatenate(
            (masked_point, masked_point[random_index]), axis=0
        )
        point_cloud[:] = masked_point  # modify in place

    return point_cloud.astype(np.float32)


def randomly_drop_point(point_clouds, drop_point_min=0.1, drop_point_max=0.5):
    """
    randomly drop points such that the pointcloud can work in the real world
    """
    orig_shape = point_clouds.shape
    if len(orig_shape) == 2:  # (N, 3)
        point_clouds = point_clouds[None, :, :]  # B, N, 3

    remained_point_ratio = np.random.uniform(drop_point_min, drop_point_max)
    remained_point_num = int(point_clouds.shape[1] * remained_point_ratio)

    for idx, point_cloud in enumerate(point_clouds):
        try:
            random_index = np.random.choice(point_cloud.shape[0], remained_point_num)
            resampled_index = np.random.choice(random_index, point_cloud.shape[0])
            point_cloud = point_cloud[resampled_index]
        except:
            pass
        point_clouds[idx] = point_cloud  # in-place

    if len(orig_shape) == 2:
        point_clouds = point_clouds.squeeze(0)

    return point_clouds.astype(np.float32)


def add_gaussian_noise(clouds: np.ndarray, noise_level=1):
    # cloud should be (B, n, 3)
    orig_shape = clouds.shape
    if len(orig_shape) == 3:
        B, N, _ = clouds.shape
        clouds = clouds.reshape(-1, 3)
    num_points = clouds.shape[0]
    multiplicative_noise = (
        1 + np.random.randn(num_points)[:, None] * 0.01 * noise_level
    )  # (n, 1)
    clouds = clouds * multiplicative_noise
    if len(orig_shape) == 3:
        clouds = clouds.reshape(B, N, 3).astype(np.float32)
    return clouds


def add_pointoutlier_aug(point_cloud, outlier_point_num=20):
    """
    add outlier points to the pointcloud
    """

    # completely random points to increate robustness
    outlier_points = np.random.uniform(-1, 1, size=(outlier_point_num, 3))
    random_index = np.random.choice(point_cloud.shape[0], outlier_point_num)
    point_cloud[random_index] = outlier_points

    # point_clouds[:,:3] = point_cloud.T
    return point_cloud.astype(np.float32)


def cutplane_pointcloud_aug(point_cloud, action):
    # print("in cutplane")
    N = len(point_cloud)

    for b in range(N):
        if np.random.uniform() < 0.5:
            # tool
            cut_tool = True
            pcd = point_cloud[b].T[:512, :3]
        else:
            # object
            cut_tool = False
            pcd = point_cloud[b].T[512:, :3]

        bounding_box = trimesh.PointCloud(pcd).bounding_box
        # bounding_box = bounding_box.to_mesh()
        random_pts = bounding_box.sample_volume(2)

        # first point will be the point, second point will be used for the normal
        pt = random_pts[0]
        vec = random_pts[1] - pt
        normal = vec / np.linalg.norm(vec)
        normal = normal.reshape(3, 1)

        # get points that are on one side of this plane
        shifted_pcd = pcd - pt
        dots = np.matmul(shifted_pcd, normal)
        pos_inds = np.where(dots > 0)[0]
        neg_inds = np.where(dots < 0)[0]
        # pick the majority and then oversample it to match the pointcloud size

        # print(f"pos: {len(pos_inds)} neg: {len(neg_inds)}")
        if len(pos_inds) > len(neg_inds):
            keep_pts = pcd[pos_inds]
        else:
            keep_pts = pcd[neg_inds]

        if pcd.shape[0] > keep_pts.shape[0]:
            random_index = np.random.choice(
                keep_pts.shape[0], pcd.shape[0] - keep_pts.shape[0], replace=True
            )
        keep_pts = np.concatenate((keep_pts, keep_pts[random_index]), axis=0)
        if cut_tool:
            point_cloud[b].T[:512, :3] = keep_pts  # in-place should be
        else:
            point_cloud[b].T[512:, :3] = keep_pts  # in-place should be

    return point_cloud.astype(np.float32), action.astype(np.float32)


def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )
    return Rz, Ry


def get_arrow(end, origin=np.array([0, 0, 0]), scale=1, color=(1.0, 0.0, 0.0)):
    assert not np.all(end == origin)
    import open3d as o3d

    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=size / 17.5 * scale,
        cone_height=size * 0.2 * scale,
        cylinder_radius=size / 30 * scale,
        cylinder_height=size * (1 - 0.2 * scale),
    )
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)
    return mesh


def rhlb(bounds):
    val = np.random.random() * (max(bounds) - min(bounds)) + min(bounds)
    return val


def transform_pcd(pcd, transform):
    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new


def vis_pcd_html(pcds, rgbs, name, gt_traj=None):
    rgb_strings = [f"rgb{rgbs[i][0],rgbs[i][1],rgbs[i][2]}" for i in range(len(rgbs))]

    if gt_traj is not None:
        gx, gy, gz = gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2]

    pcd_plots = [
        go.Scatter3d(
            x=pcds[:, 0],
            y=pcds[:, 1],
            z=pcds[:, 2],
            mode="markers",
            marker=dict(
                size=8,
                color=rgb_strings,
            ),
        )
    ]

    if gt_traj is not None:
        gt_plot = [
            go.Scatter3d(
                x=gx,
                y=gy,
                z=gz,
                mode="markers",
                marker=dict(size=10, color="red"),
            )
        ]
        pcd_plots += gt_plot

    fig = go.Figure(pcd_plots)
    path = f"./plots"
    os.makedirs(path, exist_ok=True)
    fig.write_html(os.path.join(path, f"vis_{name}.html"))


def simulate_deform_contact_point(point_cloud, uniform=False):
    # pcd: N x 4 x 1024
    # high, low = 1.5, 0.5
    from scipy.spatial.transform import Rotation as R

    high, low = 2.0, 0.4
    for idx, pcd_ in enumerate(point_cloud):
        pcd = pcd_.T[:, :3]
        deform_about_point = np.random.randint(len(pcd))
        deform_about_point = pcd[deform_about_point]

        # scale up the points about this specific location
        # scale_x, scale_y, scale_z = rhlb((1.5, 0.5)), rhlb((1.5, 0.5)), rhlb((1.5, 0.5))
        scale_x, scale_y, scale_z = (
            rhlb((high, low)),
            rhlb((high, low)),
            rhlb((high, low)),
        )

        # apply the scaling to the place pcd
        pcd_contact_cent = pcd - deform_about_point
        if uniform:
            pcd_contact_cent = pcd_contact_cent * scale_x
            pcd_aug = pcd_contact_cent + deform_about_point
        else:
            # apply a random rotation, scale, and then unrotate
            # rot_grid = R.random().as_matrix()
            # rot_idx = np.random.randint(rot_grid.shape[0], size=1)
            rnd_rot = R.random().as_matrix()  # rot_grid[rot_idx]
            rnd_rot_T = np.eye(4)
            rnd_rot_T[:-1, :-1] = rnd_rot

            pcd_contact_cent = transform_pcd(pcd_contact_cent, rnd_rot_T)
            pcd_contact_cent[:, 0] *= scale_x
            pcd_contact_cent[:, 1] *= scale_y
            pcd_contact_cent[:, 2] *= scale_z

            pcd_contact_cent = transform_pcd(pcd_contact_cent, np.linalg.inv(rnd_rot_T))

            pcd_aug = pcd_contact_cent + deform_about_point
        pcd_[:3, :] = pcd_aug.T
    return point_cloud


def se3_augmentation(gripper_poses_euler, point_clouds, bounds, rgbs=None):
    """
    Apply SE(3) augmentation to batches of gripper poses (represented in Euler angles) and point clouds,
    ensuring each gripper pose remains within specified bounds.

    Parameters:
    gripper_poses_euler (np.ndarray): Batch of gripper poses as a Bx6 matrix [x, y, z, roll, pitch, yaw].
    point_clouds (np.ndarray): Batch of point clouds as a BxNx3 matrix.
    bounds (list or np.ndarray): The scene bounds [x0, x1, y0, y1, z0, z1].

    Returns:
    np.ndarray: Batch of augmented gripper poses in Euler angles.
    np.ndarray: Batch of augmented point clouds.
    """
    # Generate random rotation
    # random_rotation = R.random().as_matrix()  # Random rotation matrix

    ## We limit the rotation to yaw and +-15 degrees
    # Generate a random angle between -15 and 5 degrees
    angle_rad = np.radians(np.random.uniform(-15, 15))

    # Create the yaw rotation matrix
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    random_rotation = np.array(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
    )

    # Generate random translation within bounds
    translation = np.zeros(3)
    perturb_attempts = 0
    while True:
        perturb_attempts += 1
        if perturb_attempts > 100:
            print(
                "Failing to perturb action and keep it within bounds. Use the original one."
            )
            return gripper_poses_euler, point_clouds

        trans_range = (0.125, 0.125, 0.125)  # Adjust range as needed
        translation = trans_range * np.random.uniform(-1, 1, size=3)
        new_positions = gripper_poses_euler[:, :3] + translation

        # Check if all new positions are within bounds
        if np.all(
            (bounds[0] <= new_positions[:, 0])
            & (new_positions[:, 0] <= bounds[1])
            & (bounds[2] <= new_positions[:, 1])
            & (new_positions[:, 1] <= bounds[3])
            & (bounds[4] <= new_positions[:, 2])
            & (new_positions[:, 2] <= bounds[5])
        ):
            break

    new_euler_angles = gripper_poses_euler[:, 3:]
    # # Convert Euler angles to rotation matrices
    # rotation_matrices = R.from_euler("xyz", gripper_poses_euler[:, 3:]).as_matrix()

    # # Calculate new rotation matrices
    # new_rotation_matrices = np.einsum("ij,bjk->bik", random_rotation, rotation_matrices)

    # # Convert new rotation matrices back to Euler angles
    # new_euler_angles = R.from_matrix(new_rotation_matrices).as_euler("xyz")

    # Apply the translation to the gripper pose positions
    augmented_positions = gripper_poses_euler[:, :3] + translation

    # Create the augmented gripper poses
    augmented_gripper_poses_euler = np.hstack((augmented_positions, new_euler_angles))

    # shift points to have action_gripper pose as the origin
    gripper_pos = np.tile(
        gripper_poses_euler[:, :3], (point_clouds.shape[1], 1, 1)
    ).transpose(1, 0, 2)
    augmented_point_clouds = point_clouds - gripper_pos
    # Apply the SE(3) transformation to each point cloud in the batch
    # augmented_point_clouds = np.einsum("ij,bnj->bni", random_rotation, augmented_point_clouds)
    # Shift the point clouds back to the original position
    augmented_point_clouds = augmented_point_clouds + gripper_pos + translation

    # vis_pcd_html(augmented_point_clouds[0], rgbs[0], "augmented", augmented_gripper_poses_euler)
    # vis_pcd_html(point_clouds[0], rgbs[0], "origin", gripper_poses_euler)

    return augmented_gripper_poses_euler, augmented_point_clouds


def vis_pcd(pcd):
    import open3d as o3d

    pcd_o3d = o3d.geometry.PointCloud()
    if isinstance(pcd, dict):
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd["pos"])
        if "colors" in pcd:
            pcd_o3d.colors = o3d.utility.Vector3dVector(pcd["colors"] / 255)
    else:
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd[..., :3])
        if len(pcd.shape[-1] > 3):
            pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[..., 3:])
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )  # create a coordinate frame
    o3d.visualization.draw_geometries([pcd_o3d, frame_base])


def vis_depths(colors, depths, near_depth=-1.0, far_depth=-1.0):
    imgs = []
    for color in colors:
        imgs.append(color.transpose((1, 2, 0)))
    for depth in depths:
        img = colorize_depth_maps(
            depth,
            depth.min() if near_depth < 0.0 else near_depth,
            depth.max() if far_depth < 0.0 else far_depth,
        )[0].transpose((1, 2, 0))
        imgs.append(img)

    vis_img = np.concatenate(imgs, axis=1)

    plt.imshow(vis_img)
    # plt.savefig("test.png")
    plt.show()


def select_mask(obs, key, mask):
    if key in obs:
        obs[key] = obs[key][mask]


def pcd_filter_bound(cloud, eps=1e-3, max_dis=1.5, bound=BOUND):
    # return (
    #     (pcd["pos"][..., 2] > eps)
    #     & (pcd["pos"][..., 1] < max_dis)
    #     & (pcd["pos"][..., 0] < max_dis)
    #     & (pcd["pos"][..., 2] < max_dis)
    # )
    if isinstance(cloud, dict):
        pc = cloud["pos"]  # (n, 3)
    else:
        assert isinstance(cloud, np.ndarray), f"{type(cloud)}"
        assert cloud.shape[1] == 3, f"{cloud.shape}"
        pc = cloud

    # remove robot table
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = np.nonzero(
        np.logical_and.reduce((within_bound_x, within_bound_y, within_bound_z))
    )[0]

    return within_bound


def pcd_filter_with_mask(obs, mask, env=None):
    assert isinstance(obs, dict), f"{type(obs)}"
    for key in ["pos", "color", "seg", "visual_seg", "robot_seg"]:
        select_mask(obs, key, mask)


def pcd_downsample(
    obs,
    env=None,
    bound_clip=False,
    ground_eps=-1e-3,
    max_dis=15,
    num=1200,
    method="fps",
    bound=BOUND,
):
    assert method in [
        "fps",
        "uniform",
    ], "expected method to be 'fps' or 'uniform', got {method}"

    sample_mehod = uniform_sampling if method == "uniform" else fps_sampling
    # import ipdb; ipdb.set_trace()
    if bound_clip:
        pcd_filter_with_mask(
            obs,
            pcd_filter_bound(obs, eps=ground_eps, max_dis=max_dis, bound=bound),
            env,
        )
    pcd_filter_with_mask(obs, sample_mehod(obs["pos"], num), env)
    return obs


def fps_sampling(points, npoints=1200):
    num_curr_pts = points.shape[0]
    if num_curr_pts < npoints:
        return np.random.choice(num_curr_pts, npoints, replace=True)
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    try:
        fps_idx = furthest_point_sample(points[..., :3], npoints)
    except:
        npoints = torch.tensor([npoints]).cuda()
        _, fps_idx = torch3d_ops.sample_farthest_points(points[..., :3], K=npoints)

    return fps_idx.squeeze(0).cpu().numpy()


def uniform_sampling(points, npoints=1200):
    n = points.shape[0]
    index = np.arange(n)
    if n == 0:
        return np.zeros(npoints, dtype=np.int64)
    if index.shape[0] > npoints:
        np.random.shuffle(index)
        index = index[:npoints]
    elif index.shape[0] < npoints:
        num_repeat = npoints // index.shape[0]
        index = np.concatenate([index for i in range(num_repeat)])
        index = np.concatenate([index, index[: npoints - index.shape[0]]])
    return index


def add_gaussian_noise(
    cloud: np.ndarray, np_random: np.random.RandomState, noise_level=1
):
    # cloud is (n, 3)
    num_points = cloud.shape[0]
    multiplicative_noise = (
        1 + np_random.randn(num_points)[:, None] * 0.01 * noise_level
    )  # (n, 1)
    return cloud * multiplicative_noise


def add_perlin_noise(
    points, scale=0.1, octaves=1, persistence=0.5, lacunarity=2.0, amplitude=1.0
):
    """
    Adds Perlin noise to a point cloud.

    :param points: A numpy array of shape (n, 3) representing the point cloud.
    :param scale: Scale of the Perlin noise.
    :param octaves: Number of octaves for the Perlin noise.
    :param persistence: Persistence of the Perlin noise.
    :param lacunarity: Lacunarity of the Perlin noise.
    :param amplitude: Amplitude of the noise to make the effect more noticeable.
    :return: A numpy array of the same shape as points with added Perlin noise.
    """
    import noise

    noisy_points = np.zeros_like(points)

    for i, point in enumerate(points):
        x, y, z = point
        noise_x = (
            noise.pnoise3(
                x * scale,
                y * scale,
                z * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            * amplitude
        )
        noise_y = (
            noise.pnoise3(
                y * scale,
                z * scale,
                x * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            * amplitude
        )
        noise_z = (
            noise.pnoise3(
                z * scale,
                x * scale,
                y * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            * amplitude
        )
        noisy_points[i] = point + np.array([noise_x, noise_y, noise_z])

    return noisy_points


def uniform_sampling_torch(points, npoints=1200):
    """
    sampling the point cloud to a fixed number of points

    Args:
        points: torch.Tensor -shape is [B, N, 3] or [N, 3]
        npoints: int - number of points to sample

    Returns:
        torch.Tensor - sampled point indices, shape is [B, npoints] or [npoints]
    """
    # examine the input shape
    if len(points.shape) == 3:  # [B, N, 3]
        batch_size, n, _ = points.shape
        batch_mode = True
    elif len(points.shape) == 2:  # [N, 3]
        n = points.shape[0]
        batch_mode = False
        # expand to batch mode for unified processing
        points = points.unsqueeze(0)
        batch_size = 1
    else:
        raise ValueError(
            f"The input point cloud is not valid, expected shape is [B, N, 3] or [N, 3], but got {points.shape}"
        )

    # process the case when there is no point
    if n == 0:
        if batch_mode:
            return torch.zeros(
                (batch_size, npoints), dtype=torch.int64, device=points.device
            )
        else:
            return torch.zeros(npoints, dtype=torch.int64, device=points.device)

    # create index tensor [B, N]
    indices = torch.arange(n, device=points.device).expand(batch_size, n)

    if n > npoints:
        # random sampling with batch processing
        rand_indices = torch.argsort(
            torch.rand(batch_size, n, device=points.device), dim=1
        )
        sampled_indices = torch.gather(indices, 1, rand_indices[:, :npoints])
    elif n < npoints:
        num_repeat = npoints // n
        remaining = npoints - num_repeat * n

        repeated_indices = indices.repeat_interleave(num_repeat, dim=1)

        if remaining > 0:
            remaining_indices = indices[:, :remaining]
            sampled_indices = torch.cat([repeated_indices, remaining_indices], dim=1)
        else:
            sampled_indices = repeated_indices
    else:
        sampled_indices = indices

    if batch_mode:
        return sampled_indices
    else:
        return sampled_indices[0]  


def pcd_filter_bound_torch(pc, bound):
    # boundtensor
    if not isinstance(bound, torch.Tensor):
        bound = torch.tensor(bound, device=pc.device)

    batch_mode = len(pc.shape) == 3

    if not batch_mode:
        pc = pc.unsqueeze(0)  # [N, 3] -> [1, N, 3]

    # bound
    if len(bound.shape) == 1:
        bound = bound.unsqueeze(0).expand(pc.shape[0], -1)

    within_bound_x = (pc[..., 0] > bound[:, 0:1]) & (pc[..., 0] < bound[:, 1:2])
    within_bound_y = (pc[..., 1] > bound[:, 2:3]) & (pc[..., 1] < bound[:, 3:4])
    within_bound_z = (pc[..., 2] > bound[:, 4:5]) & (pc[..., 2] < bound[:, 5:6])

    within_bound = within_bound_x & within_bound_y & within_bound_z

    if batch_mode:
        return within_bound
    else:
        return within_bound[0]


def pcd_downsample_torch(
    obs,
    bound_clip=False,
    num=1200,
    method="uniform",
    bound=None,
):
    assert method in [
        "uniform",
    ], "expected method to be 'uniform', got {method}"

    if bound_clip:
        mask = pcd_filter_bound_torch(obs["pos"], bound=bound)
        obs = {k: [v[i][mask[i]] for i in range(len(v))] for k, v in obs.items()}
    res_obs = {k: [] for k in obs}
    for i in range(len(obs["pos"])):
        mask = uniform_sampling_torch(obs["pos"][i], npoints=num)
        for k in res_obs:
            res_obs[k].append(obs[k][i][mask])
    for k in res_obs:
        obs[k] = torch.stack(res_obs[k])
    return obs


def sample_pcd_data(data, npoints, in_channels):
    """
    Refer to https://github.com/guochengqian/PointNeXt/blob/master/examples/classification/train.py#L248
    """
    points = data["x"]
    point_shape = len(points.size())
    if point_shape == 4:
        B, T, N, D = points.shape
        points = points.reshape(B * T, N, D)
    num_curr_pts = points.shape[1]
    if num_curr_pts > npoints:  # point resampling strategy
        if npoints == 1024:
            point_all = 1200
        elif npoints == 2048:
            point_all = 2400
        elif npoints == 4096:
            point_all = 4800
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()
        if points.size(1) < point_all:
            point_all = points.size(1)
        try:
            fps_idx = furthest_point_sample(points[..., :3].contiguous(), point_all)
        except:
            _, fps_idx = torch3d_ops.sample_farthest_points(
                points[..., :3].contiguous(), K=point_all
            )
        fps_idx = fps_idx[..., np.random.choice(point_all, npoints, False)]
        points = torch.gather(
            points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1])
        )
    data["pos"] = points[:, :, :3].contiguous()
    data["x"] = points[:, :, :in_channels].transpose(1, 2).contiguous()
    if "colors" in data:
        data["colors"] = torch.gather(
            data["colors"],
            1,
            fps_idx.unsqueeze(-1).long().expand(-1, -1, data["colors"].shape[-1]),
        )
        data["colors"] = data["colors"].transpose(1, 2).contiguous()
    if point_shape == 4:
        data["pos"] = data["pos"].reshape(B, T, -1, 3)
        data["x"] = data["x"].reshape(B, T, -1, in_channels)
        if "colors" in data:
            data["colors"] = data["colors"].reshape(B, T, -1, in_channels)


def create_pointcloud_from_depth_batch(
    intrinsic_matrix: np.ndarray | torch.Tensor,
    depth: np.ndarray | torch.Tensor,
    keep_invalid: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray | torch.Tensor:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix: A (3, 3) array providing camera's calibration matrix.
        depth: An array of shape (H, W) with values encoding the depth measurement.
        keep_invalid: Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed.
            Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        An array/tensor of shape (N, 3) comprising of 3D coordinates of points.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    # convert to numpy matrix
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    depth = convert_to_torch(depth, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = depth.device
    # convert inputs to torch tensors
    intrinsic_matrix = convert_to_torch(
        intrinsic_matrix, dtype=torch.float32, device=device
    )
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # compute pointcloud
    depth_cloud = math_utils.unproject_depth(depth, intrinsic_matrix)
    # convert 3D points to world frame
    depth_cloud = math_utils.transform_points(depth_cloud, position, orientation)

    # keep only valid entries if flag is set
    if not keep_invalid:
        pts_idx_to_keep = torch.all(
            torch.logical_and(~torch.isnan(depth_cloud), ~torch.isinf(depth_cloud)),
            dim=-1,
        )
        depth_cloud = depth_cloud[pts_idx_to_keep, ...]

    # return everything according to input type
    if is_numpy:
        return depth_cloud.detach().cpu().numpy()
    else:
        return depth_cloud


# def create_pointcloud_from_rgbd_batch(
#     intrinsic_matrix: torch.Tensor | np.ndarray,
#     depth: torch.Tensor | np.ndarray,
#     rgb: torch.Tensor | np.ndarray | Tuple[float, float, float] = None,
#     normalize_rgb: bool = False,
#     position: Sequence[float] | None | torch.Tensor = None,
#     orientation: Sequence[float] | None | torch.Tensor = None,
#     device: torch.device | str | None | torch.Tensor = None,
#     num_channels: int = 3,
#     maximum_distance: float = 100.,
# ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
def create_pointcloud_from_rgbd_batch(
    intrinsic_matrix,
    depth,
    rgb=None,
    normalize_rgb=False,
    position=None,
    orientation=None,
    device=None,
    num_channels=3,
    maximum_distance=100.0,
):
    """Creates pointcloud from input depth image and camera transformation matrix.

    This function provides the same functionality as :meth:`create_pointcloud_from_depth` but also allows
    to provide the RGB values for each point.

    The ``rgb`` attribute is used to resolve the corresponding point's color:

    - If a ``np.array``/``wp.array``/``torch.tensor`` of shape (H, W, 3), then the corresponding channels encode RGB values.
    - If a tuple, then the point cloud has a single color specified by the values (r, g, b).
    - If None, then default color is white, i.e. (0, 0, 0).

    If the input ``normalize_rgb`` is set to :obj:`True`, then the RGB values are normalized to be in the range [0, 1].

    Args:
        intrinsic_matrix: A (3, 3) array/tensor providing camera's calibration matrix.
        depth: An array/tensor of shape (H, W) with values encoding the depth measurement.
        rgb: Color for generated point cloud. Defaults to None.
        normalize_rgb: Whether to normalize input rgb. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation `(w, x, y, z)` of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed. Defaults to None, in which case
            it takes the device that matches the depth image.
        num_channels: Number of channels in RGB pointcloud. Defaults to 3.

    Returns:
        A tuple of (N, 3) arrays or tensors containing the 3D coordinates of points and their RGB color respectively.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.

    Raises:
        ValueError:  When rgb image is a numpy array but not of shape (H, W, 3) or (H, W, 4).
    """
    # check valid inputs
    if rgb is not None and not isinstance(rgb, tuple):
        if len(rgb.shape) == 3:
            rgb = rgb[None]
            if rgb.shape[2] not in [3, 4]:
                raise ValueError(
                    f"Input rgb image of invalid shape: {rgb.shape} != (H, W, 3) or (H, W, 4)."
                )
        elif len(rgb.shape) == 4:
            if rgb.shape[3] not in [3, 4]:
                raise ValueError(
                    f"Input rgb image of invalid shape: {rgb.shape}!= (H, W, 3) or (H, W, 4)."
                )
        else:
            raise ValueError(
                f"Input rgb image not three-dimensional. Received shape: {rgb.shape}."
            )
    if num_channels not in [3, 4]:
        raise ValueError(f"Invalid number of channels: {num_channels} != 3 or 4.")

    # check if input depth is numpy array
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    if is_numpy:
        depth = torch.from_numpy(depth).to(device=device)
    # retrieve XYZ pointcloud
    points_xyz = create_pointcloud_from_depth_batch(
        intrinsic_matrix, depth, True, position, orientation, device=device
    )

    # get image height and width
    im_height, im_width = depth.shape[-2:]
    # total number of points
    num_points = im_height * im_width
    # extract color value
    if rgb is not None:
        if isinstance(rgb, (np.ndarray, torch.Tensor)):
            # copy numpy array to preserve
            rgb = convert_to_torch(rgb, device=device, dtype=torch.float32)
            rgb = rgb[:, :, :, :3]
            # convert the matrix to (W, H, 3) from (H, W, 3) since depth processing
            # is done in the order (u, v) where u: (0, W-1) and v: (0 - H-1)
            batch_size, H, W, _ = rgb.shape

            #  permute ， [B, H, W, 3]  [B, W, H, 3]
            #  reshape  [B, W*H, 3]
            points_rgb = rgb.permute(0, 2, 1, 3).reshape(batch_size, W * H, 3)
        elif isinstance(rgb, (tuple, list)):
            # same color for all points
            points_rgb = torch.Tensor(
                (rgb,) * num_points, device=device, dtype=torch.uint8
            )
        else:
            # default color is white
            points_rgb = torch.Tensor(
                ((0, 0, 0),) * num_points, device=device, dtype=torch.uint8
            )
    else:
        points_rgb = torch.Tensor(
            ((0, 0, 0),) * num_points, device=device, dtype=torch.uint8
        )
    # normalize color values
    if normalize_rgb:
        points_rgb = points_rgb.float() / 255

    # remove invalid points
    pts_idx_to_keep = torch.all(
        torch.logical_and(~torch.isnan(points_xyz), ~torch.isinf(points_xyz)), dim=-1
    )
    points_rgb[~pts_idx_to_keep, ...] = maximum_distance
    points_xyz[~pts_idx_to_keep, ...] = 0

    # add additional channels if required
    if num_channels == 4:
        points_rgb = torch.nn.functional.pad(
            points_rgb, (0, 1), mode="constant", value=1.0
        )

    # return everything according to input type
    if is_numpy:
        res = OrderedDict()
        res["pos"] = points_xyz.cpu().numpy()
        res["color"] = points_rgb.cpu().numpy()
        return res
    else:
        res = OrderedDict()
        res["pos"] = points_xyz
        res["color"] = points_rgb
        return res


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def capture_depth(depth, name):
    depth_colormap_proj0 = colorize_depth_maps(depth, 0.0, 5.0)[0].transpose(1, 2, 0)
    depth_colormap_proj0 = depth_colormap_proj0[..., ::-1]  # flip RGB to BGR
    cv2.imshow(name, depth_colormap_proj0)
    cv2.waitKey(0)
