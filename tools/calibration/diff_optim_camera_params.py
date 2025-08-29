import os
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger as lgr
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from tools.diff_optim_camera_utils import (
    NVDiffrastRenderer,
    as_mesh,
    robust_compute_rotation_matrix_from_ortho6d,
    SAMPromptDrawer,
    SAM_TYPE,
    SAM_PATH,
)


def extract_colors_from_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    global_materials = {}
    for material in root.findall("material"):
        name = material.attrib["name"]
        color_elem = material.find("color")
        if color_elem is not None and "rgba" in color_elem.attrib:
            rgba = [float(c) for c in color_elem.attrib["rgba"].split()]
            global_materials[name] = rgba

    link_colors = {}

    for link in root.iter("link"):
        link_name = link.attrib["name"]
        visual = link.find("./visual")
        if visual is not None:
            material = visual.find("./material")
            if material is not None:
                color = material.find("color")
                if color is not None and "rgba" in color.attrib:
                    rgba = [float(c) for c in color.attrib["rgba"].split()]
                    link_colors[link_name] = rgba
                elif "name" in material.attrib:
                    material_name = material.attrib["name"]
                    if material_name in global_materials:
                        link_colors[link_name] = global_materials[material_name]

    return link_colors


def parse_origin(element):
    """Parse the origin element for translation and rotation."""
    origin = element.find("origin")
    xyz = np.zeros(3)
    rotation = np.eye(3)
    if origin is not None:
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(
            origin.attrib.get("rpy", "0 0 0"), sep=" "
        )  # Roll, pitch, yaw
        rotation = R.from_euler("xyz", rpy).as_matrix()
    return xyz, rotation


def create_primitive_mesh(geometry, translation, rotation):
    """Create a trimesh object from primitive geometry definitions with transformations."""
    if geometry.tag.endswith("box"):
        size = np.fromstring(geometry.attrib["size"], sep=" ")
        mesh = trimesh.creation.box(extents=size)
    elif geometry.tag.endswith("sphere"):
        radius = float(geometry.attrib["radius"])
        mesh = trimesh.creation.icosphere(radius=radius)
    elif geometry.tag.endswith("cylinder"):
        radius = float(geometry.attrib["radius"])
        length = float(geometry.attrib["length"])
        mesh = trimesh.creation.cylinder(radius=radius, height=length)
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.tag}")
    return apply_transform(mesh, translation, rotation)


def apply_transform(mesh, translation, rotation):
    """Apply translation and rotation to a mesh."""
    # mesh.apply_translation(-mesh.centroid)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mesh.apply_transform(transform)
    return mesh


def load_link_geometries(urdf_path, link_names, collision=False):
    """Load geometries (trimesh objects) for specified links from a URDF file, considering origins."""
    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_geometries = {}
    link_colors_from_urdf = extract_colors_from_urdf(urdf_path)

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        link_color = link_colors_from_urdf.get(link_name, None)
        if link_name in link_names:
            geom_index = "visual"
            if collision:
                geom_index = "collision"
            link_mesh = []
            for visual in link.findall(".//" + geom_index):
                geometry = visual.find("geometry")
                xyz, rotation = parse_origin(visual)
                try:
                    if geometry[0].tag.endswith("mesh"):
                        mesh_filename = geometry[0].attrib["filename"]
                        if mesh_filename.startswith("package://"):
                            mesh_filename = mesh_filename.split("package://")[1]
                        full_mesh_path = os.path.join(urdf_dir, mesh_filename)
                        mesh = as_mesh(trimesh.load(full_mesh_path))
                        scale = np.fromstring(
                            geometry[0].attrib.get("scale", "1 1 1"), sep=" "
                        )
                        mesh.apply_scale(scale)
                        link_mesh.append(apply_transform(mesh, xyz, rotation))
                    else:  # Handle primitive shapes
                        mesh = create_primitive_mesh(geometry[0], xyz, rotation)
                        link_mesh.append(mesh)

                except Exception as e:
                    print(f"Failed to load geometry for {link_name}: {e}")
            if len(link_mesh) == 0:
                continue
            elif len(link_mesh) > 1:
                link_trimesh = as_mesh(trimesh.Scene(link_mesh))
            elif len(link_mesh) == 1:
                link_trimesh = link_mesh[0]

            if link_color is not None:
                link_trimesh.visual.face_colors = np.array(link_color)
            link_geometries[link_name] = link_trimesh

    return link_geometries


class PKRobotArm:
    def __init__(self, urdf_path: str, device: str = "cuda"):
        self.urdf_path = urdf_path
        self.device = device
        self.dtype = torch.float
        self.pk_chain = pk.build_chain_from_urdf(open(self.urdf_path).read().encode())
        self.pk_chain.to(device=self.device, dtype=self.dtype)
        self.all_link_names = self.pk_chain.get_link_names()
        self.load_meshes()

    def load_meshes(self):
        self.link_visuals_dict = load_link_geometries(
            self.urdf_path, self.all_link_names
        )
        self.link_collisions_dict = load_link_geometries(
            self.urdf_path, self.all_link_names, collision=True
        )

    def ensure_tensor(self, th, ensure_batch_dim=True):
        """
        Converts a number of possible types into a tensor. The order of the tensor is determined by the order
        of self.get_joint_parameter_names(). th must contain all joints in the entire chain.
        """
        if isinstance(th, np.ndarray):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
        elif isinstance(th, list):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
        if len(th.shape) < 2 and ensure_batch_dim:
            th = th.unsqueeze(0)
        return th

    def get_state_trimesh(
        self,
        joint_pos,
        X_w_b=torch.eye(4),
        visual=True,
        collision=False,
    ):
        """
        Get the trimesh representation of the robotic arm based on the provided joint positions and base transformation.

        Parameters:
        - joint_pos (list of float, or np.array, or torch tensor): Joint positions of the robot hand.
        - X_w_b (torch.tensor): A 4x4 transformation matrix representing the pose of the hand base in the world frame.
        - visual (bool): Whether to return the visual mesh of the hand.
        - collision (bool): Whether to return the collision mesh of the hand.

        Returns:
        - scene (trimesh.Trimesh): A trimesh object representing the robotic hand in its current pose.
        """

        self.current_status = self.pk_chain.forward_kinematics(
            th=self.ensure_tensor(joint_pos)
        )
        return_dict = {}
        face_num_count = 0
        if visual:
            scene = trimesh.Scene()
            for link_name in self.link_visuals_dict:
                mesh_transform_matrix = X_w_b @ self.current_status[
                    link_name
                ].get_matrix().detach().cpu().numpy().reshape(4, 4)
                part_mesh = (
                    self.link_visuals_dict[link_name]
                    .copy()
                    .apply_transform(mesh_transform_matrix)
                )
                part_mesh_face_num = len(part_mesh.faces)
                scene.add_geometry(part_mesh)
                face_num_count += part_mesh_face_num
            return_dict["visual"] = scene
        if collision:
            collision_scene = trimesh.Scene()
            for link_name in self.link_collisions_dict:
                mesh_transform_matrix = X_w_b @ self.current_status[
                    link_name
                ].get_matrix().detach().cpu().numpy().reshape(4, 4)
                part_mesh = (
                    self.link_collisions_dict[link_name]
                    .copy()
                    .apply_transform(mesh_transform_matrix)
                )
                part_mesh_face_num = len(part_mesh.faces)

                collision_scene.add_geometry(part_mesh)
            return_dict["collision"] = collision_scene
        return return_dict


def segment_robot_arm(rgb_image):
    """
    Use SAM to segment the robot arm from the RGB image

    Args:
        rgb_image: RGB image (H, W, 3)
        components: Dictionary containing calibration components

    Returns:
        mask: Binary mask of the robot arm (H, W)
    """
    lgr.info("Starting robot arm segmentation with SAM...")

    sam_drawer = SAMPromptDrawer(
        window_name="Robot Arm Segmentation",
        screen_scale=2.0,
        sam_checkpoint=SAM_PATH,
        device="cuda",
        model_type=SAM_TYPE,
    )

    # Reset SAM and run interactive segmentation
    sam_drawer.reset()
    mask = sam_drawer.run(rgb_image)

    if mask is not None:
        lgr.info(f"Generated mask with shape: {mask.shape}")
        lgr.info(f"Mask coverage: {mask.sum() / mask.size * 100:.2f}%")
    else:
        lgr.error("Failed to generate mask!")

    return mask


def optimize_camera_pose(
    mask_list,
    joint_values_list,
    init_X_CameraBase,
    K,
    n_epochs=200,
    lr=3e-3,
):
    """
    Optimize camera extrinsics using differentiable rendering with multiple samples

    Args:
        mask_list: List of binary masks of robot arm (each H, W)
        joint_values_list: List of robot joint values for each sample
        init_X_CameraBase: Initial camera-to-base transformation matrix
        K: Camera intrinsic matrix
        n_epochs: Number of optimization epochs
        lr: Learning rate

    Returns:
        tuple: (optimized_X_BaseCamera, loss_history)
    """
    lgr.info("Starting camera pose optimization with multiple samples...")

    renderer = NVDiffrastRenderer([480, 640])
    pk_robot_arm = PKRobotArm(urdf_path="assets/ur5_urdf/robot.urdf")

    # Pre-compute robot meshes for all joint configurations
    arm_visual_meshes = []
    for joint_values in joint_values_list:
        arm_visual_mesh = as_mesh(
            pk_robot_arm.get_state_trimesh(joint_values)["visual"]
        )
        arm_visual_meshes.append(arm_visual_mesh)

    # Convert masks to torch tensors
    target_masks = []
    for mask in mask_list:
        target_mask = torch.from_numpy(mask.astype(np.float32)).cuda()
        target_masks.append(target_mask)
    target_masks = torch.stack(target_masks, dim=0)  # [N, H, W]

    # Initialize optimization parameters
    cam_pose_params = [
        init_X_CameraBase[:3, 3],  # position
        init_X_CameraBase[:3, 0],  # x-axis
        init_X_CameraBase[:3, 1],  # y-axis
    ]
    cam_pose_params = (
        torch.from_numpy(np.array(cam_pose_params).flatten()).cuda().float()
    )
    cam_pose_params.requires_grad = True

    optimizer = torch.optim.Adam([cam_pose_params], lr=lr)

    loss_history = []
    best_loss = float("inf")
    best_pose = None

    lgr.info(
        f"Starting optimization for {n_epochs} epochs with {len(mask_list)} samples..."
    )

    for epoch in tqdm(range(n_epochs)):
        # Extract camera parameters
        cam_position = cam_pose_params[:3]  # [3]
        cam_6d = cam_pose_params[3:]  # [6]

        # Compute rotation matrix from 6D representation
        cam_rot = robust_compute_rotation_matrix_from_ortho6d(
            cam_6d.unsqueeze(0)
        ).squeeze(
            0
        )  # [3, 3]

        # Construct camera pose matrix
        current_X_CameraBase = torch.cat(
            [cam_rot, cam_position.unsqueeze(1)], dim=1
        )  # [3, 4]
        current_X_CameraBase = torch.cat(
            [current_X_CameraBase, torch.tensor([[0, 0, 0, 1]]).cuda().float()],
            dim=0,
        )  # [4, 4]

        # Render masks for all samples using current camera pose
        rendered_masks = []
        for arm_visual_mesh in arm_visual_meshes:
            rendered_mask = renderer.render_mask(
                torch.from_numpy(arm_visual_mesh.vertices).cuda().float(),
                torch.from_numpy(arm_visual_mesh.faces).cuda().int(),
                torch.from_numpy(K).cuda().float(),
                current_X_CameraBase,
            )
            rendered_masks.append(rendered_mask)

        rendered_masks = torch.stack(rendered_masks, dim=0)  # [N, H, W]

        # Compute loss (L1 difference between rendered and target masks)
        loss = torch.abs(rendered_masks - target_masks).mean()

        loss_history.append(loss.item())

        if epoch % 20 == 0:
            lgr.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        # Save best pose
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_pose = current_X_CameraBase.detach().cpu().numpy()

        # Early stopping
        if loss.item() < 1e-3:
            lgr.info(f"Converged at epoch {epoch} with loss {loss.item():.6f}")
            break

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Convert back to base-to-camera transformation
    optimized_X_CameraBase = best_pose
    optimized_X_BaseCamera = np.linalg.inv(optimized_X_CameraBase)

    lgr.info("Optimization completed!")
    lgr.info(f"Final loss: {best_loss:.6f}")
    lgr.info(f"Optimized X_BaseCamera:\n{optimized_X_BaseCamera}")

    return optimized_X_BaseCamera, loss_history


def visualize_results(
    rgb_image,
    mask,
    joint_values,
    optimized_X_BaseCamera,
    K,
):
    """
    Visualize the optimization results

    Args:
        rgb_image: Original RGB image
        mask: Target mask
        joint_values: Robot joint values
        optimized_X_BaseCamera: Optimized camera pose
        components: Dictionary containing calibration components
    """
    lgr.info("Generating visualization...")

    renderer = NVDiffrastRenderer([480, 640])
    pk_robot_arm = PKRobotArm(urdf_path="assets/ur5_urdf/robot.urdf")

    # Get robot mesh
    arm_visual_mesh = as_mesh(pk_robot_arm.get_state_trimesh(joint_values)["visual"])

    # Render mask with optimized pose
    optimized_X_CameraBase = np.linalg.inv(optimized_X_BaseCamera)
    rendered_mask = renderer.render_mask(
        torch.from_numpy(arm_visual_mesh.vertices).cuda().float(),
        torch.from_numpy(arm_visual_mesh.faces).cuda().int(),
        torch.from_numpy(K).cuda().float(),
        torch.from_numpy(optimized_X_CameraBase).cuda().float(),
    )
    rendered_mask_np = rendered_mask.detach().cpu().numpy()

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original RGB image
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("Original RGB Image")
    axes[0, 0].axis("off")

    # Target mask (from SAM)
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Target Mask (SAM)")
    axes[0, 1].axis("off")

    # Rendered mask (optimized)
    axes[1, 0].imshow(rendered_mask_np, cmap="gray")
    axes[1, 0].set_title("Rendered Mask (Optimized)")
    axes[1, 0].axis("off")

    # Difference
    diff = np.abs(rendered_mask_np - mask)
    axes[1, 1].imshow(diff, cmap="hot")
    axes[1, 1].set_title(f"Difference (MAE: {diff.mean():.4f})")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("camera_calibration_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    lgr.info("Visualization saved as 'camera_calibration_results.png'")


def run_single_frame_calibration(
    init_X_BaseCamera,
    K,
    save_path=None,
    n_epochs=200,
    lr=3e-3,
    visualize=True,
):
    """
    Run the complete multi-frame camera calibration process

    Args:
        init_X_BaseCamera: Initial guess for camera pose (4x4 matrix)
        K: Camera intrinsic matrix
        save_path: Path to save the optimized camera pose
        n_epochs: Number of optimization epochs
        lr: Learning rate
        visualize: Whether to show visualization

    Returns:
        optimized_X_BaseCamera: Optimized camera pose (4x4)
    """
    lgr.info("Starting multi-frame camera calibration...")

    try:
        with np.load("multi_frame_calibration_data.npz") as data:
            rgb_image_list = data["rgb_images"]
            joint_values_list = data["captured_joint_values"]

        # Step 1: Segment robot arm with SAM
        masks = []
        for rgb_image in rgb_image_list:
            mask = segment_robot_arm(rgb_image)
            if mask is None:
                raise ValueError("Failed to generate robot arm mask")
            masks.append(mask)

        np.save("masks.npy", np.array(masks))

        # Step 1: Load pre-computed masks
        with open("masks.npy", "rb") as f:
            mask_list = np.load(f)

        init_X_CameraBase = np.linalg.inv(init_X_BaseCamera)

        # Step 2: Optimize camera pose using multiple samples
        optimized_X_BaseCamera, loss_history = optimize_camera_pose(
            mask_list,
            joint_values_list,
            init_X_CameraBase,
            K,
            n_epochs=n_epochs,
            lr=lr,
        )

        rot_quat = R.from_matrix(optimized_X_BaseCamera[:3, :3]).as_quat(
            scalar_first=True, canonical=True
        )
        pos = optimized_X_BaseCamera[:3, 3]
        pos = pos + np.array(
            [-0.625179097, 0.0498731775, -0.0444625377]
        )  # Adjust position to table frame
        print(rot_quat, pos)

        # Step 3: Visualize results using the first sample
        if visualize:
            for i in range(len(rgb_image_list)):
                visualize_results(
                    rgb_image_list[i],
                    mask_list[i],
                    joint_values_list[i],
                    optimized_X_BaseCamera,
                    K,
                )

        return optimized_X_BaseCamera

    except Exception as e:
        lgr.error(f"Calibration failed: {e}")
        raise e


def main():
    """
    Example usage of the single-frame camera calibration functions
    """

    rot_mat = R.from_quat([0.2, -0.979, 0.0, 0.03], scalar_first=True).as_matrix()
    pos = np.array([-0.1992 + 6.25179097e-01, -0.44, 1.02])

    K = np.array(
        [
            [606.65, 0.0, 329.51],
            [0.0, 606.71, 243.67],
            [0.0, 0.0, 1.0],
        ]
    )

    # Example initial camera pose (you should replace this with your initial guess)
    init_X_BaseCamera = np.eye(4)
    init_X_BaseCamera[:3, :3] = rot_mat
    init_X_BaseCamera[:3, 3] = pos

    # Run calibration
    optimized_pose = run_single_frame_calibration(
        init_X_BaseCamera=init_X_BaseCamera,
        K=K,
        save_path="optimized_camera_pose.npy",
        n_epochs=1000,
        lr=3e-3,
        visualize=True,
    )

    print("Calibration completed successfully!")
    return optimized_pose


if __name__ == "__main__":
    main()
