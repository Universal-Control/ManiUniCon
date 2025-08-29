import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

import cv2
import subprocess
from enum import Enum

from segment_anything import sam_model_registry, SamPredictor


SAM_TYPE = "vit_h"
# SAM_PATH = "/home/bytedance/zhengbang/segment-anything/ckpt/sam_vit_h_4b8939.pth"
SAM_PATH = "/home/nuc001/zhengbang/segment-anything/ckpt/sam_vit_h_4b8939.pth"


def to_array(x, dtype=float):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)
    elif isinstance(x, list):
        return [to_array(a) for a in x]
    elif isinstance(x, dict):
        return {k: to_array(v) for k, v in x.items()}
    else:
        return x


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith("4"):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith("3"):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError("cv2 must be either version 3 or 4 to call this method")

    return contours, hierarchy


class DrawingMode(Enum):
    Box = 0
    Point = 1


def vis_mask(
    img,
    mask,
    color=[255, 255, 255],
    alpha=0.4,
    show_border=True,
    border_alpha=0.5,
    border_thick=1,
    border_color=None,
):
    """Visualizes a single binary mask."""
    if isinstance(mask, torch.Tensor):
        mask = to_array(mask > 0).astype(np.uint8)
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += [alpha * x for x in color]

    if show_border:
        contours, _ = findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # contours = [c for c in contours if c.shape[0] > 10]
        if border_color is None:
            border_color = color
        if not isinstance(border_color, list):
            border_color = border_color.tolist()
        if border_alpha < 1:
            with_border = img.copy()
            cv2.drawContours(
                with_border, contours, -1, border_color, border_thick, cv2.LINE_AA
            )
            img = (1 - border_alpha) * img + border_alpha * with_border
        else:
            cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)
    return img.astype(np.uint8)


class SAMPromptDrawer(object):
    def __init__(
        self,
        window_name="Prompt Drawer",
        screen_scale=1.0,
        sam_checkpoint="",
        device="cuda",
        model_type="default",
    ):
        self.window_name = window_name
        self.reset()
        self.screen_scale = screen_scale

        # Initialize the SAM predictor
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def reset(self):
        self.done = False
        self.drawing = False
        self.current = (0, 0)
        self.box = np.zeros([4], dtype=np.float32)
        self.points = np.empty((0, 2))
        self.labels = np.empty([0], dtype=int)
        self.mask = None
        self.mode = DrawingMode.Box
        self.boxes = np.zeros([0, 4], dtype=np.float32)
        self.box_labels = np.empty([0], dtype=int)

    def on_mouse(self, event, x, y, flags, user_param):
        # Mouse callback for every mouse event
        if self.done:
            return
        if self.mode == DrawingMode.Box:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    self.box_labels = np.hstack([self.box_labels, 0])
                else:
                    self.box_labels = np.hstack([self.box_labels, 1])
                self.boxes = np.vstack([self.boxes, [x, y, x, y]])
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.boxes[-1, 2] = x
                self.boxes[-1, 3] = y
                self.detect()  # Recalculate mask
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.boxes[-1, 2] = x
                    self.boxes[-1, 3] = y
        elif self.mode == DrawingMode.Point:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points = np.vstack([self.points, [x, y]])
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    label = 0
                else:
                    label = 1
                self.labels = np.hstack([self.labels, label])
                self.detect()  # Recalculate mask
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.points = np.vstack([self.points, [x, y]])
                self.labels = np.hstack([self.labels, 1])
                self.detect()  # Recalculate mask

    def detect(self):
        # Prepare inputs for the predictor
        if len(self.points) != 0:
            input_point = self.points / self.ratio
            input_label = self.labels.astype(int)
        else:
            input_point = None
            input_label = None

        # Initialize the final mask
        final_mask = None

        # Process each box
        if len(self.boxes) == 0:
            # If no boxes are present, use only points for detection
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=None,
                multimask_output=True,
            )
            maxidx = np.argmax(scores)
            final_mask = masks[maxidx].copy()
        else:
            # Iterate through all boxes and calculate masks
            for i in range(len(self.boxes)):
                box = self.boxes[i]
                box_label = self.box_labels[i]

                if np.all(box == 0):
                    box = None
                else:
                    box = box / self.ratio

                # Generate masks for each box
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=box,
                    multimask_output=True,
                )
                maxidx = np.argmax(scores)
                mask = masks[maxidx]

                # Combine masks logically based on the labels
                if final_mask is None:
                    final_mask = mask.copy()
                else:
                    if box_label == 0:
                        final_mask = np.logical_and(final_mask, ~mask)
                    else:
                        final_mask = np.logical_or(final_mask, mask)

        # Update the mask attribute
        if final_mask is not None:
            self.mask = final_mask.copy()
        elif self.mask is not None:
            self.mask = np.zeros_like(self.mask)

    def run(self, rgb):
        self.rgb = rgb
        self.predictor.set_image(rgb)
        image_to_show = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image_h, image_w = image_to_show.shape[:2]

        if not hasattr(self, "ratio"):
            output = subprocess.check_output(["xrandr"]).decode("utf-8")
            current_mode = [line for line in output.splitlines() if "*" in line][0]
            screen_width, screen_height = [
                int(x) for x in current_mode.split()[0].split("x")
            ]
            scale = self.screen_scale
            screen_w = int(screen_width / scale)
            screen_h = int(screen_height / scale)

            ratio = min(screen_w / image_w, screen_h / image_h)
            self.ratio = ratio
        target_size = (int(image_w * self.ratio), int(image_h * self.ratio))
        image_to_show = cv2.resize(image_to_show, target_size)

        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, image_to_show)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while not self.done:
            tmp = image_to_show.copy()
            tmp = cv2.circle(
                tmp, self.current, radius=2, color=(0, 0, 255), thickness=-1
            )
            for box, box_label in zip(self.boxes, self.box_labels):
                color = (0, 255, 0) if box_label == 1 else (0, 0, 255)
                cv2.rectangle(
                    tmp,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    2,
                )
            if self.points.shape[0] > 0:
                for ptidx, pt in enumerate(self.points):
                    color = (0, 255, 0) if self.labels[ptidx] == 1 else (0, 0, 255)
                    tmp = cv2.circle(
                        tmp,
                        (int(pt[0]), int(pt[1])),
                        radius=5,
                        color=color,
                        thickness=-1,
                    )
            if self.mask is not None:
                mask_to_show = cv2.resize(
                    self.mask.astype(np.uint8), target_size
                ).astype(bool)
                tmp = vis_mask(
                    tmp, mask_to_show.astype(np.uint8), color=[0, 255, 0], alpha=0.5
                ).astype(np.uint8)
            cv2.imshow(self.window_name, tmp)
            waittime = 50
            key = cv2.waitKey(waittime)
            if key == 27 or key == 13:  # ESC hit
                self.done = True
            elif key == ord("r"):
                print("Reset")
                self.reset()
            elif key == ord("p"):
                print("Switch to point mode")
                self.mode = DrawingMode.Point
            elif key == ord("b"):
                print("Switch to box mode")
                self.mode = DrawingMode.Box
            elif key == ord("z"):
                print("Undo")
                if self.mode == DrawingMode.Point and len(self.points) > 0:
                    self.points = self.points[:-1]
                    self.labels = self.labels[:-1]
                    self.detect()
                elif self.mode == DrawingMode.Box and len(self.boxes) > 0:
                    self.boxes = self.boxes[:-1]
                    self.box_labels = self.box_labels[:-1]
                    self.detect()
        cv2.destroyWindow(self.window_name)
        return self.mask

    def track(self, rgb):
        self.rgb = rgb
        out_obj_ids, out_mask_logits = self.predictor.track(self.rgb)
        self.mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]

        return self.mask

    def close(self):
        del self.predictor
        torch.cuda.empty_cache()


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    assert (torch.stack([torch.det(mat) for mat in matrix]) < 0).sum() == 0
    return matrix


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def K_to_projection(K, H, W, n=0.001, f=10.0):  # near and far plane
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    proj = (
        torch.tensor(
            [
                [2 * fu / W, 0, -2 * cu / W + 1, 0],
                [0, 2 * fv / H, 2 * cv / H - 1, 0],
                [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                [0, 0, -1, 0],
            ]
        )
        .cuda()
        .float()
    )
    return proj


def transform_pos(mtx, pos):
    """mtx: 4x4, pos: Nx3
    mtx means the transformation from world to camera space
    pos means the position in world space
    """
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


class NVDiffrastRenderer:
    def __init__(self, image_size):
        """
        image_size: H,W
        """
        # self.
        self.H, self.W = image_size
        self.resolution = image_size
        blender2opencv = (
            torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            .float()
            .cuda()
        )
        self.opencv2blender = torch.inverse(blender2opencv)
        self.glctx = dr.RasterizeCudaContext()

    def render_mask(self, verts, faces, K, object_pose, anti_aliasing=True):
        """
        @param verts: N,3, torch.tensor, float, cuda
        @param faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        @param object_pose: 4,4 torch.tensor, float, cuda
        @return: mask: 0 to 1, HxW torch.cuda.FloatTensor
        """
        proj = K_to_projection(K, self.H, self.W)

        pose = self.opencv2blender @ object_pose

        pos_clip = transform_pos(proj @ pose, verts)

        rast_out, _ = dr.rasterize(
            self.glctx, pos_clip, faces, resolution=self.resolution
        )
        if anti_aliasing:
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
            color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            mask = rast_out[0, :, :, 2] > 0
        mask = torch.flip(mask, dims=[0])
        return mask

    def batch_render_mask(self, verts, faces, K, anti_aliasing=True):
        """
        @param batch_verts: N,3, torch.tensor, float, cuda
        @param batch_faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        # @param batch_object_poses: N,4,4 torch.tensor, float, cuda
        @return: mask: 0 to 1, HxW torch.cuda.FloatTensor
        """
        proj = K_to_projection(K, self.H, self.W)

        pose = self.opencv2blender

        pos_clip = transform_pos(proj @ pose, verts)

        rast_out, _ = dr.rasterize(
            self.glctx, pos_clip, faces, resolution=self.resolution
        )
        if anti_aliasing:
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
            color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            mask = rast_out[0, :, :, 2] > 0
        mask = torch.flip(mask, dims=[0])
        return mask
