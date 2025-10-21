import copy
import numpy as np
import scipy.spatial.transform as st

def unnoramalize_action_perdim(action, action_min=-1, action_max=1, maintain_last=False):
    last_val = copy.deepcopy(action[..., -1])
    res = 0.5 * (action + 1) * (action_max - action_min) + action_min
    if maintain_last:
        res[..., -1] = last_val
    return res

# Based on https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/common/pose_util.py
# convert from euler angle to rot
def euler_pose_to_pos_rot(euler_pose):
    pos = euler_pose[..., :3]
    rot = st.Rotation.from_euler('xyz', euler_pose[..., 3:], degrees=False)
    return pos, rot

def pos_rot_to_quat(pos, rot):
    shape = pos.shape[:-1]
    quat = np.zeros(shape + (7,), dtype=pos.dtype)
    quat[...,:3] = pos
    quat[...,3:] = rot.as_quat()
    return quat

# convert rot from euler angle to quat
def euler_pose_to_quat(euler_pose):
    return pos_rot_to_quat(*euler_pose_to_pos_rot(euler_pose))