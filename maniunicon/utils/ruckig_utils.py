from ruckig import InputParameter, OutputParameter, Result, Ruckig
import math
import numpy as np
import time


def init_ruckig(
    init_q: np.ndarray, init_dq: np.ndarray, DT: float
) -> tuple[Ruckig, InputParameter, OutputParameter, Result]:
    num_joints = init_q.shape[0]
    otg = Ruckig(num_joints, DT)
    otg_inp = InputParameter(num_joints)
    otg_out = OutputParameter(num_joints)
    otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
    # otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [math.radians(450)]
    otg_inp.current_position = init_q.copy()
    otg_inp.current_velocity = init_dq.copy()
    otg_inp.target_position = init_q.copy()
    otg_inp.target_velocity = np.zeros(num_joints)
    otg_res = Result.Finished
    return otg, otg_inp, otg_out, otg_res


def update_ruckig(
    otg: Ruckig,
    otg_inp: InputParameter,
    otg_out: OutputParameter,
    otg_res: Result,
    target_q: np.ndarray,
    last_command_time: float,
    dt: float,
):
    if target_q is not None:
        otg_inp.target_position = target_q.copy()
        last_command_time = time.time()
        otg_res = Result.Working

    # Maintain current pose if command stream is disrupted
    if time.time() - last_command_time > 2.5 * dt:
        otg_inp.target_position = otg_out.new_position
        otg_res = Result.Working

    # Update OTG
    if otg_res == Result.Working:
        otg_res = otg.update(otg_inp, otg_out)
        otg_out.pass_to_input(otg_inp)

        return otg_out.new_position, otg_out.new_velocity, last_command_time
    else:
        return None, None, last_command_time
