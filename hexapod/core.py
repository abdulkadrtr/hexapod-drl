"""Shared robot, terrain, observation and reward definitions (NumPy and JAX)."""
from pathlib import Path
from typing import NamedTuple

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "hexapod.xml"

LEGS = ("LF", "LM", "LR", "RF", "RM", "RR")
WHEEL_LEGS = ("LF", "LR", "RF", "RR")
JOINTS = tuple(f"{leg}_{seg}" for leg in LEGS for seg in ("coxa", "femur", "tibia"))
WHEELS = tuple(f"{leg}_wheel" for leg in WHEEL_LEGS)
ACTUATORS = JOINTS + WHEELS
N_JOINTS, N_WHEELS, N_ACT, N_LEGS = len(JOINTS), len(WHEELS), len(ACTUATORS), len(LEGS)
ACT_LEG = np.array([i // 3 for i in range(N_JOINTS)] + [LEGS.index(leg) for leg in WHEEL_LEGS])
CONTACT_GEOMS = tuple(f"{leg}_tire" if leg in WHEEL_LEGS else f"{leg}_foot" for leg in LEGS)
WHEEL_LEG_IDX = np.array([LEGS.index(leg) for leg in WHEEL_LEGS])
HAS_WHEEL = np.array([leg in WHEEL_LEGS for leg in LEGS])
LEG_WHEEL = np.array([WHEEL_LEGS.index(leg) if leg in WHEEL_LEGS else 0 for leg in LEGS])
IS_WHEEL = np.arange(N_ACT) >= N_JOINTS

CTRL_DT = 0.02
N_SUBSTEPS = 5
KP, KD = 10.0, 0.1
WHEEL_KV = 0.05
WHEEL_SPEED_MAX = 35.0
TORQUE_LIMITS = np.array([2.0] * N_JOINTS + [0.6] * N_WHEELS)
NO_LOAD_SPEED = np.array([8.0] * N_JOINTS + [45.0] * N_WHEELS)
ACTION_SCALE = 0.6
SPAWN_HEIGHT = 0.145
NOMINAL_HEIGHT = 0.13
EPISODE_STEPS = 1000

TILE = 2.0
LEVELS = 8
TERRAIN_TYPES = ("gravel", "ramp", "bumps", "rubble", "blocks", "stairs", "trenches", "mixed")
N_TYPES = len(TERRAIN_TYPES)
START_X = 0.5
BORDER = 1.0
INIT_LEVEL_MAX = 2
SPAWN_NOISE = 0.15
P_EDGE_SPAWN = 0.5
EDGE_OFFSET = 0.9
PROMOTE_DIST = 1.0
DEMOTE_RATIO = 0.25

CMD_LOW = np.array([-0.5, -0.2, -1.2])
CMD_HIGH = np.array([1.3, 0.2, 1.2])
CMD_SCALE = 1.0 / np.maximum(-CMD_LOW, CMD_HIGH)
SPEED_DROP = 0.08
CMD_HOLD = (200, 300)
CMD_SEGMENTS = EPISODE_STEPS // CMD_HOLD[0] + 1
P_STAND = 0.1

HEALTHY, LIMP, LOCKED, WEAK = 0, 1, 2, 3
FAIL_P = np.array([0.25, 0.15, 0.20, 0.10, 0.25, 0.05])
FAIL_CUM = np.cumsum(FAIL_P)
NONE, ONE_MOTOR, ONE_LEG, TWO_LEGS, ALL_WHEELS, WHEELS_AND_LEG = range(6)
P_LIMP, P_LOCKED = 0.5, 0.25
P_WHEELS_LIMP = 0.5
WEAK_RANGE = (0.3, 0.6)
FAIL_EARLIEST, FAIL_LATEST = 50, 750
NEVER = 10**6
GEAR_FRICTION = np.array([0.08] * N_JOINTS + [0.005] * N_WHEELS)
GEAR_DAMPING = np.array([0.02] * N_JOINTS + [0.002] * N_WHEELS)

FRICTION_RANGE = (0.6, 1.2)
SOFTNESS_RANGE = (0.01, 0.03)

SCAN_X = np.round(np.arange(-0.2, 0.401, 0.05), 3)
SCAN_Y = np.round(np.arange(-0.2, 0.201, 0.1), 3)
SCAN_POINTS = np.array([(x, y) for x in SCAN_X for y in SCAN_Y])
N_SCAN = len(SCAN_POINTS)
FOOTPRINT = np.flatnonzero((np.abs(SCAN_POINTS[:, 0]) < 0.151) & (np.abs(SCAN_POINTS[:, 1]) < 0.101))
SPAWN_GRID = np.array([(x, y) for x in np.linspace(-0.2, 0.2, 5) for y in np.linspace(-0.2, 0.2, 5)])
NORMAL_SPAN = 0.12
CONTACT_TOL = 0.006

GAIT_FREQ = (1.5, 2.0)
GAIT_FAST = 0.45
WALK_MAX = 0.5
GAIT_OFFSET = np.array([0.0 if leg in ("LF", "RM", "LR") else 0.5 for leg in LEGS])
GAIT_MARGIN = 0.05
SWING_HEIGHT = 0.05
HOP_MIN = 0.2

VEL_NOISE = 0.05
HISTORY = 10
ACTOR_OBS = 3 + 3 + 3 + 2 * N_JOINTS + N_WHEELS + N_ACT + 3 + 2
HIST_SIZE = ACTOR_OBS * HISTORY
OBS_SIZE = HIST_SIZE + N_SCAN
CRITIC_OBS = ACTOR_OBS + N_SCAN + 3 + 1 + 3 + 3 * N_ACT + 2 * N_LEGS
REWARD_KEYS = ("track_lin", "track_yaw", "yaw_drift", "upright", "bounce", "smooth", "effort", "alive", "wheels_down",
               "gait", "clearance", "wheel_hop", "posture", "height", "stand_still", "stand_pose", "stand_feet")


class Terrain(NamedTuple):
    heights: object
    x0: float
    y0: float
    cell: float


def addresses(model):
    for i, name in enumerate(ACTUATORS):
        assert model.actuator(i).trnid[0] == model.joint(name).id, f"actuator {i} does not drive joint {name}"
    qadr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
    vadr = np.array([model.joint(name).dofadr[0] for name in JOINTS])
    wadr = np.array([model.joint(name).dofadr[0] for name in WHEELS])
    contact = np.array([model.geom(name).id for name in CONTACT_GEOMS])
    return qadr, vadr, wadr, contact, model.geom_size[contact, 0].copy()


def load_terrain(model):
    gid = model.geom("terrain").id
    hid = model.geom_dataid[gid]
    nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    size, pos, adr = model.hfield_size[hid], model.geom_pos[gid], model.hfield_adr[hid]
    heights = model.hfield_data[adr:adr + nrow * ncol].reshape(nrow, ncol) * size[2] + pos[2]
    cell = 2.0 * size[0] / (ncol - 1)
    assert abs(2.0 * size[1] / (nrow - 1) - cell) < 1e-9, "terrain cells must be square"
    assert abs(pos[0] - size[0] - (START_X - BORDER)) < 1e-6 and abs(2 * size[0] - LEVELS * TILE - 2 * BORDER) < 1e-6 \
        and abs(2 * size[1] - N_TYPES * TILE - 2 * BORDER) < 1e-6, "hexapod.xml terrain does not match the layout in hexapod/core.py"
    return Terrain(np.asarray(heights, np.float64), float(pos[0] - size[0]), float(pos[1] - size[1]), float(cell))


def tile_center(level, column):
    return START_X + (level + 0.5) * TILE, (column + 0.5 - N_TYPES / 2) * TILE


def ground_height(xp, terrain, x, y):
    h = terrain.heights
    nrow, ncol = h.shape
    fx = xp.clip((x - terrain.x0) / terrain.cell, 0.0, ncol - 1.001)
    fy = xp.clip((y - terrain.y0) / terrain.cell, 0.0, nrow - 1.001)
    ix, iy = xp.floor(fx).astype(xp.int32), xp.floor(fy).astype(xp.int32)
    tx, ty = fx - ix, fy - iy
    return ((1 - ty) * ((1 - tx) * h[iy, ix] + tx * h[iy, ix + 1])
            + ty * ((1 - tx) * h[iy + 1, ix] + tx * h[iy + 1, ix + 1]))


def heading(xp, q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def scan_heights(xp, terrain, qpos):
    yaw = heading(xp, qpos[3:7])
    c, s = xp.cos(yaw), xp.sin(yaw)
    px = qpos[0] + c * SCAN_POINTS[:, 0] - s * SCAN_POINTS[:, 1]
    py = qpos[1] + s * SCAN_POINTS[:, 0] + c * SCAN_POINTS[:, 1]
    return ground_height(xp, terrain, px, py)


def ground_level(xp, scan):
    return xp.mean(scan[FOOTPRINT])


def ground_normal(xp, terrain, x, y):
    d = NORMAL_SPAN
    h = ground_height(xp, terrain, x + np.array([d, -d, 0.0, 0.0]), y + np.array([0.0, 0.0, d, -d]))
    gx, gy = (h[0] - h[1]) / (2 * d), (h[2] - h[3]) / (2 * d)
    inv = 1.0 / xp.sqrt(1.0 + gx * gx + gy * gy)
    return xp.stack([-gx * inv, -gy * inv, inv])


def spawn_height(xp, terrain, x, y):
    return xp.max(ground_height(xp, terrain, x + SPAWN_GRID[:, 0], y + SPAWN_GRID[:, 1])) + SPAWN_HEIGHT


def foot_gaps(xp, terrain, pos, radius):
    return pos[:, 2] - radius - ground_height(xp, terrain, pos[:, 0], pos[:, 1])


def quat_to_mat(xp, q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return xp.stack([
        xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        xp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        xp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def command_at(xp, cmd_seq, cmd_t, t):
    return cmd_seq[xp.sum(cmd_t <= t) - 1]


def active_modes(xp, mode, fail_t, t):
    return xp.where(t >= fail_t, mode, HEALTHY)


def leg_roles(xp, mode):
    working = (mode == HEALTHY) | (mode == WEAK)
    joints_ok = xp.all(working[:N_JOINTS].reshape(N_LEGS, 3), axis=1)
    wheel_ok = working[N_JOINTS:][LEG_WHEEL] & HAS_WHEEL
    return joints_ok & wheel_ok, joints_ok & ~wheel_ok


def actuator_torque(xp, action, q, qd, w, mode, strength, q_lock):
    target = xp.where(mode[:N_JOINTS] == LOCKED, q_lock, action[:N_JOINTS] * ACTION_SCALE)
    wheel_target = xp.where(mode[N_JOINTS:] == LOCKED, 0.0, action[N_JOINTS:] * WHEEL_SPEED_MAX)
    tau = xp.concatenate([KP * (target - q) - KD * qd, WHEEL_KV * (wheel_target - w)])
    speed = xp.concatenate([qd, w])
    limit = TORQUE_LIMITS * strength
    tau = xp.clip(tau, -limit * xp.clip(1.0 + speed / NO_LOAD_SPEED, 0.0, 1.0),
                  limit * xp.clip(1.0 - speed / NO_LOAD_SPEED, 0.0, 1.0))
    unpowered = -GEAR_FRICTION * xp.tanh(speed / 0.05) - GEAR_DAMPING * speed
    return xp.where(mode == LIMP, unpowered, tau)


def park_brake(xp, action, cmd):
    standing = xp.sum(xp.abs(cmd)) < 0.05
    return xp.where(standing & IS_WHEEL, 0.0, action)


def gait_step(xp, phase, cmd):
    speed = xp.sqrt(xp.square(cmd[0]) + xp.square(cmd[1]))
    freq = GAIT_FREQ[0] + (GAIT_FREQ[1] - GAIT_FREQ[0]) * xp.clip(speed / GAIT_FAST, 0.0, 1.0)
    moving = xp.sum(xp.abs(cmd)) >= 0.05
    return xp.where(moving, (phase + freq * CTRL_DT) % 1.0, phase)


def air_time_step(xp, air_time, last_contact, contact):
    contact_filt = contact | last_contact
    first = (air_time > 0.0) & contact_filt
    air_time = air_time + CTRL_DT
    landed = xp.where(first, air_time, 0.0)
    return xp.where(contact_filt, 0.0, air_time), landed, first


def actor_obs(xp, qpos, qvel, qadr, vadr, wadr, last_action, cmd, phase, vel_noise=0.0):
    rot = quat_to_mat(xp, qpos[3:7])
    angle = 2.0 * np.pi * phase
    return xp.concatenate([
        2.0 * (rot.T @ qvel[0:3] + vel_noise),
        0.25 * qvel[3:6],
        -rot[2],
        qpos[qadr],
        0.05 * qvel[vadr],
        0.02 * qvel[wadr],
        last_action,
        cmd * CMD_SCALE,
        xp.stack([xp.sin(angle), xp.cos(angle)]),
    ])


def scan_obs(xp, qpos, scan):
    return 5.0 * xp.clip(qpos[2] - scan - NOMINAL_HEIGHT, -0.3, 0.3)


def critic_obs(xp, actor_now, scan_o, qpos, qvel, ground, normal, mode, strength, contact, air_time):
    rot = quat_to_mat(xp, qpos[3:7])
    return xp.concatenate([
        actor_now,
        scan_o,
        2.0 * (rot.T @ qvel[0:3]),
        5.0 * (qpos[2:3] - ground),
        rot.T @ normal,
        xp.where(mode == LIMP, 1.0, 0.0),
        xp.where(mode == LOCKED, 1.0, 0.0),
        strength,
        xp.where(contact, 1.0, 0.0),
        5.0 * air_time,
    ])


def body_velocity(xp, qpos, qvel):
    v = quat_to_mat(xp, qpos[3:7]).T @ qvel[0:3]
    return xp.concatenate([v[:2], qvel[5:6]])


def reward_terms(xp, qpos, qvel, qadr, vadr, wadr, cmd, action, last_action, tau, mode,
                 ground, normal, contact, landed, first, phase, gaps):
    rot = quat_to_mat(xp, qpos[3:7])
    v = rot.T @ qvel[0:3]
    w = qvel[3:6]
    lin_tol = 0.02 + 0.05 * (xp.square(cmd[0]) + xp.square(cmd[1]))
    yaw_tol = 0.05 + 0.1 * xp.square(cmd[2])
    yaw_err = xp.abs(w[2] - cmd[2])
    rolling, walking = leg_roles(xp, mode)
    usable = rolling | walking
    working = (mode == HEALTHY) | (mode == WEAK)
    moving = xp.sum(xp.abs(cmd)) >= 0.05
    fidget = xp.sum(xp.square(qvel[vadr])) + 0.5 * xp.sum(xp.square(qvel[wadr]))
    off_pose = xp.sum(xp.where(working[:N_JOINTS], xp.abs(qpos[qadr]), 0.0))
    lifted = xp.sum(xp.where(usable & ~contact, 1.0, 0.0))
    speed_cmd = xp.sqrt(xp.square(cmd[0]) + xp.square(cmd[1]))
    no_rolling = xp.sum(xp.where(rolling, 1.0, 0.0)) < 0.5
    rhythmic = walking & moving & (HAS_WHEEL | (speed_cmd <= WALK_MAX) | no_rolling)
    leg_phase = (phase + GAIT_OFFSET) % 1.0
    swing = leg_phase < 0.5
    certain = xp.abs(leg_phase % 0.5 - 0.25) < 0.25 - GAIT_MARGIN
    in_rhythm = xp.where(swing, ~contact, contact)
    gait = xp.sum(xp.where(rhythmic & certain, xp.where(in_rhythm, 1.0, -1.0), 0.0))
    clearance = xp.sum(xp.where(rhythmic & certain & swing, xp.clip(gaps, 0.0, SWING_HEIGHT) / SWING_HEIGHT, 0.0))
    hops = xp.where(rolling & first, xp.clip(HOP_MIN - landed, 0.0, HOP_MIN), 0.0)
    return {
        "track_lin": 2.0 * xp.exp(-(xp.square(v[0] - cmd[0]) + xp.square(v[1] - cmd[1])) / lin_tol),
        "track_yaw": 1.5 * xp.exp(-xp.square(yaw_err) / yaw_tol),
        "yaw_drift": -0.5 * yaw_err,
        "upright": -2.0 * (1.0 - rot[:, 2] @ normal),
        "bounce": -1.0 * xp.square(v[2]) - 0.05 * (xp.square(w[0]) + xp.square(w[1])),
        "smooth": -0.02 * xp.sum(xp.square(action - last_action)),
        "effort": -2e-3 * xp.sum(xp.square(tau)),
        "alive": 0.25 + 0.0 * rot[2, 2],
        "wheels_down": 0.1 * xp.sum(xp.where(rolling & contact, 1.0, 0.0)),
        "gait": 0.3 * gait,
        "clearance": 0.1 * clearance,
        "wheel_hop": -1.0 * xp.where(moving, xp.sum(hops), 0.0),
        "posture": -0.1 * xp.sum(xp.where(working[:N_JOINTS], xp.square(qpos[qadr]), 0.0)),
        "height": -20.0 * xp.square(qpos[2] - ground - NOMINAL_HEIGHT),
        "stand_still": -0.02 * xp.where(moving, 0.0, fidget),
        "stand_pose": -1.0 * xp.where(moving, 0.0, off_pose),
        "stand_feet": -0.5 * xp.where(moving, 0.0, lifted),
    }


def fallen(xp, qpos, ground):
    up = 1.0 - 2.0 * (xp.square(qpos[4]) + xp.square(qpos[5]))
    return (qpos[2] - ground < 0.05) | (up < 0.5)
