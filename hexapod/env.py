"""CPU MuJoCo environment: viewing, evaluation and video."""
from collections import deque

import mujoco
import numpy as np

from . import core


class HexapodEnv:
    def __init__(self, failures=True, seed=None):
        self.model = mujoco.MjModel.from_xml_path(str(core.MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.failures = failures
        self.rng = np.random.default_rng(seed)
        self.qadr, self.vadr, self.wadr, self.contact, self.contact_r = core.addresses(self.model)
        self.terrain = core.load_terrain(self.model)
        self.external_cmd = None
        self.level = self.column = None
        self._renderer = None

    def reset(self, seed=None, level=None, column=None, yaw=None, pos=(0.0, 0.0)):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if isinstance(column, str):
            column = core.TERRAIN_TYPES.index(column)
        self.level, self.column = level, column
        rng, d = self.rng, self.data
        mujoco.mj_resetData(self.model, d)
        if level is None:
            x, y = pos
            yaw = 0.0 if yaw is None else yaw
        else:
            cx, cy = core.tile_center(level, column)
            x, y = cx + rng.uniform(-core.SPAWN_NOISE, core.SPAWN_NOISE), cy + rng.uniform(-core.SPAWN_NOISE, core.SPAWN_NOISE)
            yaw = rng.uniform(-np.pi, np.pi) if yaw is None else yaw
        d.qpos[0:3] = x, y, core.spawn_height(np, self.terrain, x, y)
        d.qpos[3:7] = np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
        d.qpos[self.qadr] += rng.uniform(-0.05, 0.05, core.N_JOINTS)
        mujoco.mj_forward(self.model, d)
        self.t = 0
        self.last_action = np.zeros(core.N_ACT)
        self.cmd_seq, self.cmd_t = self._sample_commands(level or 0)
        if self.failures:
            self.mode, self.strength, self.fail_t = self._sample_failures()
        else:
            self.mode = np.full(core.N_ACT, core.HEALTHY)
            self.strength = np.ones(core.N_ACT)
            self.fail_t = np.full(core.N_ACT, core.NEVER)
        self.q_lock = d.qpos[self.qadr].copy()
        self.air_time = np.zeros(core.N_LEGS)
        self.last_contact = np.ones(core.N_LEGS, bool)
        self.phase = 0.0
        self.history = deque([self._actor_obs()] * core.HISTORY, maxlen=core.HISTORY)
        scan = core.scan_heights(np, self.terrain, d.qpos)
        return self._observation(scan), {}

    @property
    def cmd(self):
        if self.external_cmd is not None:
            return self.external_cmd
        return core.command_at(np, self.cmd_seq, self.cmd_t, self.t)

    def set_command(self, cmd):
        self.external_cmd = None if cmd is None else np.asarray(cmd, dtype=np.float64)

    def fail_leg(self, leg, mode=core.LIMP, strength=1.0):
        self.fail_actuators(np.flatnonzero(core.ACT_LEG == leg), mode, strength)

    def fail_wheels(self, mode=core.LOCKED):
        self.fail_actuators(np.flatnonzero(core.IS_WHEEL), mode)

    def fail_actuators(self, idx, mode=core.LIMP, strength=1.0):
        self.mode[idx], self.strength[idx], self.fail_t[idx] = mode, strength, self.t

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        cmd = self.cmd
        action = core.park_brake(np, action, cmd)
        newly = self.fail_t[:core.N_JOINTS] == self.t
        self.q_lock[newly] = self.data.qpos[self.qadr][newly]
        mode = core.active_modes(np, self.mode, self.fail_t, self.t)
        d = self.data
        for _ in range(core.N_SUBSTEPS):
            tau = core.actuator_torque(np, action, d.qpos[self.qadr], d.qvel[self.vadr], d.qvel[self.wadr],
                                       mode, self.strength, self.q_lock)
            d.ctrl[:] = tau
            mujoco.mj_step(self.model, d)
        scan = core.scan_heights(np, self.terrain, d.qpos)
        ground = core.ground_level(np, scan)
        normal = core.ground_normal(np, self.terrain, d.qpos[0], d.qpos[1])
        gaps = core.foot_gaps(np, self.terrain, d.geom_xpos[self.contact], self.contact_r)
        contact = gaps < core.CONTACT_TOL
        self.air_time, landed, first = core.air_time_step(np, self.air_time, self.last_contact, contact)
        self.last_contact = contact
        self.phase = float(core.gait_step(np, self.phase, cmd))
        terms = core.reward_terms(np, d.qpos, d.qvel, self.qadr, self.vadr, self.wadr, cmd, action,
                                  self.last_action, tau, mode, ground, normal, contact, landed, first,
                                  self.phase, gaps)
        self.last_action = action
        self.t += 1
        self.history.append(self._actor_obs())
        obs = self._observation(scan)
        terminated = bool(core.fallen(np, d.qpos, ground)) or not np.isfinite(obs).all()
        info = {
            "velocity": core.body_velocity(np, d.qpos, d.qvel),
            "cmd": cmd,
            "failed": int((mode != core.HEALTHY).sum()),
            "ground": float(ground),
            "contact": contact,
            **{f"r_{k}": float(v) for k, v in terms.items()},
        }
        return obs, float(sum(terms.values())), terminated, self.t >= core.EPISODE_STEPS, info

    def render(self, width=640, height=480, distance=0.8, elevation=-16.0, azimuth=145.0):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._camera = mujoco.MjvCamera()
            self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._camera.trackbodyid = self.model.body("torso").id
        self._camera.distance, self._camera.elevation, self._camera.azimuth = distance, elevation, azimuth
        self._renderer.update_scene(self.data, self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _actor_obs(self):
        d = self.data
        return core.actor_obs(np, d.qpos, d.qvel, self.qadr, self.vadr, self.wadr,
                              self.last_action, self.cmd, self.phase).astype(np.float32)

    def _observation(self, scan):
        return np.concatenate([*self.history, core.scan_obs(np, self.data.qpos, scan).astype(np.float32)])

    def _sample_commands(self, level):
        holds = self.rng.integers(*core.CMD_HOLD, core.CMD_SEGMENTS)
        cmd_t = np.concatenate([[0], np.cumsum(holds)[:-1]])
        high = core.CMD_HIGH.copy()
        high[0] -= core.SPEED_DROP * level
        cmd = core.CMD_LOW + self.rng.random((core.CMD_SEGMENTS, 3)) * (high - core.CMD_LOW)
        cmd[self.rng.random(core.CMD_SEGMENTS) < core.P_STAND] = 0.0
        return cmd, cmd_t

    def _sample_failures(self):
        rng = self.rng
        mode = np.full(core.N_ACT, core.HEALTHY)
        strength = np.ones(core.N_ACT)
        fail_t = np.full(core.N_ACT, core.NEVER)
        kind = min(int(np.searchsorted(core.FAIL_CUM, rng.random(), side="right")), len(core.FAIL_P) - 1)
        legs = rng.permutation(core.N_LEGS)
        groups = []
        if kind == core.ONE_MOTOR:
            groups = [np.array([rng.integers(core.N_ACT)])]
        elif kind in (core.ONE_LEG, core.WHEELS_AND_LEG):
            groups = [np.flatnonzero(core.ACT_LEG == legs[0])]
        elif kind == core.TWO_LEGS:
            groups = [np.flatnonzero(core.ACT_LEG == legs[0]), np.flatnonzero(core.ACT_LEG == legs[1])]
        when = lambda: 0 if rng.random() < 0.5 else rng.integers(core.FAIL_EARLIEST, core.FAIL_LATEST)
        for idx in groups:
            v = rng.random()
            kind_ = core.LIMP if v < core.P_LIMP else core.LOCKED if v < core.P_LIMP + core.P_LOCKED else core.WEAK
            mode[idx] = kind_
            strength[idx] = rng.uniform(*core.WEAK_RANGE) if kind_ == core.WEAK else 1.0
            fail_t[idx] = when()
        if kind >= core.ALL_WHEELS:
            wheels = np.flatnonzero(core.IS_WHEEL)
            mode[wheels] = core.LIMP if rng.random() < core.P_WHEELS_LIMP else core.LOCKED
            strength[wheels] = 1.0
            fail_t[wheels] = when()
        return mode, strength, fail_t
