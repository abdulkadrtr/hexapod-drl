"""MJX (GPU) environment with terrain curriculum, used for training."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
from mujoco import mjx
from mujoco.mjx._src import collision_convex, collision_driver
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src.types import GeomType

from . import core


@collision_convex.collider(ncon=1)
def _hfield_sphere_single(h, s, subgrid_size):
    dist, pos, n = collision_convex._hfield_collision(collision_convex._sphere_convex, h, s, jnp.max(s.size),
                                                      subgrid_size)
    i = jnp.argmin(dist)
    return dist[i], h.mat @ pos[i] + h.pos, mjx_math.make_frame(h.mat @ n[i])


collision_driver._COLLISION_FUNC[(GeomType.HFIELD, GeomType.SPHERE)] = _hfield_sphere_single

INFO_KEYS = ("rng", "t", "cmd_seq", "cmd_t", "mode", "strength", "fail_t", "q_lock", "last_action",
             "air_time", "last_contact", "level", "column", "spawn_xy", "cmd_disp", "friction", "softness", "phase")
METRIC_KEYS = tuple(f"r_{k}" for k in core.REWARD_KEYS) + (
    "vx", "vy", "wz", "lin_err", "yaw_err", "failed", "wheels_dead", "level_per_step")
CURRICULUM = {"levels": np.full(core.LEVELS, np.nan)}


def _record_levels(histogram):
    CURRICULUM["levels"] = np.asarray(histogram, np.float64)


class HexapodMJX(Env):
    def __init__(self, failures=True, curriculum=True, randomize=None, init_level_max=core.INIT_LEVEL_MAX):
        self.mj_model = mujoco.MjModel.from_xml_path(str(core.MODEL_PATH))
        self.mjx_model = mjx.put_model(self.mj_model)
        self.qadr, self.vadr, self.wadr, self.contact, self.contact_r = core.addresses(self.mj_model)
        terrain = core.load_terrain(self.mj_model)
        self.terrain = terrain._replace(heights=jnp.asarray(terrain.heights, jnp.float32))
        self.failures, self.curriculum = failures, curriculum
        self.randomize = curriculum if randomize is None else randomize
        self.init_level_max = init_level_max

    @property
    def observation_size(self):
        return {"state": (core.OBS_SIZE,), "privileged_state": (core.CRITIC_OBS,)}

    @property
    def action_size(self):
        return core.N_ACT

    @property
    def backend(self):
        return "mjx"

    def reset(self, rng):
        rng, k_level, k_column = jax.random.split(rng, 3)
        top = self.init_level_max if self.curriculum else core.LEVELS - 1
        level = jax.random.randint(k_level, (), 0, top + 1)
        column = jax.random.randint(k_column, (), 0, core.N_TYPES)
        return self.reset_at(rng, level, column)

    def reset_at(self, rng, level, column):
        rng, k_pose, k_xy, k_yaw, k_cmd, k_fail, k_friction, k_soft, k_edge, k_side, k_phase = jax.random.split(rng, 11)
        cx, cy = core.tile_center(level, column)
        edge = jax.random.uniform(k_edge) < core.P_EDGE_SPAWN
        side = jnp.where(jax.random.uniform(k_side) < 0.5, -1.0, 1.0)
        cx = cx + jnp.where(edge, side * core.EDGE_OFFSET, 0.0)
        xy = jnp.stack([cx, cy]) + jax.random.uniform(k_xy, (2,), minval=-core.SPAWN_NOISE, maxval=core.SPAWN_NOISE)
        yaw = jax.random.uniform(k_yaw, minval=-jnp.pi, maxval=jnp.pi)
        z = core.spawn_height(jnp, self.terrain, xy[0], xy[1])
        qpos = jnp.asarray(self.mj_model.qpos0, jnp.float32)
        qpos = qpos.at[0:3].set(jnp.stack([xy[0], xy[1], z]))
        qpos = qpos.at[3:7].set(jnp.stack([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)]))
        qpos = qpos.at[self.qadr].add(jax.random.uniform(k_pose, (core.N_JOINTS,), minval=-0.05, maxval=0.05))
        data = mjx.make_data(self.mj_model).replace(qpos=qpos)
        cmd_seq, cmd_t = self._sample_commands(k_cmd, level)
        if self.failures:
            mode, strength, fail_t = self._sample_failures(k_fail)
        else:
            mode = jnp.full(core.N_ACT, core.HEALTHY, jnp.int32)
            strength = jnp.ones(core.N_ACT)
            fail_t = jnp.full(core.N_ACT, core.NEVER, jnp.int32)
        t = jnp.zeros((), jnp.int32)
        last_action = jnp.zeros(core.N_ACT)
        contact = jnp.ones(core.N_LEGS, bool)
        air_time = jnp.zeros(core.N_LEGS)
        phase = jax.random.uniform(k_phase)
        actor_now = core.actor_obs(jnp, qpos, data.qvel, self.qadr, self.vadr, self.wadr, last_action,
                                   core.command_at(jnp, cmd_seq, cmd_t, t), phase)
        obs = self._obs(jnp.tile(actor_now, core.HISTORY), actor_now, qpos, data.qvel,
                        core.active_modes(jnp, mode, fail_t, t), strength, contact, air_time)
        info = {"rng": rng, "t": t, "cmd_seq": cmd_seq, "cmd_t": cmd_t, "mode": mode, "strength": strength,
                "fail_t": fail_t, "q_lock": qpos[self.qadr], "last_action": last_action, "air_time": air_time,
                "last_contact": contact, "level": jnp.asarray(level, jnp.int32), "column": jnp.asarray(column, jnp.int32),
                "spawn_xy": xy, "cmd_disp": jnp.zeros(2), "phase": phase}
        if self.randomize:
            info["friction"] = jax.random.uniform(k_friction, minval=core.FRICTION_RANGE[0], maxval=core.FRICTION_RANGE[1])
            info["softness"] = jax.random.uniform(k_soft, minval=core.SOFTNESS_RANGE[0], maxval=core.SOFTNESS_RANGE[1])
        else:
            info["friction"], info["softness"] = jnp.ones(()), jnp.full((), 0.02)
        metrics = {key: jnp.zeros(()) for key in METRIC_KEYS}
        return State(pipeline_state=data, obs=obs, reward=jnp.zeros(()), done=jnp.zeros(()), metrics=metrics, info=info)

    def step(self, state, action):
        info = state.info
        t = info["t"]
        action = jnp.clip(action, -1.0, 1.0)
        cmd = core.command_at(jnp, info["cmd_seq"], info["cmd_t"], t)
        action = core.park_brake(jnp, action, cmd)
        data = state.pipeline_state
        q_lock = jnp.where(info["fail_t"][:core.N_JOINTS] == t, data.qpos[self.qadr], info["q_lock"])
        mode = core.active_modes(jnp, info["mode"], info["fail_t"], t)
        model = self.mjx_model
        if self.randomize:
            model = model.replace(geom_friction=model.geom_friction.at[:, 0].set(info["friction"]),
                                  geom_solref=model.geom_solref.at[:, 0].set(info["softness"]))

        def substep(data, _):
            tau = core.actuator_torque(jnp, action, data.qpos[self.qadr], data.qvel[self.vadr], data.qvel[self.wadr],
                                       mode, info["strength"], q_lock)
            return mjx.step(model, data.replace(ctrl=tau)), tau

        data, taus = jax.lax.scan(substep, data, None, length=core.N_SUBSTEPS)
        qpos, qvel = data.qpos, data.qvel
        scan = core.scan_heights(jnp, self.terrain, qpos)
        ground = core.ground_level(jnp, scan)
        normal = core.ground_normal(jnp, self.terrain, qpos[0], qpos[1])
        gaps = core.foot_gaps(jnp, self.terrain, data.geom_xpos[self.contact], self.contact_r)
        contact = gaps < core.CONTACT_TOL
        air_time, landed, first = core.air_time_step(jnp, info["air_time"], info["last_contact"], contact)
        phase = core.gait_step(jnp, info["phase"], cmd)
        terms = core.reward_terms(jnp, qpos, qvel, self.qadr, self.vadr, self.wadr, cmd, action, info["last_action"],
                                  taus[-1], mode, ground, normal, contact, landed, first, phase, gaps)
        reward = sum(terms.values())
        t_next = t + 1
        vel_noise = 0.0
        if self.randomize:
            vel_noise = core.VEL_NOISE * jax.random.normal(jax.random.fold_in(info["rng"], t), (3,))
        actor_now = core.actor_obs(jnp, qpos, qvel, self.qadr, self.vadr, self.wadr, action,
                                   core.command_at(jnp, info["cmd_seq"], info["cmd_t"], t_next), phase, vel_noise)
        history = jnp.concatenate([state.obs["state"][core.ACTOR_OBS:core.HIST_SIZE], actor_now])
        obs = self._obs(history, actor_now, qpos, qvel, core.active_modes(jnp, info["mode"], info["fail_t"], t_next),
                        info["strength"], contact, air_time, scan, ground, normal)
        done = core.fallen(jnp, qpos, ground) | ~jnp.isfinite(obs["state"]).all() | ~jnp.isfinite(reward)
        yaw = core.heading(jnp, qpos[3:7])
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        cmd_disp = info["cmd_disp"] + core.CTRL_DT * jnp.stack([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1]])
        vel = core.body_velocity(jnp, qpos, qvel)
        wheel_working = (mode[core.N_JOINTS:] == core.HEALTHY) | (mode[core.N_JOINTS:] == core.WEAK)
        metrics = dict(state.metrics)
        metrics.update({f"r_{key}": jnp.nan_to_num(value) for key, value in terms.items()})
        metrics.update(
            vx=jnp.nan_to_num(vel[0]), vy=jnp.nan_to_num(vel[1]), wz=jnp.nan_to_num(vel[2]),
            lin_err=jnp.nan_to_num(jnp.linalg.norm(vel[:2] - cmd[:2])),
            yaw_err=jnp.nan_to_num(jnp.abs(vel[2] - cmd[2])),
            failed=jnp.sum(mode != core.HEALTHY).astype(jnp.float32),
            wheels_dead=jnp.all(~wheel_working).astype(jnp.float32),
            level_per_step=info["level"].astype(jnp.float32),
        )
        info = {**info, "t": t_next, "q_lock": q_lock, "last_action": action, "air_time": air_time,
                "last_contact": contact, "cmd_disp": cmd_disp, "phase": phase}
        return state.replace(pipeline_state=data, obs=obs, reward=jnp.nan_to_num(reward),
                             done=done.astype(jnp.float32), metrics=metrics, info=info)

    def _obs(self, history, actor_now, qpos, qvel, mode, strength, contact, air_time, scan=None, ground=None,
             normal=None):
        if scan is None:
            scan = core.scan_heights(jnp, self.terrain, qpos)
            ground = core.ground_level(jnp, scan)
            normal = core.ground_normal(jnp, self.terrain, qpos[0], qpos[1])
        scan_o = core.scan_obs(jnp, qpos, scan)
        return {"state": jnp.concatenate([history, scan_o]),
                "privileged_state": core.critic_obs(jnp, actor_now, scan_o, qpos, qvel, ground, normal, mode, strength,
                                                    contact, air_time)}

    def _sample_commands(self, rng, level):
        k_hold, k_cmd, k_stand = jax.random.split(rng, 3)
        holds = jax.random.randint(k_hold, (core.CMD_SEGMENTS,), *core.CMD_HOLD)
        cmd_t = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(holds)[:-1].astype(jnp.int32)])
        high = jnp.asarray(core.CMD_HIGH, jnp.float32).at[0].add(-core.SPEED_DROP * level)
        cmd = core.CMD_LOW + jax.random.uniform(k_cmd, (core.CMD_SEGMENTS, 3)) * (high - core.CMD_LOW)
        stand = jax.random.uniform(k_stand, (core.CMD_SEGMENTS,)) < core.P_STAND
        return jnp.where(stand[:, None], 0.0, cmd), cmd_t

    def _sample_failures(self, rng):
        k_kind, k_legs, k_motor, k_a, k_b, k_w = jax.random.split(rng, 6)
        kind = jnp.minimum(jnp.searchsorted(jnp.asarray(core.FAIL_CUM), jax.random.uniform(k_kind), side="right"),
                           len(core.FAIL_P) - 1)
        legs = jax.random.permutation(k_legs, core.N_LEGS)
        motor = jax.random.randint(k_motor, (), 0, core.N_ACT)
        leg_a = (kind == core.ONE_LEG) | (kind == core.TWO_LEGS) | (kind == core.WHEELS_AND_LEG)
        in_a = jnp.where(kind == core.ONE_MOTOR, jnp.arange(core.N_ACT) == motor, leg_a & (core.ACT_LEG == legs[0]))
        in_b = (kind == core.TWO_LEGS) & (core.ACT_LEG == legs[1])
        in_w = (kind >= core.ALL_WHEELS) & core.IS_WHEEL

        def when(key):
            k_now, k_time = jax.random.split(key)
            return jnp.where(jax.random.uniform(k_now) < 0.5, 0,
                             jax.random.randint(k_time, (), core.FAIL_EARLIEST, core.FAIL_LATEST))

        def group(key):
            k_mode, k_strength, k_when = jax.random.split(key, 3)
            v = jax.random.uniform(k_mode)
            mode = jnp.where(v < core.P_LIMP, core.LIMP, jnp.where(v < core.P_LIMP + core.P_LOCKED, core.LOCKED, core.WEAK))
            strength = jnp.where(mode == core.WEAK, jax.random.uniform(k_strength, minval=core.WEAK_RANGE[0],
                                                                      maxval=core.WEAK_RANGE[1]), 1.0)
            return mode, strength, when(k_when)

        (mode_a, strength_a, when_a), (mode_b, strength_b, when_b) = group(k_a), group(k_b)
        k_wmode, k_wwhen = jax.random.split(k_w)
        mode_w = jnp.where(jax.random.uniform(k_wmode) < core.P_WHEELS_LIMP, core.LIMP, core.LOCKED)
        when_w = when(k_wwhen)
        mode = jnp.where(in_w, mode_w, jnp.where(in_a, mode_a, jnp.where(in_b, mode_b, core.HEALTHY))).astype(jnp.int32)
        strength = jnp.where(in_w, 1.0, jnp.where(in_a, strength_a, jnp.where(in_b, strength_b, 1.0)))
        fail_t = jnp.where(in_w, when_w, jnp.where(in_a, when_a, jnp.where(in_b, when_b, core.NEVER))).astype(jnp.int32)
        return mode, strength, fail_t


class FullResetWrapper(Wrapper):
    def step(self, state, action):
        if "steps" in state.info:
            state.info["steps"] = jnp.where(state.done, jnp.zeros_like(state.info["steps"]), state.info["steps"])
        state = self.env.step(state.replace(done=jnp.zeros_like(state.done)), action)
        base = self.env.unwrapped
        info = dict(state.info)
        keys = jax.vmap(lambda key: jax.random.split(key, 3))(info["rng"])
        level = info["level"]
        if base.curriculum:
            done = state.done > 0
            fell = done & (info.get("truncation", jnp.zeros_like(state.done)) == 0)
            moved = jnp.linalg.norm(state.pipeline_state.qpos[:, :2] - info["spawn_xy"], axis=-1)
            asked = jnp.linalg.norm(info["cmd_disp"], axis=-1)
            up = done & ~fell & (moved > core.PROMOTE_DIST)
            down = done & ~up & (fell | (moved < core.DEMOTE_RATIO * asked))
            new = level + up.astype(jnp.int32) - down.astype(jnp.int32)
            anywhere = jax.vmap(lambda key: jax.random.randint(key, (), 0, core.LEVELS))(keys[:, 2])
            level = jnp.where(new >= core.LEVELS, anywhere, jnp.maximum(new, 0))
            histogram = jnp.mean(jax.nn.one_hot(level, core.LEVELS), axis=0)
            report = jax.random.uniform(jax.random.fold_in(keys[0, 2], 7)) < 1.0 / 64
            jax.lax.cond(report, lambda h: jax.debug.callback(_record_levels, h), lambda h: None, histogram)
        fresh = jax.vmap(base.reset_at)(keys[:, 1], level, info["column"])

        def pick(new, old):
            done = state.done.reshape(state.done.shape + (1,) * (old.ndim - state.done.ndim))
            return jnp.where(done > 0, new, old)

        for key in INFO_KEYS:
            info[key] = pick(fresh.info[key], info[key])
        info["rng"] = pick(fresh.info["rng"], keys[:, 0])
        return state.replace(
            pipeline_state=jax.tree_util.tree_map(pick, fresh.pipeline_state, state.pipeline_state),
            obs=jax.tree_util.tree_map(pick, fresh.obs, state.obs),
            info=info,
        )


def wrap_for_training(env, episode_length=core.EPISODE_STEPS, action_repeat=1, randomization_fn=None, **_):
    return FullResetWrapper(VmapWrapper(EpisodeWrapper(env, episode_length, action_repeat)))
