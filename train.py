"""Trains the policy with Brax PPO on MJX.

    python train.py --steps 300_000_000 --video
"""
import argparse
import functools
import inspect
import sys
import time
from pathlib import Path

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from hexapod import core, mjx_env
from hexapod.env import HexapodEnv
from hexapod.mjx_env import HexapodMJX, wrap_for_training
from hexapod.policy import Policy

HIDDEN = (512, 256, 128)
PPO_CONFIG = dict(
    num_envs=4096, num_eval_envs=128, batch_size=256, num_minibatches=32, unroll_length=20,
    num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97,
    gae_lambda=0.95, clipping_epsilon=0.2, max_grad_norm=1.0, normalize_observations=True,
    reward_scaling=1.0, episode_length=core.EPISODE_STEPS,
)
SMOKE_CONFIG = dict(
    num_timesteps=4_000, num_evals=2, num_envs=8, num_eval_envs=8, batch_size=8,
    num_minibatches=4, unroll_length=10, num_updates_per_batch=1, episode_length=60,
)
FLAT = ((-3.0, 0.0), np.pi)
SCENARIOS = (
    ("flat, forward 0.6", (0.6, 0.0, 0.0), None, FLAT),
    ("flat, forward 0.3 (middle legs in rhythm)", (0.3, 0.0, 0.0), None, FLAT),
    ("flat, forward 1.2 (wheels + steps)", (1.2, 0.0, 0.0), None, FLAT),
    ("flat, turn in place 1.0 rad/s", (0.0, 0.0, 1.0), None, FLAT),
    ("flat, forward 0.3 + turn 1.0", (0.3, 0.0, 1.0), None, FLAT),
    ("flat, sideways 0.15", (0.0, 0.15, 0.0), None, FLAT),
    ("flat, wheels locked, 0.4", (0.4, 0.0, 0.0), ("wheels", core.LOCKED), FLAT),
    ("flat, wheels limp, 0.4", (0.4, 0.0, 0.0), ("wheels", core.LIMP), FLAT),
    ("flat, LF leg limp, 0.6", (0.6, 0.0, 0.0), ("leg", 0, core.LIMP), FLAT),
    ("ramp 6 from the edge (climb), 0.5", (0.5, 0.0, 0.0), None, ("ramp", 6, "edge")),
    ("rubble 5, 0.5", (0.5, 0.0, 0.0), None, ("rubble", 5)),
    ("blocks 5, 0.5", (0.5, 0.0, 0.0), None, ("blocks", 5)),
    ("stairs 5 from the edge (climb), 0.4", (0.4, 0.0, 0.0), None, ("stairs", 5, "edge")),
    ("stairs 5 from the top (descent), 0.4", (0.4, 0.0, 0.0), None, ("stairs", 5)),
    ("trenches 5, 0.4", (0.4, 0.0, 0.0), None, ("trenches", 5)),
    ("stairs 3 from the edge, wheels locked, 0.3", (0.3, 0.0, 0.0), ("wheels", core.LOCKED), ("stairs", 3, "edge")),
)


def export_policy(params, path, quiet=False):
    normalizer, policy_params = params[0], params[1]
    mean, std = normalizer.mean, normalizer.std
    if isinstance(mean, dict):
        mean, std = mean["state"], std["state"]
    layers = policy_params["params"] if "params" in policy_params else policy_params
    names = sorted(layers, key=lambda name: int(name.rsplit("_", 1)[-1]))
    arrays = {"obs_mean": mean, "obs_std": std}
    for i, name in enumerate(names):
        arrays[f"w{i}"], arrays[f"b{i}"] = layers[name]["kernel"], layers[name]["bias"]
    arrays = {key: np.asarray(value, np.float32) for key, value in arrays.items()}
    np.savez(path, act_size=core.N_ACT, n_layers=len(names), **arrays)
    if not quiet:
        print(f"wrote {path}; layers {[arrays[f'w{i}'].shape for i in range(len(names))]}")


CHECKPOINT_CONFIG = "ppo_network_config.json"


def latest_checkpoint(root):
    root = Path(root)
    found = [p.parent for p in root.rglob(CHECKPOINT_CONFIG) if p.parent.name.isdigit() and len(p.parent.name) == 12]
    if (root / CHECKPOINT_CONFIG).exists():
        found.append(root)
    if not found:
        sys.exit(f"no checkpoint under {root} (looking for checkpoints/<12-digit step>/{CHECKPOINT_CONFIG})")
    return max(found, key=lambda p: int(p.name))


def spawn(env, place, seed):
    if place == FLAT or place is None:
        (x, y), yaw = FLAT
        return env.reset(seed=seed, pos=(x, y), yaw=yaw)
    kind, level, *where = place
    if where:
        cx, cy = core.tile_center(level, core.TERRAIN_TYPES.index(kind))
        return env.reset(seed=seed, pos=(cx - core.EDGE_OFFSET, cy), yaw=0.0)
    return env.reset(seed=seed, level=level, column=kind, yaw=0.0)


def apply_failure(env, failure):
    kind, *rest = failure
    if kind == "wheels":
        env.fail_wheels(rest[0])
    elif kind == "leg":
        env.fail_leg(rest[0], rest[1])
    else:
        env.fail_actuators([rest[0]], rest[1])


def evaluate(path, episodes=3, seconds=10.0, fail_at=3.0):
    policy = Policy(path)
    steps, t_fail, window = int(seconds / core.CTRL_DT), int(fail_at / core.CTRL_DT), int(3.0 / core.CTRL_DT)
    print(f"Sim2sim (CPU MuJoCo), {seconds:.0f} s, failures at {fail_at:.0f} s; velocity: mean of the last 3 s, "
          f"progress: distance along the initial heading:")
    print(f"  {'scenario':44s}   vx     vy     ωz   (command)             progress  falls")
    for label, cmd, failure, place in SCENARIOS:
        env = HexapodEnv(failures=False)
        env.set_command(cmd)
        means, progress, falls = [], [], 0
        for episode in range(episodes):
            obs, _ = spawn(env, place, episode)
            start, yaw = env.data.qpos[:2].copy(), core.heading(np, env.data.qpos[3:7])
            trace = []
            for t in range(steps):
                if failure and t == t_fail:
                    apply_failure(env, failure)
                obs, _, terminated, _, info = env.step(policy.act(obs))
                trace.append(info["velocity"])
                if terminated:
                    falls += 1
                    break
            means.append(np.mean(trace[-window:], axis=0))
            progress.append(np.dot(env.data.qpos[:2] - start, (np.cos(yaw), np.sin(yaw))))
        v = np.mean(means, axis=0)
        print(f"  {label:44s} {v[0]:+.2f}  {v[1]:+.2f}  {v[2]:+.2f}   ({cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f})"
              f"   {np.mean(progress):5.1f} m    {falls}/{episodes}")


def record_video(path, out="hexapod.mp4"):
    import mediapy

    policy = Policy(path)
    segments = (
        ("flat ground", FLAT, ((0.0, (0.3, 0.0, 0.0)), (3.0, (1.2, 0.0, 0.0)), (6.0, (0.0, 0.0, 1.0)),
                             (8.5, (0.0, 0.0, 0.0))), None, 10.0),
        ("stairs 4, from the edge", ("stairs", 4, "edge"), ((0.0, (0.4, 0.0, 0.0)),), None, 8.0),
        ("trenches 4, wheels locked", ("trenches", 4), ((0.0, (0.4, 0.0, 0.0)),), ("wheels", core.LOCKED), 8.0),
    )
    frames = []
    for label, place, program, failure, seconds in segments:
        env = HexapodEnv(failures=False)
        env.set_command(program[0][1])
        obs, _ = spawn(env, place, 0)
        for t in range(int(seconds / core.CTRL_DT)):
            for at, cmd in program:
                if t == int(at / core.CTRL_DT):
                    env.set_command(cmd)
            if failure and t == int(2.0 / core.CTRL_DT):
                apply_failure(env, failure)
            obs, _, terminated, _, _ = env.step(policy.act(obs))
            if t % 2 == 0:
                frames.append(env.render(distance=1.0, elevation=-20.0))
            if terminated:
                print(f"  video: fell during '{label}' ({t * core.CTRL_DT:.1f} s)")
                break
        env.close()
    mediapy.write_video(out, frames, fps=round(0.5 / core.CTRL_DT))
    print(f"wrote {out} ({len(frames)} frames): " + " | ".join(s[0] for s in segments))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=300_000_000)
    parser.add_argument("--evals", type=int, default=10)
    parser.add_argument("--envs", type=int, default=PPO_CONFIG["num_envs"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="policy.npz")
    parser.add_argument("--checkpoints", default="checkpoints",
                        help="folder for full training checkpoints (empty: do not save)")
    parser.add_argument("--resume", default=None,
                        help="checkpoints folder of a previous run (or a folder containing it); resumes from the latest step")
    parser.add_argument("--video", action="store_true", help="record hexapod.mp4 after training")
    parser.add_argument("--smoke", action="store_true", help="few-minute end-to-end test on CPU")
    args = parser.parse_args()

    print("JAX devices:", jax.devices(), flush=True)
    cfg = {**PPO_CONFIG, "num_timesteps": args.steps, "num_evals": args.evals, "num_envs": args.envs, "seed": args.seed}
    if args.smoke:
        cfg.update(SMOKE_CONFIG)
    cfg["policy_params_fn"] = lambda step, make_policy, params: export_policy(params, args.out, quiet=True)
    if args.checkpoints:
        cfg["save_checkpoint_path"] = str(Path(args.checkpoints).resolve())
    if args.resume:
        cfg["restore_checkpoint_path"] = str(latest_checkpoint(args.resume))
        print(f"Resuming from {cfg['restore_checkpoint_path']} (policy, critic and normalizer)", flush=True)

    accepted = inspect.signature(ppo.train).parameters
    env = HexapodMJX(curriculum=True, init_level_max=core.LEVELS - 1 if args.resume else core.INIT_LEVEL_MAX)
    eval_env = HexapodMJX(curriculum=False)
    if "wrap_env_fn" in accepted:
        cfg["wrap_env_fn"] = wrap_for_training
    else:
        env = wrap_for_training(env, cfg["episode_length"])
        eval_env = wrap_for_training(eval_env, cfg["episode_length"])
        cfg["wrap_env"] = False
    skipped = sorted(set(cfg) - set(accepted))
    if skipped:
        print("Settings not supported by this Brax version, skipped:", skipped)

    network = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=HIDDEN,
                                value_hidden_layer_sizes=HIDDEN, policy_obs_key="state",
                                value_obs_key="privileged_state")
    start = time.time()

    def progress(step, metrics):
        length = max(float(metrics.get("eval/avg_episode_length", 1.0)), 1.0)
        mean = lambda key: float(metrics.get(f"eval/episode_{key}", float("nan"))) / length
        levels = mjx_env.CURRICULUM["levels"]
        print(f"{step:>13,} steps | reward {float(metrics.get('eval/episode_reward', float('nan'))):7.1f} "
              f"| velocity error {mean('lin_err'):.3f} m/s | yaw error {mean('yaw_err'):.2f} rad/s "
              f"| episode {length:4.0f} | curriculum level mean {np.dot(levels, np.arange(core.LEVELS)):.2f}, "
              f"top {100 * levels[-1]:.0f}% | {time.time() - start:5.0f} s", flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=env, eval_env=eval_env, network_factory=network, progress_fn=progress,
        **{key: value for key, value in cfg.items() if key in accepted})
    print(f"Training finished: {time.time() - start:.0f} s", flush=True)

    export_policy(params, args.out)
    inference = jax.jit(make_inference_fn(params, deterministic=True))
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    obs = {"state": jax.random.normal(k1, (core.OBS_SIZE,)), "privileged_state": jax.random.normal(k2, (core.CRITIC_OBS,))}
    reference = np.asarray(inference(obs, jax.random.PRNGKey(0))[0])
    gap = float(np.abs(reference - Policy(args.out).act(np.asarray(obs["state"]))).max())
    print(f"Brax vs numpy policy difference: {gap:.1e}" + ("" if gap < 1e-3 else "   <-- WARNING: export is wrong"))

    evaluate(args.out, episodes=1 if args.smoke else 3, seconds=2.0 if args.smoke else 10.0,
             fail_at=1.0 if args.smoke else 3.0)
    if args.video:
        try:
            record_video(args.out)
        except Exception as error:
            print(f"Could not record the video ({type(error).__name__}: {error})")


if __name__ == "__main__":
    main()
