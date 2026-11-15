"""Drive the hexapod with the keyboard; kill motors while driving and switch terrains.

    python play.py             # with models/policy.npz
    python play.py --stand     # no policy, just stand
    python play.py --quality   # shadows and full-resolution terrain (needs a strong GPU)
    python play.py --panel     # show MuJoCo's own side panels
    python play.py --free-camera  # free camera instead of the chase camera

Drive (hold):  Up forward 0.4 m/s (walk)   Ctrl + Up 1.3 m/s (wheels + steps)   Down back   Left / Right turn
               Shift + Left / Right strafe
Faults:        1-6 kill that leg's motors (LF LM LR RF RM RR)   7 all wheels limp   8 all wheels locked
Terrain:       9 next terrain type   Page Up / Page Down level   Backspace start area (respawning repairs faults)
Panels:        P hide / show the on-screen panels
"""
import argparse
import ctypes
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from hexapod import core
from hexapod.env import HexapodEnv
from hexapod.policy import Policy

try:
    from hexapod.hud import Hud
except ImportError:
    Hud = None

KEY_BACKSPACE, KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_P = 259, 266, 267, ord("P")
VK_SHIFT, VK_CONTROL, VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN = 0x10, 0x11, 0x25, 0x26, 0x27, 0x28
HUD_EVERY = 3


def held(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def keyboard_command():
    if sys.platform != "win32":
        return np.zeros(3)
    forward = held(VK_UP) - held(VK_DOWN)
    side = held(VK_LEFT) - held(VK_RIGHT)
    vx = (1.3 if held(VK_CONTROL) else 0.4) * forward if forward > 0 else 0.4 * forward
    if held(VK_SHIFT):
        return np.array([vx, 0.2 * side, 0.0])
    return np.array([vx, 0.0, 1.0 * side])


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default=str(Path(__file__).resolve().parent / "models" / "policy.npz"))
    parser.add_argument("--stand", action="store_true", help="no policy, just stand")
    parser.add_argument("--quality", action="store_true", help="shadows and full-resolution terrain (slow on a weak GPU)")
    parser.add_argument("--panel", action="store_true", help="show MuJoCo's own side panels")
    parser.add_argument("--free-camera", action="store_true", help="free camera instead of the chase camera")
    args = parser.parse_args()

    policy = None if args.stand else Policy(args.policy)
    if policy is not None and policy.mean.shape[0] != core.OBS_SIZE:
        sys.exit(f"{args.policy} is from another stage (observation size {policy.mean.shape[0]}, this model expects "
                 f"{core.OBS_SIZE}). Use a matching policy.npz or run with --stand.")
    env = HexapodEnv(failures=False)
    cmd = np.zeros(3)
    env.set_command(cmd)
    course = {"level": None, "column": 0}

    def terrain_label():
        if course["level"] is None:
            return "start area"
        return f"{core.TERRAIN_TYPES[course['column']]}, level {course['level']}"

    def respawn():
        if course["level"] is None:
            return env.reset()
        print(f"terrain: {terrain_label()}")
        return env.reset(level=course["level"], column=course["column"], yaw=0.0)

    obs, _ = respawn()
    info = {"velocity": np.zeros(3), "contact": np.ones(core.N_LEGS, bool), "failed": 0}
    keys = []
    print(__doc__)
    hud = None if Hud is None else Hud()
    if hud is None:
        print("(the on-screen panels need Pillow: .venv\\Scripts\\pip install pillow)")
    if not args.quality:
        env.model.vis.quality.shadowsize = 0
        env.model.vis.quality.numslices, env.model.vis.quality.numstacks = 12, 6

    def heading_deg():
        return float(np.degrees(core.heading(np, env.data.qpos[3:7])))

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=keys.append,
                                      show_left_ui=args.panel, show_right_ui=args.panel) as viewer:
        if not args.quality:
            viewer.opt.geomgroup[2] = 0
            viewer.opt.geomgroup[4] = 1
        groups, flags = viewer.opt.geomgroup.copy(), viewer.opt.flags.copy()
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.model.body("torso").id
        viewer.cam.distance, viewer.cam.elevation = 1.5, -35.0
        viewer.cam.azimuth = heading_deg() if not args.free_camera else 145.0
        while viewer.is_running():
            start = time.perf_counter()
            target = keyboard_command()
            cmd += 0.15 * (target - cmd)
            env.set_command(cmd)
            toggle_hud = False
            with viewer.lock():
                if not args.panel:
                    viewer.opt.geomgroup[:] = groups
                    viewer.opt.flags[:] = flags
                while keys:
                    key = keys.pop(0)
                    if ord("1") <= key <= ord("6"):
                        env.fail_leg(key - ord("1"))
                        print(f"{core.LEGS[key - ord('1')]} leg: motors dead")
                    elif key in (ord("7"), ord("8")):
                        env.fail_wheels(core.LIMP if key == ord("7") else core.LOCKED)
                        print("all wheels " + ("limp (free-spinning)" if key == ord("7") else "locked (braked)"))
                    elif key == ord("9"):
                        if course["level"] is None:
                            course["level"] = 0
                        else:
                            course["column"] = (course["column"] + 1) % core.N_TYPES
                        obs, _ = respawn()
                    elif key in (KEY_PAGE_UP, KEY_PAGE_DOWN):
                        step = 1 if key == KEY_PAGE_UP else -1
                        course["level"] = int(np.clip((course["level"] or 0) + step, 0, core.LEVELS - 1))
                        obs, _ = respawn()
                    elif key == KEY_BACKSPACE:
                        course["level"] = None
                        obs, _ = respawn()
                        print("start area")
                    elif key == KEY_P:
                        toggle_hud = True
                action = np.zeros(core.N_ACT) if policy is None else policy.act(obs)
                obs, _, terminated, _, info = env.step(action)
                if terminated:
                    print("fell over, respawning")
                    obs, _ = respawn()
                elif env.t % 50 == 0:
                    v = info["velocity"]
                    print(f"cmd ({cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f})  "
                          f"vel ({v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f})  failed motors {info['failed']}")
                if not args.free_camera:
                    turn = (heading_deg() - viewer.cam.azimuth + 180.0) % 360.0 - 180.0
                    viewer.cam.azimuth += 0.15 * turn
            if hud is not None:
                if toggle_hud:
                    hud.toggle(viewer)
                if toggle_hud or env.t % HUD_EVERY == 0:
                    hud.update(viewer, env, info, cmd, target, terrain_label())
            viewer.sync()
            time.sleep(max(0.0, core.CTRL_DT - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
