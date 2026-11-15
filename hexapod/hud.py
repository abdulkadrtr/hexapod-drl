"""On-screen overlay for play.py: robot schematic, status and controls."""
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import core

SS = 2
FONT_DIR = Path(r"C:\Windows\Fonts")
BG, TEXT = (21, 25, 32), (228, 232, 238)
GOOD, BAD, WARN = (63, 185, 80), (248, 81, 73), (222, 162, 42)
BODY, BODY_EDGE, ARROW = (76, 85, 100), (118, 128, 145), (178, 186, 198)
STATE_COLOR = {core.HEALTHY: GOOD, core.LIMP: BAD, core.LOCKED: BAD, core.WEAK: WARN}
FAULT = {core.LIMP: "limp", core.LOCKED: "locked", core.WEAK: "weak"}
BODY_W, BODY_H = 19, 46
LEG_Y = {"F": -28, "M": 0, "R": 28}
FOOT = {"F": (50, -52), "M": (56, 0), "R": (50, 52)}
PANEL_W, PANEL_H, CX, CY = 140, 150, 70, 76
CONTROLS = (
    ("Arrows", "drive / turn"),
    ("Ctrl / Shift", "fast / strafe"),
    ("1-6  7  8", "kill leg / wheels limp / locked"),
    ("9  PgUp  PgDn", "terrain / level"),
    ("Backspace  P", "respawn / hide overlay"),
)


@lru_cache(maxsize=None)
def _font(px, bold):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        if (FONT_DIR / name).exists():
            return ImageFont.truetype(str(FONT_DIR / name), px)
    return ImageFont.load_default(px)


class Canvas:
    def __init__(self, w, h, scale, base=None):
        self.out = (round(w * scale), round(h * scale))
        self.k = scale * SS
        self.img = Image.new("RGB", (self.out[0] * SS, self.out[1] * SS), BG) if base is None else base.copy()
        self.draw = ImageDraw.Draw(self.img)

    def text(self, x, y, s, size, color, bold=False, anchor="ls"):
        self.draw.text((x * self.k, y * self.k), s, font=_font(max(6, round(size * self.k)), bold), fill=color,
                       anchor=anchor)

    def rrect(self, x0, y0, x1, y1, r, fill=None, outline=None, width=1.0):
        self.draw.rounded_rectangle([x0 * self.k, y0 * self.k, x1 * self.k, y1 * self.k], radius=r * self.k,
                                    fill=fill, outline=outline, width=max(1, round(width * self.k)))

    def line(self, points, color, width):
        self.draw.line([c * self.k for p in points for c in p], fill=color, width=max(1, round(width * self.k)),
                       joint="curve")

    def circle(self, x, y, r, fill=None, outline=None, width=1.0):
        self.draw.ellipse([(x - r) * self.k, (y - r) * self.k, (x + r) * self.k, (y + r) * self.k],
                          fill=fill, outline=outline, width=max(1, round(width * self.k)))

    def polygon(self, points, fill):
        self.draw.polygon([c * self.k for p in points for c in p], fill=fill)

    def image(self):
        return np.asarray(self.img.reduce(SS))


def leg_states(env):
    mode = core.active_modes(np, env.mode, env.fail_t, env.t)

    def worst(m):
        for kind in (core.LIMP, core.LOCKED, core.WEAK):
            if np.any(m == kind):
                return kind
        return core.HEALTHY

    joints = [worst(mode[(core.ACT_LEG == i) & ~core.IS_WHEEL]) for i in range(core.N_LEGS)]
    wheels = [worst(mode[(core.ACT_LEG == i) & core.IS_WHEEL]) if core.HAS_WHEEL[i] else None
              for i in range(core.N_LEGS)]
    return joints, wheels


def fault_text(joints, wheels):
    parts = [f"{leg} leg {FAULT[s]}" for leg, s in zip(core.LEGS, joints) if s != core.HEALTHY]
    wheel_legs = [i for i in range(core.N_LEGS) if wheels[i] is not None]
    broken = [i for i in wheel_legs if wheels[i] != core.HEALTHY]
    if broken and len(broken) == len(wheel_legs):
        kind = ([wheels[i] for i in broken if joints[i] == core.HEALTHY] or [wheels[broken[0]]])[0]
        parts.append(f"wheels {FAULT[kind]}")
    else:
        parts += [f"{core.LEGS[i]} wheel {FAULT[wheels[i]]}" for i in broken if joints[i] == core.HEALTHY]
    return ", ".join(parts) or "none"


def _leg_geometry(leg):
    side = -1 if leg[0] == "L" else 1
    fx, fy = FOOT[leg[1]]
    ay = LEG_Y[leg[1]]
    return ((CX + side * BODY_W, CY + ay), (CX + side * (BODY_W + 15), CY + ay + 0.3 * (fy - ay)),
            (CX + side * fx, CY + fy))


@lru_cache(maxsize=8)
def _base(scale):
    c = Canvas(PANEL_W, PANEL_H, scale)
    c.rrect(CX - BODY_W, CY - BODY_H, CX + BODY_W, CY + BODY_H, 10, fill=BODY, outline=BODY_EDGE, width=1.5)
    c.polygon([(CX, CY - BODY_H + 4), (CX - 5, CY - BODY_H + 11), (CX + 5, CY - BODY_H + 11)], ARROW)
    for leg in core.LEGS:
        c.text(CX + (-9.5 if leg[0] == "L" else 9.5), CY + LEG_Y[leg[1]], leg, 7.5, TEXT, bold=True, anchor="mm")
    return c.img


def schematic(scale, env, info):
    joints, wheels = leg_states(env)
    c = Canvas(PANEL_W, PANEL_H, scale, base=_base(scale))
    for i, leg in enumerate(core.LEGS):
        anchor, knee, (x, y) = _leg_geometry(leg)
        c.line([anchor, knee, (x, y)], STATE_COLOR[joints[i]], 5)
        on_ground = info["contact"][i]
        if wheels[i] is None:
            color = STATE_COLOR[joints[i]]
            c.circle(x, y, 6.5, fill=color if on_ground else BG, outline=color, width=2)
        else:
            color = STATE_COLOR[wheels[i]]
            c.rrect(x - 5.5, y - 11, x + 5.5, y + 11, 3, fill=color if on_ground else BG, outline=color, width=2)
    return c.image()


def status(env, speed, cmd, terrain):
    shown = "parked" if np.sum(np.abs(cmd)) < 0.05 else f"{speed:.2f} m/s"
    return "speed\nterrain\nfaults", f"{shown}\n{terrain}\n{fault_text(*leg_states(env))}"


class Hud:
    def __init__(self):
        self.visible = True
        self.speed = 0.0

    def toggle(self, viewer):
        self.visible = not self.visible
        if not self.visible:
            viewer.clear_images()
            viewer.clear_texts()

    def update(self, viewer, env, info, cmd, target, terrain):
        if not self.visible:
            return
        vp = viewer.viewport
        if vp is None or vp.width < 320 or vp.height < 240:
            viewer.clear_images()
            return
        scale = round(float(np.clip(vp.height / 700, 0.8, 1.5)) * 20) / 20
        img = schematic(scale, env, info)
        margin = 8
        rect = mujoco.MjrRect(vp.left + margin, vp.bottom + vp.height - margin - img.shape[0], img.shape[1],
                              img.shape[0])
        font = mujoco.mjtFont.mjFONT_NORMAL
        self.speed += 0.25 * (float(np.hypot(*info["velocity"][:2])) - self.speed)
        labels, values = status(env, self.speed, cmd, terrain)
        viewer.set_texts([(font, mujoco.mjtGridPos.mjGRID_TOPRIGHT, labels, values),
                          (font, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "\n".join(k for k, _ in CONTROLS),
                           "\n".join(a for _, a in CONTROLS))])
        viewer.set_images([(rect, img)])
