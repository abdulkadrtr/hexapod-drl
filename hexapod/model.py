"""Generates hexapod.xml and the terrain (terrain.bin, terrain_lite.bin, terrain_tex.png).

    python -m hexapod.model
"""
import hashlib
import math
import struct
import zlib
from pathlib import Path

import numpy as np

from .core import BORDER, LEVELS, START_X, TERRAIN_TYPES, TILE

ASSETS = Path(__file__).resolve().parent.parent / "assets"
OUT = ASSETS / "hexapod.xml"
TERRAIN_FILE = ASSETS / "terrain.bin"
TERRAIN_LITE_FILE = ASSETS / "terrain_lite.bin"
TEXTURE_FILE = ASSETS / "terrain_tex.png"
RES = TILE / 66
MARGIN = 0.15
WHEEL_RADIUS = 0.03
WHEEL_DROP = 0.102
WHEEL_OFFSET = 0.022

LEGS = (
    ("LF", (0.12, 0.08), 0.7854, True),
    ("LM", (0.0, 0.09), 1.5708, False),
    ("LR", (-0.12, 0.08), 2.3562, True),
    ("RF", (0.12, -0.08), -0.7854, True),
    ("RM", (0.0, -0.09), -1.5708, False),
    ("RR", (-0.12, -0.08), -2.3562, True),
)

LOGO_PIECES = (
    ((-0.5000, 0.4146), (-0.1208, 0.3104), (-0.1167, 0.3063), (0.0208, -0.0771)),
    ((0.3625, 0.2979), (0.3625, 0.3146), (0.4958, 0.2771), (0.5000, 0.2729), (0.4042, 0.1854), (0.3958, 0.1812),
     (0.3917, 0.1812)),
    ((0.1833, 0.1521), (0.3292, 0.3021), (0.4000, 0.0104), (0.4000, 0.0063)),
    ((0.3833, -0.0229), (0.3833, -0.0271), (0.0792, -0.1146), (0.0708, -0.1146), (0.0667, -0.1104), (-0.0708, 0.2812)),
    ((0.0750, -0.1521), (0.0792, -0.1479), (0.3667, -0.0646), (0.3792, -0.0646), (0.0833, -0.4146)),
    ((0.0458, -0.1771), (0.0458, -0.3521), (-0.2083, -0.3563), (0.0375, -0.1729), (0.0417, -0.1729)),
)
LOGO_CENTER, LOGO_WIDTH, LOGO_Z, LOGO_INK = (-0.06, 0.0), 0.062, 0.044, 0.0003


def logo_xml():
    meshes, geoms = [], []
    for i, piece in enumerate(LOGO_PIECES):
        verts = []
        for px, py in piece:
            x, y = LOGO_CENTER[0] + py * LOGO_WIDTH, LOGO_CENTER[1] - px * LOGO_WIDTH
            verts += [f"{x:.5f} {y:.5f} {LOGO_Z}", f"{x:.5f} {y:.5f} {LOGO_Z + LOGO_INK:.4f}"]
        meshes.append(f'    <mesh name="logo{i}" vertex="{"  ".join(verts)}"/>')
        geoms.append(f'      <geom class="visual" type="mesh" mesh="logo{i}" material="logo_ink"/>')
    return "\n".join(meshes), "\n".join(geoms)

COXA_STATOR = ('      <geom class="visual" type="box" size="0.015 0.0115 0.0105" pos="{x:.4f} {y:.4f} 0" '
               'euler="0 0 {yaw}" material="servo"/>')

LEG_TOP = """      <body name="{n}_coxa" pos="{x} {y} 0" euler="0 0 {yaw}">
        <joint name="{n}_coxa" class="coxa"/>
        <geom class="coxa" fromto="0 0 0 0.04 0 0"/>
        <geom class="visual" type="cylinder" size="0.0085 0.0012" pos="0 0 0.0117" material="silver"/>
        <geom class="visual" type="cylinder" size="0.0085 0.0012" pos="0 0 -0.0117" material="silver"/>
        <geom class="visual" type="box" size="0.032 0.0125 0.0018" pos="0.028 0 0.0147" material="bracket"/>
        <geom class="visual" type="box" size="0.032 0.0125 0.0018" pos="0.028 0 -0.0147" material="bracket"/>
        <geom class="visual" type="cylinder" size="0.0016 0.0005" pos="0.048 0.006 0.017" material="silver"/>
        <geom class="visual" type="cylinder" size="0.0016 0.0005" pos="0.048 -0.006 0.017" material="silver"/>
        <geom class="visual" type="box" size="0.016 0.0115 0.0105" pos="0.05 0 0" material="servo"/>
        <body name="{n}_femur" pos="0.04 0 0">
          <joint name="{n}_femur" class="femur"/>
          <geom class="femur" fromto="0 0 0 0.08 0 0"/>
          <geom class="visual" type="cylinder" fromto="0 0.0115 0 0 0.0139 0" size="0.0085" material="silver"/>
          <geom class="visual" type="cylinder" fromto="0 -0.0115 0 0 -0.0139 0" size="0.0085" material="silver"/>
          <geom class="visual" type="box" size="0.04 0.0012 0.0095" pos="0.04 0.0151 0" material="bracket"/>
          <geom class="visual" type="box" size="0.04 0.0012 0.0095" pos="0.04 -0.0151 0" material="bracket"/>
          <geom class="visual" type="cylinder" fromto="0 0.0139 0 0 0.0163 0" size="0.0095" material="bracket"/>
          <geom class="visual" type="cylinder" fromto="0 -0.0139 0 0 -0.0163 0" size="0.0095" material="bracket"/>
          <geom class="visual" type="cylinder" fromto="0.08 0.0139 0 0.08 0.0163 0" size="0.0095" material="bracket"/>
          <geom class="visual" type="cylinder" fromto="0.08 -0.0139 0 0.08 -0.0163 0" size="0.0095" material="bracket"/>
          <geom class="visual" type="box" size="0.016 0.0115 0.0105" pos="0.07 0 0" material="servo"/>
          <geom class="visual" type="capsule" fromto="0.064 0.004 0.0112 0.028 0.004 0.0112" size="0.0015" material="cable"/>
          <geom class="visual" type="capsule" fromto="0.028 0.004 0.0112 0.012 0.006 0.0135" size="0.0015" material="cable"/>
          <body name="{n}_tibia" pos="0.08 0 0">
            <joint name="{n}_tibia" class="tibia"/>
            <geom class="tibia" fromto="0 0 0 0 0 {tibia_end}"/>
            <geom class="visual" type="cylinder" fromto="0 0.0115 0 0 0.0139 0" size="0.0085" material="silver"/>
            <geom class="visual" type="cylinder" fromto="0 -0.0115 0 0 -0.0139 0" size="0.0085" material="silver"/>
            <geom class="visual" type="box" size="0.009 0.0012 0.021" pos="0 0.0177 -0.021" material="bracket"/>
            <geom class="visual" type="box" size="0.009 0.0012 0.021" pos="0 -0.0177 -0.021" material="bracket"/>
            <geom class="visual" type="cylinder" fromto="0 0.0165 0 0 0.0189 0" size="0.009" material="bracket"/>
            <geom class="visual" type="cylinder" fromto="0 -0.0165 0 0 -0.0189 0" size="0.009" material="bracket"/>
            <geom class="visual" type="box" size="0.009 0.0189 0.0045" pos="0 0 -0.0445" material="bracket"/>
            <geom class="visual" type="cylinder" size="0.0016 0.0005" pos="0.0092 0.01 -0.0445" euler="0 1.5708 0" material="silver"/>
            <geom class="visual" type="cylinder" size="0.0016 0.0005" pos="0.0092 -0.01 -0.0445" euler="0 1.5708 0" material="silver"/>
"""

FOOT_END = """            <geom class="visual" type="capsule" fromto="0 0 -0.049 0 0 -0.108" size="0.0058" material="carbon"/>
            <geom class="visual" type="cylinder" size="0.0082 0.003" pos="0 0 -0.106" material="rubber"/>
            <geom name="{n}_foot" class="foot" pos="0 0 -0.12"/>
          </body>
        </body>
      </body>"""

WHEEL_END = """            <geom class="visual" type="capsule" fromto="0 0 -0.049 0 0 -0.094" size="0.0058" material="carbon"/>
            <geom class="visual" type="box" size="0.009 0.009 0.009" pos="0 0 -{drop}" material="bracket"/>
            <geom class="visual" type="cylinder" fromto="0 0 -{drop} {stub}" size="0.0045" material="silver"/>
            <body name="{n}_wheel" pos="{hub} -{drop}">
              <joint name="{n}_wheel" class="wheel" axis="{axis} 0"/>
              <geom name="{n}_tire" class="tire"/>
              <geom class="visual" type="cylinder" fromto="{tread}" size="{radius}" material="rubber"/>
              <geom class="visual" type="cylinder" fromto="{rim}" size="0.0195" material="bracket"/>
              <geom class="visual" type="cylinder" fromto="{face}" size="0.014" material="servo"/>
{bolts}
            </body>
          </body>
        </body>
      </body>"""


def blur(a, cells):
    x = np.arange(-3 * cells, 3 * cells + 1)
    k = np.exp(-0.5 * (x / cells) ** 2)
    k /= k.sum()
    for axis in (0, 1):
        a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), axis, a)
    return a


def tile_heights(kind, d, rng, n):
    u = (np.arange(n + 1) * RES)[None, :]
    v = (np.arange(n + 1) * RES)[:, None]
    shape = (n + 1, n + 1)
    cheb = np.maximum(np.abs(u - TILE / 2), np.abs(v - TILE / 2))
    plateau = np.clip(0.8 - cheb, 0.0, 0.6)

    def bumps(amp, scale):
        b = blur(rng.standard_normal(shape), scale / RES)
        return amp * (b - b.min()) / (b.max() - b.min())

    def stones(amp, size):
        k = max(1, round(size / RES))
        m = -(-(n + 1) // k)
        return amp * np.kron(rng.uniform(0.0, 1.0, (m, m)), np.ones((k, k)))[: n + 1, : n + 1]

    def blocks(count, size, height, clear=0.3):
        h = np.zeros(shape)
        placed = 0
        while placed < count:
            cx, cy = rng.uniform(0.2, TILE - 0.2, 2)
            if max(abs(cx - TILE / 2), abs(cy - TILE / 2)) < clear:
                continue
            sx, sy = rng.uniform(size[0], size[1], 2) / 2
            top = rng.uniform(0.5, 1.0) * height
            h = np.where((np.abs(u - cx) < sx) & (np.abs(v - cy) < sy), np.maximum(h, top), h)
            placed += 1
        return h

    if kind == "gravel":
        h = stones(0.003 + 0.017 * d, 0.04)
    elif kind == "ramp":
        h = math.tan(math.radians(4 + 22 * d)) * plateau
    elif kind == "bumps":
        h = bumps(0.01 + 0.07 * d, 0.12)
    elif kind == "rubble":
        h = stones(0.005 + 0.035 * d, 0.08) + stones(0.003 + 0.007 * d, 0.04)
    elif kind == "blocks":
        h = blocks(int(10 + 16 * d), (0.1, 0.3), 0.015 + 0.065 * d)
    elif kind == "stairs":
        h = np.floor(plateau / 0.2 + 1e-9) * (0.015 + 0.055 * d)
    elif kind == "trenches":
        width = 0.03 + 0.09 * d
        gap = (np.abs(cheb - 0.35) < width / 2) | (np.abs(cheb - 0.65) < width / 2)
        h = (0.04 + 0.03 * d) * ~gap
    else:
        h = (math.tan(math.radians(3 + 12 * d)) * plateau + stones(0.002 + 0.018 * d, 0.06)
             + blocks(int(4 + 6 * d), (0.1, 0.2), 0.01 + 0.04 * d))
    w = np.clip(np.minimum(np.minimum(u, TILE - u), np.minimum(v, TILE - v)) / MARGIN, 0.0, 1.0)
    return h * w * w * (3 - 2 * w)


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    chunk = lambda tag, data: (struct.pack(">I", len(data)) + tag + data
                               + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def terrain_texture(heights, up=2):
    h = np.kron(heights, np.ones((up, up)))
    px = RES / up
    gy, gx = np.gradient(h, px)
    normal = np.stack([-gx, -gy, np.ones_like(h)], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([-0.6, 0.5, 0.75])
    shade = np.clip(normal @ (light / np.linalg.norm(light)), 0.0, 1.0)
    tint = 0.9 + 0.35 * np.clip(h / max(heights.max(), 1e-6), 0.0, 1.0)
    rgb = np.array([0.62, 0.59, 0.54]) * ((0.25 + 0.85 * shade) * tint)[..., None]
    rows, cols = np.indices(h.shape)
    x, y = cols * px - BORDER, rows * px - BORDER
    fine = ((x + 1e-9) % 0.5 < px) | ((y + 1e-9) % 0.5 < px)
    tile = ((x + 1e-9) % TILE < 1.5 * px) | ((y + 1e-9) % TILE < 1.5 * px)
    rgb = np.where(fine[..., None], rgb * 0.82, rgb)
    rgb = np.where(tile[..., None], rgb * 0.65, rgb)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def make_terrain(seed=0):
    rng = np.random.default_rng(seed)
    n, b = round(TILE / RES), round(BORDER / RES)
    heights = np.zeros((len(TERRAIN_TYPES) * n + 1 + 2 * b, LEVELS * n + 1 + 2 * b))
    for j, kind in enumerate(TERRAIN_TYPES):
        for i in range(LEVELS):
            rows, cols = slice(b + j * n, b + (j + 1) * n + 1), slice(b + i * n, b + (i + 1) * n + 1)
            heights[rows, cols] = np.maximum(heights[rows, cols], tile_heights(kind, i / (LEVELS - 1), rng, n))
    for path, data in ((TERRAIN_FILE, heights), (TERRAIN_LITE_FILE, heights[::2, ::2])):
        with open(path, "wb") as f:
            np.array(data.shape, np.int32).tofile(f)
            data.astype(np.float32).tofile(f)
    return heights


def vec(*parts):
    return " ".join(f"{v:.5f}" for p in parts for v in p)


def wheel_end(name, yaw):
    axle = (math.sin(yaw), math.cos(yaw), 0.0)
    ahead = (math.cos(yaw), -math.sin(yaw), 0.0)
    side = 1.0 if name[0] == "L" else -1.0
    along = lambda d: tuple(side * d * a for a in axle)
    span = lambda d: vec(tuple(-d * a for a in axle), tuple(d * a for a in axle))
    bolts = []
    for k in range(5):
        th = 2 * math.pi * k / 5
        off = tuple(0.0095 * (math.cos(th) * h + math.sin(th) * z) for h, z in zip(ahead, (0.0, 0.0, 1.0)))
        start = tuple(o + a for o, a in zip(off, along(0.0136)))
        end = tuple(o + a for o, a in zip(off, along(0.0143)))
        bolts.append(f'              <geom class="visual" type="cylinder" fromto="{vec(start, end)}" size="0.0016" material="silver"/>')
    hub = along(WHEEL_OFFSET)
    return WHEEL_END.format(
        n=name, drop=WHEEL_DROP, radius=WHEEL_RADIUS,
        stub=vec(tuple(c for c in along(WHEEL_OFFSET - 0.011)[:2]) + (-WHEEL_DROP,)),
        hub=vec(hub[:2]), axis=vec(axle[:2]),
        tread=span(0.011), rim=span(0.0116),
        face=vec(along(0.0116), along(0.0136)), bolts="\n".join(bolts))


HEAD = """<mujoco model="hexapod">
  <!-- Generated by hexapod/model.py; do not edit by hand. -->
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" iterations="6" ls_iterations="12"/>

  <visual>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.25 0.25 0.25" specular="0.05 0.05 0.05"/>
    <quality shadowsize="4096" offsamples="8"/>
    <global offwidth="1280" offheight="720" azimuth="145" elevation="-16"/>
    <rgba haze="0.44 0.45 0.46 1"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.6 0.61 0.63" rgb2="0.33 0.34 0.35" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.43 0.43 0.44" rgb2="0.45 0.45 0.46" mark="edge" markrgb="0.37 0.37 0.38" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="1 1" texuniform="true"/>
    <texture name="terrain" type="2d" file="terrain_tex.png"/>
    <material name="terrain" texture="terrain" texrepeat="1 1"/>
    <hfield name="terrain" file="terrain.bin" size="{terrain_size}"/>
    <hfield name="terrain_lite" file="terrain_lite.bin" size="{terrain_lite_size}"/>
    <material name="carbon" rgba="0.06 0.06 0.065 1" specular="0.6" shininess="0.85"/>
    <material name="servo" rgba="0.08 0.08 0.085 1" specular="0.15" shininess="0.25"/>
    <material name="bracket" rgba="0.5 0.51 0.53 1" specular="0.7" shininess="0.6"/>
    <material name="silver" rgba="0.66 0.67 0.69 1" specular="0.8" shininess="0.7"/>
    <material name="rubber" rgba="0.13 0.13 0.13 1" specular="0.03" shininess="0.05"/>
    <material name="pcb" rgba="0.05 0.22 0.12 1" specular="0.35" shininess="0.5"/>
    <material name="chip" rgba="0.03 0.03 0.035 1" specular="0.4" shininess="0.5"/>
    <material name="battery" rgba="0.14 0.15 0.17 1" specular="0.55" shininess="0.65"/>
    <material name="camera" rgba="0.2 0.2 0.21 1" specular="0.45" shininess="0.55"/>
    <material name="glass" rgba="0.015 0.015 0.02 1" specular="1" shininess="1" reflectance="0.25"/>
    <material name="lens" rgba="0.02 0.02 0.025 1" specular="1" shininess="1"/>
    <material name="lidar" rgba="0.12 0.12 0.13 1" specular="0.4" shininess="0.5"/>
    <material name="cable" rgba="0.03 0.03 0.03 1" specular="0.2" shininess="0.3"/>
    <material name="led" rgba="0.25 1 0.35 1" emission="0.8"/>
    <material name="shell" rgba="0.1 0.1 0.105 1" specular="0.35" shininess="0.5"/>
    <material name="logo_ink" rgba="0.93 0.93 0.94 1" specular="0.25" shininess="0.3"/>
    <mesh name="plate" vertex="0.17 0 -0.003  0.12 0.095 -0.003  -0.12 0.095 -0.003  -0.17 0 -0.003  -0.12 -0.095 -0.003  0.12 -0.095 -0.003
                               0.17 0 0.003  0.12 0.095 0.003  -0.12 0.095 0.003  -0.17 0 0.003  -0.12 -0.095 0.003  0.12 -0.095 0.003"/>
{logo_meshes}
  </asset>

  <default>
    <joint damping="0.1" armature="0.002"/>
    <geom contype="0" conaffinity="0"/>
    <motor ctrlrange="-2 2"/>
    <default class="coxa">
      <joint axis="0 0 1" range="-0.6 0.6"/>
      <geom type="capsule" size="0.012" mass="0.04" group="3" rgba="0 0 0 0"/>
    </default>
    <default class="femur">
      <joint axis="0 1 0" range="-0.9 0.9"/>
      <geom type="capsule" size="0.01" mass="0.06" group="3" rgba="0 0 0 0"/>
    </default>
    <default class="tibia">
      <joint axis="0 1 0" range="-0.9 0.9"/>
      <geom type="capsule" size="0.008" mass="0.03" group="3" rgba="0 0 0 0"/>
    </default>
    <default class="foot">
      <geom type="sphere" size="0.012" mass="0.01" material="rubber" conaffinity="3" condim="3" friction="1 0.005 0.0001"/>
    </default>
    <default class="wheel">
      <joint damping="0.005" armature="0.0005"/>
    </default>
    <default class="tire">
      <geom type="sphere" size="{radius}" mass="0.07" group="3" rgba="0 0 0 0" conaffinity="3" condim="3" friction="1 0.005 0.0001"/>
    </default>
    <default class="belly">
      <geom type="sphere" size="0.025" mass="0" group="3" rgba="0 0 0 0" conaffinity="3" condim="3" friction="0.6 0.005 0.0001"/>
    </default>
    <default class="visual">
      <geom density="0"/>
    </default>
  </default>

  <worldbody>
    <light name="fill" directional="true" pos="-1 1 3" dir="0.4 -0.4 -1" diffuse="0.35 0.35 0.36" specular="0.05 0.05 0.05" castshadow="false"/>
    <light name="rim" directional="true" pos="-1 1 1.5" dir="0.5 -0.5 -0.5" diffuse="0.22 0.22 0.23" specular="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.005" material="floor" contype="1" conaffinity="0" condim="3" friction="1 0.005 0.0001"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="{terrain_pos}" material="terrain" group="2" contype="2" conaffinity="0" condim="3" friction="1 0.005 0.0001"/>
    <geom name="terrain_lite" type="hfield" hfield="terrain_lite" pos="{terrain_pos}" material="terrain" group="4" contype="0" conaffinity="0"/>
    <body name="torso" pos="0 0 0.145">
      <freejoint name="root"/>
      <light name="key" mode="trackcom" pos="0.8 -1.1 2" dir="-0.35 0.5 -1" diffuse="0.9 0.9 0.9" specular="0.4 0.4 0.4" cutoff="70" exponent="2" castshadow="true"/>
      <geom name="torso" type="box" size="0.15 0.08 0.025" mass="1.2" group="3" rgba="0 0 0 0"/>
      <geom name="belly_front" class="belly" pos="0.09 0 -0.002"/>
      <geom name="belly_rear" class="belly" pos="-0.09 0 -0.002"/>
      <geom name="head" type="sphere" pos="0.16 0 0.01" size="0.025" mass="0.05" group="3" rgba="0 0 0 0"/>

      <geom class="visual" type="mesh" mesh="plate" pos="0 0 0.022" material="carbon"/>
      <geom class="visual" type="mesh" mesh="plate" pos="0 0 -0.022" material="carbon"/>
      <geom class="visual" type="cylinder" size="0.004 0.019" pos="0.06 0.085 0" material="silver"/>
      <geom class="visual" type="cylinder" size="0.004 0.019" pos="0.06 -0.085 0" material="silver"/>
      <geom class="visual" type="cylinder" size="0.004 0.019" pos="-0.06 0.085 0" material="silver"/>
      <geom class="visual" type="cylinder" size="0.004 0.019" pos="-0.06 -0.085 0" material="silver"/>
      <geom class="visual" type="cylinder" size="0.004 0.019" pos="-0.14 0.03 0" material="silver"/>
      <geom class="visual" type="cylinder" size="0.004 0.019" pos="-0.14 -0.03 0" material="silver"/>
      <geom class="visual" type="cylinder" size="0.0028 0.0006" pos="0.06 0.085 0.0256" material="silver"/>
      <geom class="visual" type="cylinder" size="0.0028 0.0006" pos="0.06 -0.085 0.0256" material="silver"/>
      <geom class="visual" type="cylinder" size="0.0028 0.0006" pos="-0.06 0.085 0.0256" material="silver"/>
      <geom class="visual" type="cylinder" size="0.0028 0.0006" pos="-0.06 -0.085 0.0256" material="silver"/>
      <geom class="visual" type="cylinder" size="0.0028 0.0006" pos="-0.14 0.03 0.0256" material="silver"/>
      <geom class="visual" type="cylinder" size="0.0028 0.0006" pos="-0.14 -0.03 0.0256" material="silver"/>
      <geom class="visual" type="box" size="0.05 0.03 0.011" pos="-0.045 0 -0.006" material="battery"/>
      <geom class="visual" type="box" size="0.004 0.0303 0.0113" pos="-0.03 0 -0.006" material="cable"/>
      <geom class="visual" type="box" size="0.03 0.028 0.0008" pos="0.045 0 -0.0172" material="pcb"/>
      <geom class="visual" type="box" size="0.007 0.007 0.001" pos="0.045 0.006 -0.0154" material="chip"/>
      <geom class="visual" type="box" size="0.004 0.009 0.003" pos="0.062 -0.012 -0.0134" material="chip"/>

      <geom class="visual" type="cylinder" size="0.0015 0.003" pos="-0.025 0.024 0.028" material="silver"/>
      <geom class="visual" type="cylinder" size="0.0015 0.003" pos="-0.025 -0.024 0.028" material="silver"/>
      <geom class="visual" type="cylinder" size="0.0015 0.003" pos="-0.095 0.024 0.028" material="silver"/>
      <geom class="visual" type="cylinder" size="0.0015 0.003" pos="-0.095 -0.024 0.028" material="silver"/>
      <geom class="visual" type="box" size="0.042 0.029 0.0008" pos="-0.06 0 0.0318" material="pcb"/>
      <geom class="visual" type="box" size="0.007 0.007 0.0008" pos="-0.052 0.004 0.0334" material="chip"/>
      <geom class="visual" type="box" size="0.006 0.0035 0.0007" pos="-0.038 -0.01 0.0333" material="chip"/>
      <geom class="visual" type="box" size="0.022 0.0025 0.003" pos="-0.06 0.0255 0.0356" material="chip"/>
      <geom class="visual" type="box" size="0.007 0.0065 0.0045" pos="-0.097 0.011 0.0371" material="silver"/>
      <geom class="visual" type="box" size="0.007 0.0065 0.0045" pos="-0.097 -0.008 0.0371" material="silver"/>
      <geom class="visual" type="box" size="0.024 0.02 0.004" pos="0.055 0 0.029" material="servo"/>
      <geom class="visual" type="cylinder" size="0.017 0.009" pos="0.06 0 0.042" material="lidar"/>
      <geom class="visual" type="cylinder" size="0.0172 0.0028" pos="0.06 0 0.043" material="lens"/>
      <geom class="visual" type="cylinder" size="0.0165 0.0012" pos="0.06 0 0.0512" material="servo"/>
      <geom class="visual" type="box" size="0.005 0.008 0.0022" pos="0.15 0 0.0272" material="bracket"/>
      <geom class="visual" type="box" size="0.0085 0.045 0.0105" pos="0.15 0 0.0399" material="camera"/>
      <geom class="visual" type="box" size="0.0005 0.043 0.0085" pos="0.1588 0 0.0399" material="glass"/>
      <geom class="visual" type="cylinder" fromto="0.1588 -0.031 0.0399 0.1594 -0.031 0.0399" size="0.0052" material="silver"/>
      <geom class="visual" type="cylinder" fromto="0.1588 -0.012 0.0399 0.1594 -0.012 0.0399" size="0.0034" material="silver"/>
      <geom class="visual" type="cylinder" fromto="0.1588 0.007 0.0399 0.1594 0.007 0.0399" size="0.0052" material="silver"/>
      <geom class="visual" type="cylinder" fromto="0.1588 0.03 0.0399 0.1594 0.03 0.0399" size="0.0052" material="silver"/>
      <geom class="visual" type="cylinder" fromto="0.1594 -0.031 0.0399 0.1599 -0.031 0.0399" size="0.0042" material="lens"/>
      <geom class="visual" type="cylinder" fromto="0.1594 -0.012 0.0399 0.1599 -0.012 0.0399" size="0.0026" material="lens"/>
      <geom class="visual" type="cylinder" fromto="0.1594 0.007 0.0399 0.1599 0.007 0.0399" size="0.0042" material="lens"/>
      <geom class="visual" type="cylinder" fromto="0.1594 0.03 0.0399 0.1599 0.03 0.0399" size="0.0042" material="lens"/>
      <geom class="visual" type="sphere" size="0.0016" pos="-0.13 0.035 0.0262" material="led"/>
      <geom class="visual" type="box" size="0.047 0.036 0.0095" pos="-0.06 0 0.0345" material="shell"/>
{logo}

{stators}

"""

TAIL = """    </body>
  </worldbody>

  <actuator>
{motors}
  </actuator>
</mujoco>
"""


def main():
    ASSETS.mkdir(exist_ok=True)
    stators = "\n".join(COXA_STATOR.format(x=x - 0.010 * math.cos(yaw), y=y - 0.010 * math.sin(yaw), yaw=yaw)
                        for _, (x, y), yaw, _ in LEGS)
    legs = []
    for name, (x, y), yaw, wheel in LEGS:
        top = LEG_TOP.format(n=name, x=x, y=y, yaw=yaw, tibia_end=-0.09 if wheel else -0.105)
        legs.append(top + (wheel_end(name, yaw) if wheel else FOOT_END.format(n=name)))
    motors = [f'    <motor name="{n}_{s}" joint="{n}_{s}"/>' for n, *_ in LEGS for s in ("coxa", "femur", "tibia")]
    motors += [f'    <motor name="{n}_wheel" joint="{n}_wheel" ctrlrange="-0.6 0.6"/>' for n, *_, wheel in LEGS if wheel]
    heights = make_terrain()
    write_png(TEXTURE_FILE, terrain_texture(heights)[::-1])
    half_x, half_y = LEVELS * TILE / 2 + BORDER, len(TERRAIN_TYPES) * TILE / 2 + BORDER
    logo_meshes, logo_geoms = logo_xml()
    head = (HEAD.replace("{radius}", str(WHEEL_RADIUS)).replace("{stators}", stators)
            .replace("{logo_meshes}", logo_meshes).replace("{logo}", logo_geoms)
            .replace("{terrain_size}", f"{half_x} {half_y} {heights.max():.6f} 0.1")
            .replace("{terrain_lite_size}", f"{half_x} {half_y} {heights[::2, ::2].max():.6f} 0.1")
            .replace("{terrain_pos}", f"{START_X - BORDER + half_x} 0 0"))
    OUT.write_text(head + "\n".join(legs) + "\n" + TAIL.format(motors="\n".join(motors)), encoding="utf-8")
    digest = hashlib.sha1(TERRAIN_FILE.read_bytes()).hexdigest()[:12]
    print(f"wrote {OUT.name} and {TERRAIN_FILE.name}; terrain {2 * half_x:.0f} x {2 * half_y:.0f} m, "
          f"{heights.shape[1]}x{heights.shape[0]} cells, highest point {heights.max() * 100:.1f} cm, digest {digest}")


if __name__ == "__main__":
    main()
