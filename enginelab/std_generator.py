"""Synthetic standard components (exact solids) to widen the training data of the
component classifier where real CAD is scarce: bearings, shafts, pulleys, idlers, gears,
stepper motors, linear rails and carriages, extrusions, couplings, plus screws/nuts/washers.

Every part is built with exact booleans (manifold3d), randomly sized from real catalogue
ranges, and comes with its family label.
"""
from __future__ import annotations

import math

import numpy as np

from .components import BEARINGS, NEMA, RAILS
from .generator import CS, Man, box, cyl_x, cyl_z, extrude_yz, gear_section, revolve, to_trimesh, tube_z


def _bearing(rng):
    key = rng.choice(list(BEARINGS))
    d, D, B = BEARINGS[key]
    r_in, r_out = d / 2, D / 2
    t = (r_out - r_in)
    inner = tube_z(r_in, r_in + 0.32 * t, B, 48)
    outer = tube_z(r_out - 0.32 * t, r_out, B, 48)
    body = inner + outer
    if rng.random() < 0.7:   # shields fill the gap (ZZ / 2RS)
        sh = 0.12 * B
        for s in (-1, 1):
            body = body + tube_z(r_in + 0.3 * t, r_out - 0.3 * t, sh, 48).translate((0, 0, s * (B / 2 - sh / 2)))
        body = body + tube_z(r_in + 0.31 * t, r_out - 0.31 * t, B * 0.8, 48)
    if rng.random() < 0.3:   # flanged (F-series)
        body = body + tube_z(r_out - 0.3 * t, r_out + 0.12 * D, 0.15 * B + 0.3, 48).translate((0, 0, B / 2 - 0.08 * B))
    return body, "bearing"


def _shaft(rng):
    d = float(rng.choice([3, 4, 5, 6, 8, 10, 12, 16, 20]))
    L = float(rng.uniform(6, 60)) * d
    s = cyl_z(d / 2, L, 32)
    if rng.random() < 0.4:   # flat (D-cut)
        s = s - box(d, d, L * 0.3, (0, d * 0.42 + d / 2, L * 0.3))
    if rng.random() < 0.3:   # keyway
        s = s - box(d * 0.25, d * 0.2, L * 0.25, (0, d / 2, -L * 0.2))
    return s, "transmission"


def _pulley(rng):
    z = int(rng.choice([16, 20, 24, 30, 36, 40, 60]))
    pitch = 2.0 if rng.random() < 0.8 else 3.0
    r_p = z * pitch / (2 * math.pi)
    width = float(rng.choice([6, 9, 10, 15]))
    teeth = extrude_yz(gear_section(z, r_p - 0.75, r_p - 0.25), width)
    flange = cyl_x(r_p + 1.0, 1.0, 48)
    body = teeth + flange.translate((width / 2 + 0.5, 0, 0)) + flange.translate((-width / 2 - 0.5, 0, 0))
    if rng.random() < 0.7:
        body = body + cyl_x(max(4.0, r_p * 0.7), 7, 32).translate((width / 2 + 4.5, 0, 0))
    bore = float(rng.choice([5, 6.35, 8]))
    return body - cyl_x(bore / 2, 100, 24), "transmission"


def _idler(rng):
    d_in = float(rng.choice([3, 5, 8]))
    D = float(rng.uniform(10, 24))
    w = float(rng.uniform(6, 11))
    prof = [(d_in / 2, -w / 2 - 1), (D / 2 + 1.5, -w / 2 - 1), (D / 2 + 1.5, -w / 2), (D / 2, -w / 2),
            (D / 2, w / 2), (D / 2 + 1.5, w / 2), (D / 2 + 1.5, w / 2 + 1), (d_in / 2, w / 2 + 1)]
    return revolve(prof, 48), "transmission"


def _gear(rng):
    m = float(rng.choice([0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]))
    z = int(rng.integers(12, 90))
    r = m * z / 2
    b = float(rng.uniform(4, 12)) * m
    g = extrude_yz(gear_section(z, r - 1.25 * m, r + m), b)
    if rng.random() < 0.5:
        g = g + cyl_x(r * 0.4, b * 1.8, 32)
    g = g - cyl_x(max(1.5, r * float(rng.uniform(0.1, 0.3))), 10 * b, 24)
    if z > 40 and rng.random() < 0.5:
        for k in range(6):
            a = 2 * math.pi * k / 6
            g = g - cyl_x(r * 0.18, 10 * b, 20).translate((0, r * 0.6 * math.cos(a), r * 0.6 * math.sin(a)))
    return g, "transmission"


def _motor(rng):
    frame = rng.choice(["NEMA 14", "NEMA 17", "NEMA 17", "NEMA 23"])
    s = NEMA[frame]
    L = float(rng.uniform(0.4, 1.9)) * s
    body = box(s, s, L) ^ cyl_z(s * float(rng.uniform(0.64, 0.7)), L * 2, 8).rotate((0, 0, 22.5))   # chamfered corners
    if rng.random() < 0.5:   # end caps + laminated stack read as grooves round the body
        body = body - (box(s * 1.1, s * 1.1, 0.8) - box(s * 0.97, s * 0.97, 1.0)).translate((0, 0, L * 0.3))
        body = body - (box(s * 1.1, s * 1.1, 0.8) - box(s * 0.97, s * 0.97, 1.0)).translate((0, 0, -L * 0.3))
    body = body + cyl_z(s * 0.26, 2.0, 40).translate((0, 0, L / 2 + 1))
    d = 5.0 if s < 50 else 6.35
    lsh = float(rng.uniform(12, 26))
    shaft = cyl_z(d / 2, lsh, 24).translate((0, 0, L / 2 + lsh / 2 + 1))
    if rng.random() < 0.5:
        shaft = shaft - box(d, d, lsh * 0.8, (d * 0.92, 0, L / 2 + lsh * 0.6))
    body = body + shaft
    if rng.random() < 0.5:   # connector housing on one side
        body = body + box(s * 0.35, 6.0, min(L, 10.0) * 0.8, (0, s / 2 + 2.5, -L / 2 + min(L, 10.0) * 0.45))
    for sx in (-1, 1):
        for sy in (-1, 1):
            body = body - cyl_z(1.5, 8, 16).translate((sx * s * 0.37, sy * s * 0.37, L / 2))
    return body, "motor"


def _dc_motor(rng):
    """Brushed DC / coreless can motor (also inside hobby and smart servos): can with flats,
    end-bell boss, shaft, often two solder tabs."""
    d = float(rng.uniform(6, 40))
    L = float(rng.uniform(1.1, 2.2)) * d
    can = cyl_z(d / 2, L, 48)
    if rng.random() < 0.6:
        can = can ^ box(d, d * float(rng.uniform(0.72, 0.9)), L * 1.1)
    can = can + cyl_z(d * 0.18, d * 0.08 + 0.6, 32).translate((0, 0, L / 2 + d * 0.04))
    lsh = float(rng.uniform(0.25, 0.8)) * d
    can = can + cyl_z(max(0.5, d * 0.05), lsh, 16).translate((0, 0, L / 2 + lsh / 2))
    if rng.random() < 0.6:
        for sx in (-1, 1):
            can = can + box(d * 0.08, 0.4 + d * 0.02, d * 0.2, (sx * d * 0.3, 0, -L / 2 - d * 0.1))
    return can, "motor"


def _round_stepper(rng):
    """Round / can-stack stepper (e.g. 36 mm round, 28BYJ): round body, two mounting ears, shaft."""
    D = float(rng.uniform(20, 42))
    L = float(rng.uniform(0.35, 0.8)) * D
    body = cyl_z(D / 2, L, 48)
    ears = box(D * 1.4, D * 0.22, 1.2, (0, 0, L / 2 - 0.6)) + cyl_z(D * 0.11, 1.2, 24).translate((D * 0.7, 0, L / 2 - 0.6)) \
        + cyl_z(D * 0.11, 1.2, 24).translate((-D * 0.7, 0, L / 2 - 0.6))
    for sx in (-1, 1):
        ears = ears - cyl_z(D * 0.05, 4, 16).translate((sx * D * 0.7, 0, L / 2))
    body = body + ears + cyl_z(D * 0.25, 1.5, 32).translate((0, 0, L / 2 + 0.75))
    off = D * 0.2 if rng.random() < 0.4 else 0.0   # 28BYJ-style offset output
    lsh = float(rng.uniform(8, 18))
    body = body + cyl_z(max(1.5, D * 0.06), lsh, 16).translate((off, 0, L / 2 + lsh / 2))
    if rng.random() < 0.5:
        body = body + box(D * 0.45, D * 0.25, L * 0.7, (0, -D / 2, -L * 0.1))   # wire cover
    return body, "motor"


def _servo_shell(rng):
    """One housing piece of a servo (upper / middle / lower case): thin-walled open box,
    bearing boss for the output, screw bosses in the corners."""
    L, W, H = float(rng.uniform(22, 50)), float(rng.uniform(12, 32)), float(rng.uniform(5, 30))
    t = float(rng.uniform(0.8, 2.0))
    shell = box(L, W, H) - box(L - 2 * t, W - 2 * t, H, (0, 0, t))
    if rng.random() < 0.6:   # round corners like an injection moulded case
        shell = shell ^ cyl_z(math.hypot(L, W) * 0.47, H * 2, 64).scale((1.0, W / L * 1.25, 1.0))
    for sx in (-1, 1):
        for sy in (-1, 1):
            shell = shell + tube_z(0.8, 2.2, H - t, 20).translate((sx * (L / 2 - t - 2.2), sy * (W / 2 - t - 2.2), t / 2))
    if rng.random() < 0.7:   # output shaft hole with a bearing boss
        r = W * float(rng.uniform(0.15, 0.3))
        shell = shell + tube_z(r, r + 1.5, H * 0.6, 32).translate((L * 0.25, 0, -H * 0.2))
        shell = shell - cyl_z(r, H * 3, 32).translate((L * 0.25, 0, 0))
    if rng.random() < 0.5:   # mounting ears
        shell = shell + (box(L * 1.36, W, 2.2, (0, 0, -H / 2 + 1.1)) - box(L * 1.02, W * 1.1, 3, (0, 0, -H / 2 + 1.1)))
    return shell, "motor"


def _pin_header(rng):
    """Pin header: plastic strip with square pins sticking out both sides (2.54 / 2.0 mm pitch)."""
    pitch = float(rng.choice([2.54, 2.54, 2.0, 1.27]))
    n, rows = int(rng.integers(2, 21)), int(rng.choice([1, 1, 2]))
    body = box(n * pitch, rows * pitch, pitch)
    s = 0.25 * pitch
    for i in range(n):
        for j in range(rows):
            x, y = (i - (n - 1) / 2) * pitch, (j - (rows - 1) / 2) * pitch
            body = body + box(s, s, pitch * float(rng.uniform(3.2, 4.8)), (x, y, pitch * 0.6))
    return body, "electronics"


def _connector_housing(rng):
    """Wire-to-board connector housing (JST PH/XH style): box with a row of contact cavities."""
    pitch = float(rng.choice([2.0, 2.5, 1.25, 3.96]))
    n = int(rng.integers(2, 13))
    L, W, H = n * pitch + 2 * pitch, pitch * float(rng.uniform(1.8, 2.6)), pitch * float(rng.uniform(2.2, 3.4))
    body = box(L, W, H)
    for i in range(n):
        body = body - box(0.55 * pitch, 0.55 * W, H, ((i - (n - 1) / 2) * pitch, 0, H * 0.25))
    if rng.random() < 0.6:   # latch ramp
        body = body + box(L * 0.4, 0.6, H * 0.5, (0, W / 2 + 0.3, 0))
    if rng.random() < 0.5:   # PCB pins under the housing
        for i in range(n):
            body = body + box(0.5, 0.5, 3.5, ((i - (n - 1) / 2) * pitch, 0, -H / 2 - 1.5))
    return body, "electronics"


def _power_connector(rng):
    """XT30 / XT60-style power connector: D-shaped housing with two bullet contacts."""
    s = float(rng.uniform(0.8, 1.6))
    L, W, H = 10 * s, 5 * s, 9 * s
    body = box(L, W, H) ^ cyl_z(L * 0.62, H * 2, 6).scale((1, W / L * 1.6, 1))
    for x in (-L * 0.25, L * 0.25):
        body = body - cyl_z(1.2 * s, H * 1.5, 20).translate((x, 0, H * 0.3))
        body = body + cyl_z(0.9 * s, H * 0.5, 16).translate((x, 0, -H / 2 - H * 0.2))
    return body, "electronics"


def _terminal_block(rng):
    """Screw terminal block: wire entries on the front, screw holes on top, 5.08 / 3.5 mm pitch."""
    pitch = float(rng.choice([5.08, 5.0, 3.5]))
    n = int(rng.integers(2, 7))
    L, W, H = n * pitch, pitch * 1.6, pitch * 2.0
    body = box(L, W, H)
    for i in range(n):
        x = (i - (n - 1) / 2) * pitch
        body = body - box(pitch * 0.6, W * 0.6, H * 0.45, (x, -W * 0.25, -H * 0.05))
        body = body - cyl_z(pitch * 0.28, H, 16).translate((x, W * 0.1, H * 0.4))
        body = body + box(0.8, 0.8, 4.0, (x, 0, -H / 2 - 2.0))
    return body, "electronics"


def _ic_package(rng):
    """SOIC / QFP / SOT style chip: thin moulded body with gull-wing leads."""
    L, W = float(rng.uniform(2.5, 20)), float(rng.uniform(1.5, 20))
    H = float(rng.uniform(0.8, 2.0))
    body = box(L, W, H)
    pitch = float(rng.choice([0.5, 0.65, 1.27]))
    n = max(2, int(L / pitch) - 1)
    sides = (-1, 1) if rng.random() < 0.7 else (1,)
    for i in range(n):
        x = (i - (n - 1) / 2) * pitch
        for sy in sides:
            body = body + box(pitch * 0.45, 1.2, 0.2, (x, sy * (W / 2 + 0.5), -H / 2 + 0.1))
    return body, "electronics"


def _usb_socket(rng):
    """USB-C / micro-USB receptacle: thin metal shell, hollow, with an inner tongue."""
    L, W, H = float(rng.uniform(5, 9)), float(rng.uniform(7, 9.5)), float(rng.uniform(2.6, 3.4))
    shell = box(L, W, H) - box(L * 1.1, W - 0.5, H - 0.5, (0.3, 0, 0))
    shell = shell + box(L * 0.7, W * 0.6, 0.6, (0.3, 0, 0))
    for sx in (-1, 1):
        shell = shell + box(1.0, 0.8, 2.0, (0, sx * (W / 2 + 0.4), -H / 2 - 0.5))
    return shell, "electronics"


def _rail(rng):
    key = rng.choice(list(RAILS))
    w, h = RAILS[key]
    L = float(rng.choice([100, 150, 200, 250, 300, 350, 400, 500]))
    r = box(L, w, h)
    for s in (-1, 1):
        r = r - box(L * 1.1, w * 0.12, h * 0.2, (0, s * w / 2, h * 0.15))
    pitch = 20 if w < 13 else 40
    for x in np.arange(-L / 2 + pitch / 2, L / 2, pitch):
        r = r - cyl_z(w * 0.15, h * 3, 16).translate((x, 0, 0))
    if rng.random() < 0.4:  # carriage instead of rail
        c = box(w * 2.2, w * 1.9, h * 1.1) - box(w * 3, w * 1.05, h * 0.9, (0, 0, -h * 0.4))
        for sx in (-1, 1):
            for sy in (-1, 1):
                c = c - cyl_z(1.3, h * 3, 16).translate((sx * w * 0.5, sy * w * 0.6, 0))
        return c, "linear_motion"
    return r, "linear_motion"


def _extrusion(rng):
    s = float(rng.choice([15, 20, 20, 30, 40]))
    L = float(rng.uniform(80, 800))
    sec = CS.square((s, s), True) - CS.circle(s * 0.13, 24)
    for k in range(4):
        slot = CS.square((s * 0.3, s * 0.32), True).translate((0, s / 2 - s * 0.15)) + \
            CS.square((s * 0.55, s * 0.18), True).translate((0, s / 2 - s * 0.3))
        sec = sec - slot.rotate(90 * k)
    if rng.random() < 0.3:  # 2040 double profile
        sec = sec + sec.translate((s, 0))
    return Man.extrude(sec, L), "structure"


def _coupling(rng):
    D = float(rng.uniform(14, 25))
    L = float(rng.uniform(20, 30))
    c = cyl_z(D / 2, L, 40) - cyl_z(2.5, L * 1.1, 24)
    for z in np.linspace(-L * 0.25, L * 0.25, 3):
        c = c - box(D * 1.2, D * 0.7, 0.8, (0, D * 0.4, z))
    for s in (-1, 1):
        c = c - cyl_x(1.5, D * 2, 16).translate((0, 0, s * L * 0.35))
    return c, "transmission"


def _screw(rng):
    d = float(rng.choice([2, 2.5, 3, 4, 5, 6, 8, 10]))
    L = float(rng.choice([4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50]))
    kind = rng.choice(["shcs", "bhcs", "fhcs"])
    if kind == "shcs":
        head = cyl_z(0.75 * d, d, 32).translate((0, 0, d / 2)) - cyl_z(0.43 * d, d, 6).translate((0, 0, d))
    elif kind == "bhcs":
        head = revolve([(0, 0), (0.95 * d, 0), (0.8 * d, 0.35 * d), (0.4 * d, 0.55 * d), (0, 0.55 * d)], 32)
        head = head - cyl_z(0.3 * d, d, 6).translate((0, 0, 0.55 * d))
    else:
        head = revolve([(0, 0), (d / 2, 0), (d, 0.6 * d), (0, 0.6 * d)], 32).translate((0, 0, -0.6 * d))
    return head + cyl_z(d / 2, L, 24).translate((0, 0, -L / 2)), "fastener"


def _nut(rng):
    d = float(rng.choice([2, 3, 4, 5, 6, 8, 10]))
    n = cyl_z(0.92 * d, 0.8 * d, 6) - cyl_z(d / 2, 2 * d, 24)
    if rng.random() < 0.4:   # washer instead
        n = tube_z(d / 2 + 0.2, d * 1.0, 0.18 * d, 32)
    return n, "fastener"


def _servo(rng):
    """Hobby / smart servo: box body, mounting ears, offset output spline."""
    L, W, H = float(rng.uniform(22, 45)), float(rng.uniform(11, 24)), float(rng.uniform(20, 42))
    body = box(L, W, H)
    if rng.random() < 0.7:
        ears = box(L * 1.38, W, 2.5, (0, 0, H * 0.22)) - box(L * 1.2, 1.2, 3, (0, 0, H * 0.22))
        body = body + ears
    body = body + cyl_z(W * 0.3, 4, 32).translate((L * 0.25, 0, H / 2 + 2))
    body = body + cyl_z(W * 0.14, 4, 20).translate((L * 0.25, 0, H / 2 + 5))
    return body, "motor"


def _bldc(rng):
    """Outrunner / gimbal motor: short wide can with a slotted top and mounting holes."""
    D, H = float(rng.uniform(22, 90)), float(rng.uniform(0.25, 0.7))
    H *= D
    can = cyl_z(D / 2, H, 48)
    for k in range(int(rng.integers(4, 9))):
        a = 2 * math.pi * k / 8
        can = can - box(D * 0.18, D * 0.08, H, (D * 0.3 * math.cos(a), D * 0.3 * math.sin(a), H / 2))
    can = can + cyl_z(D * 0.06, H * 0.6, 20).translate((0, 0, H / 2 + H * 0.3))
    return can - cyl_z(D * 0.02, 3 * H, 12), "motor"


def _gearmotor(rng):
    """DC gear motor: cylinder motor + gearbox block + output shaft (often D-cut)."""
    d = float(rng.uniform(12, 37))
    L = float(rng.uniform(1.0, 2.2)) * d
    m = cyl_z(d / 2, L, 40).translate((0, 0, -L / 2))
    gb = box(d * 1.05, d * 0.85, d * 0.8, (0, 0, d * 0.4)) if rng.random() < 0.5 else cyl_z(d / 2, d * 0.8, 40).translate((0, 0, d * 0.4))
    sh = cyl_z(d * 0.12, d * 0.6, 20).translate((d * 0.15, 0, d * 0.8 + d * 0.3))
    return m + gb + sh, "motor"


def _pcb(rng):
    """Circuit board: thin plate, mounting holes, a few component blocks and headers."""
    L, W, t = float(rng.uniform(20, 160)), float(rng.uniform(15, 110)), float(rng.choice([0.8, 1.0, 1.6]))
    b = box(L, W, t)
    for sx in (-1, 1):
        for sy in (-1, 1):
            b = b - cyl_z(1.6, 3 * t, 16).translate((sx * (L / 2 - 3.5), sy * (W / 2 - 3.5), 0))
    for _ in range(int(rng.integers(2, 12))):
        l, w, h = rng.uniform(2, min(25, L / 3)), rng.uniform(2, min(25, W / 3)), rng.uniform(0.8, 10)
        x, y = rng.uniform(-L / 2 + l, L / 2 - l), rng.uniform(-W / 2 + w, W / 2 - w)
        b = b + box(float(l), float(w), float(h), (float(x), float(y), float(t / 2 + h / 2)))
    return b, "electronics"


def _fan(rng):
    """Axial fan: square frame, round duct, hub with blades; or a blower (scroll)."""
    s = float(rng.choice([25, 30, 40, 50, 60, 80]))
    t = float(rng.choice([7, 10, 15, 20, 25]))
    if rng.random() < 0.7:
        f = box(s, s, t) - cyl_z(s * 0.47, 2 * t, 48)
        for sx in (-1, 1):
            for sy in (-1, 1):
                f = f - cyl_z(s * 0.045, 2 * t, 16).translate((sx * s * 0.4, sy * s * 0.4, 0))
        hub = cyl_z(s * 0.18, t * 0.9, 32)
        for k in range(int(rng.integers(5, 10))):
            a = 360.0 * k / 7
            hub = hub + box(s * 0.3, s * 0.08, t * 0.2).rotate((25, 0, 0)).translate((s * 0.3, 0, 0)).rotate((0, 0, a))
        return f + hub, "electronics"
    scroll = cyl_z(s * 0.5, t, 48) + box(s * 0.35, s * 0.5, t, (s * 0.5, s * 0.25, 0))
    return scroll - cyl_z(s * 0.25, t * 0.8, 40).translate((0, 0, t * 0.3)), "electronics"


def _belt(rng):
    """Closed timing belt (GT2 / AT3 / HTD): a thin stadium-shaped loop with inner teeth."""
    c = float(rng.uniform(40, 220))          # centre distance
    r = float(rng.uniform(6, 30))            # pulley pitch radius
    t = float(rng.choice([1.4, 1.6, 2.5]))
    w = float(rng.choice([6, 9, 10, 15]))
    outer = CS.batch_hull([CS.circle(r + t, 48), CS.circle(r + t, 48).translate((c, 0))])
    inner = CS.batch_hull([CS.circle(r, 48), CS.circle(r, 48).translate((c, 0))])
    belt = Man.extrude(outer - inner, w)
    pitch = float(rng.choice([2.0, 3.0]))
    for x in np.arange(0, c, pitch):
        for y in (r - 0.5, -r + 0.5):
            belt = belt + box(pitch * 0.4, 0.9, w, (float(x), y if y > 0 else y, w / 2))
    return belt, "transmission"


def _stator(rng):
    """Brushless motor stator: laminated core with deep slots around a hub."""
    n = int(rng.choice([9, 12, 18, 24, 36]))
    R = float(rng.uniform(12, 45))
    h = float(rng.uniform(4, 25))
    core = Man.extrude(gear_section(n, R * 0.62, R) + CS.circle(R * 0.6, 48), h)
    return core - cyl_z(R * float(rng.uniform(0.2, 0.4)), 4 * h, 32), "motor"


def _rotor(rng):
    """Outrunner bell with magnets glued inside."""
    R = float(rng.uniform(14, 48))
    h = float(rng.uniform(6, 25))
    t = max(1.0, 0.06 * R)
    bell = tube_z(R - t, R, h, 64) + cyl_z(R, t, 64).translate((0, 0, h / 2 - t / 2))
    for k in range(int(rng.choice([10, 12, 14, 20]))):
        a = 360.0 * k / 14
        bell = bell + box(t * 1.2, 2 * math.pi * R / 16, h * 0.8, (R - t - t * 0.6, 0, -h * 0.05)).rotate((0, 0, a))
    return bell - cyl_z(R * 0.12, 4 * h, 24), "motor"


def _codewheel(rng):
    """Optical encoder code wheel: thin disc with a ring of fine radial slots."""
    R = float(rng.uniform(8, 30))
    t = float(rng.uniform(0.2, 1.0))
    d = cyl_z(R, t, 64) - cyl_z(R * 0.12, 4 * t, 24)
    for k in range(int(rng.choice([32, 48, 64]))):
        a = 360.0 * k / 64
        d = d - box(R * 0.15, R * 0.025, 4 * t, (R * 0.8, 0, 0)).rotate((0, 0, a))
    return d, "electronics"


BUILDERS = [(_belt, 2), (_stator, 1), (_rotor, 1), (_codewheel, 1),(_bearing, 3), (_shaft, 2), (_pulley, 2), (_idler, 1), (_gear, 2), (_motor, 2), (_rail, 2),
            (_extrusion, 1), (_coupling, 1), (_screw, 1), (_nut, 1), (_servo, 2), (_bldc, 1), (_gearmotor, 1),
            (_dc_motor, 2), (_round_stepper, 1), (_servo_shell, 2),
            (_pin_header, 1), (_connector_housing, 1), (_power_connector, 1), (_terminal_block, 1), (_ic_package, 1), (_usb_socket, 1),
            (_pcb, 2), (_fan, 1)]


def random_component(rng):
    fns, w = zip(*BUILDERS)
    w = np.asarray(w, float) / sum(w)
    fn = fns[rng.choice(len(fns), p=w)]
    man, label = fn(rng)
    return to_trimesh(man), label


def dataset_rows(n: int, seed: int = 0):
    """Feature rows for n synthetic parts. Each part also gets a small random 'assembly'
    context (alone, or with copies of itself), since real parts come in sets."""
    from .component_model import row
    from .features import context_features, shape_features
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        try:
            mesh, label = random_component(rng)
            if not mesh.is_watertight or mesh.volume <= 0:
                continue
            info = shape_features(mesh, seed=i)
            copies = int(rng.choice([1, 1, 2, 4]))
            ctx = context_features([info] * copies)[0] if copies > 1 else context_features([info])[0]
            X.append(row(info, ctx, None))
            y.append(label)
        except Exception:
            continue
    return X, y
