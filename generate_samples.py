#!/usr/bin/env python3
"""Generate 100 realistic Michelson interferometer fringe images."""
import numpy as np
import os
from PIL import Image

OUT = os.path.join(os.path.dirname(__file__), "static/sample_images")
os.makedirs(OUT, exist_ok=True)

W, H = 640, 480
LAMBDA_NM = 650.0   # laser wavelength nm
PX_TO_MM  = 0.05    # 1 pixel = 0.05 mm (realistic for 640px breadboard view)

np.random.seed(42)

def gaussian_beam(W, H, cx=None, cy=None, sx=None, sy=None):
    if cx is None: cx = W / 2
    if cy is None: cy = H / 2
    if sx is None: sx = W * 0.45
    if sy is None: sy = H * 0.45
    x = np.linspace(0, W-1, W)
    y = np.linspace(0, H-1, H)
    X, Y = np.meshgrid(x, y)
    return np.exp(-((X-cx)**2/(2*sx**2) + (Y-cy)**2/(2*sy**2)))

def make_fringe(idx, phase_shift_rad, fringe_period_px=38, contrast=0.88,
                noise_sigma=0.03, tilt_rad=0.0, beam_cx=None, beam_cy=None):
    """
    Generate one fringe image.
    phase_shift_rad: optical phase = 4π Δx / λ  →  Δx = phase * λ / (4π)
    fringe_period_px: pixels per fringe
    contrast: V = (Imax-Imin)/(Imax+Imin)
    """
    x = np.linspace(0, W-1, W)
    y = np.linspace(0, H-1, H)
    X, Y = np.meshgrid(x, y)

    # fringe phase includes tilt
    phase = (2*np.pi / fringe_period_px) * (X + Y * np.tan(tilt_rad)) + phase_shift_rad

    # Gaussian beam envelope
    beam = gaussian_beam(W, H, cx=beam_cx, cy=beam_cy,
                          sx=W*0.40 + np.random.randn()*10,
                          sy=H*0.38 + np.random.randn()*8)

    I = beam * (1.0 + contrast * np.cos(phase))
    # Normalise to [0,1], add noise, clip
    I = I / I.max()
    I += np.random.normal(0, noise_sigma, I.shape)
    I = np.clip(I, 0, 1)

    img = (I * 255).astype(np.uint8)
    # Save as grayscale PNG
    fname = f"fringe_{idx:03d}.png"
    Image.fromarray(img, mode='L').save(os.path.join(OUT, fname))
    return fname

count = 0

# ── Group 1 (001–010): Reference / zero displacement ─────────────────────────
for i in range(10):
    count += 1
    make_fringe(count,
                phase_shift_rad=0.0 + np.random.uniform(-0.05, 0.05),
                fringe_period_px=38 + np.random.randint(-2, 3),
                contrast=0.88 + np.random.uniform(-0.03, 0.03),
                noise_sigma=0.025,
                tilt_rad=np.random.uniform(-0.01, 0.01))

# ── Group 2 (011–030): Small displacement 0.1 – 0.6 μm (1–2 fringes) ─────────
displacements_g2 = np.linspace(0.10, 0.60, 20)
for d in displacements_g2:
    count += 1
    phase = (4*np.pi * d*1e-3) / (LAMBDA_NM*1e-6)   # d in μm → mm; λ in mm
    make_fringe(count,
                phase_shift_rad=phase + np.random.uniform(-0.04, 0.04),
                fringe_period_px=36 + np.random.randint(-3, 4),
                contrast=0.85 + np.random.uniform(-0.05, 0.05),
                noise_sigma=0.030)

# ── Group 3 (031–060): Medium displacement 0.7 – 3.0 μm (3–9 fringes) ────────
displacements_g3 = np.linspace(0.70, 3.00, 30)
for d in displacements_g3:
    count += 1
    phase = (4*np.pi * d*1e-3) / (LAMBDA_NM*1e-6)
    make_fringe(count,
                phase_shift_rad=phase + np.random.uniform(-0.06, 0.06),
                fringe_period_px=34 + np.random.randint(-4, 5),
                contrast=0.82 + np.random.uniform(-0.06, 0.06),
                noise_sigma=0.035,
                tilt_rad=np.random.uniform(-0.02, 0.02))

# ── Group 4 (061–080): Large displacement 3.5 – 8.0 μm (11–25 fringes) ───────
displacements_g4 = np.linspace(3.5, 8.0, 20)
for d in displacements_g4:
    count += 1
    phase = (4*np.pi * d*1e-3) / (LAMBDA_NM*1e-6)
    make_fringe(count,
                phase_shift_rad=phase,
                fringe_period_px=30 + np.random.randint(-4, 5),
                contrast=0.78 + np.random.uniform(-0.07, 0.07),
                noise_sigma=0.040,
                tilt_rad=np.random.uniform(-0.03, 0.03))

# ── Group 5 (081–100): High disp 8–15 μm, varying conditions ─────────────────
displacements_g5 = np.linspace(8.0, 15.0, 20)
for i, d in enumerate(displacements_g5):
    count += 1
    phase = (4*np.pi * d*1e-3) / (LAMBDA_NM*1e-6)
    make_fringe(count,
                phase_shift_rad=phase,
                fringe_period_px=28 + np.random.randint(-5, 6),
                contrast=0.72 + np.random.uniform(-0.10, 0.10),
                noise_sigma=0.045 + i*0.002,
                tilt_rad=np.random.uniform(-0.04, 0.04),
                beam_cx=W/2 + np.random.uniform(-20, 20),
                beam_cy=H/2 + np.random.uniform(-15, 15))

print(f"Generated {count} fringe images → {OUT}")
