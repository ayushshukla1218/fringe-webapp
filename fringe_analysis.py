"""
fringe_analysis.py
Core image processing: spatial FFT, Fourier Transform Method phase extraction,
fringe contrast measurement, displacement calculation.
"""
import numpy as np
import cv2
import base64, io
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

LAMBDA_NM  = 650.0   # laser wavelength nm
PX_TO_MM   = 0.05    # 1 pixel = 0.05 mm


def load_gray(path_or_bytes):
    if isinstance(path_or_bytes, (str, bytes)) and isinstance(path_or_bytes, str):
        img = cv2.imread(path_or_bytes, cv2.IMREAD_GRAYSCALE)
    else:
        arr = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float64) / 255.0


def intensity_profile(img):
    """Horizontal intensity profile averaged over middle 20% of rows."""
    H, W = img.shape
    r0, r1 = int(H * 0.40), int(H * 0.60)
    profile = img[r0:r1, :].mean(axis=0)
    # Normalise
    mn, mx = profile.min(), profile.max()
    if mx > mn:
        profile = (profile - mn) / (mx - mn)
    return profile


def fringe_contrast(img):
    """Michelson contrast C = (Imax - Imin) / (Imax + Imin) over central region."""
    H, W = img.shape
    region = img[int(H*0.3):int(H*0.7), int(W*0.3):int(W*0.7)]
    imax, imin = region.max(), region.min()
    if imax + imin == 0:
        return 0.0
    return float((imax - imin) / (imax + imin))


def spatial_fft_analysis(profile):
    """
    Spatial FFT of the intensity profile.
    Returns: freqs (cycles/pixel), magnitude, dominant_freq, fringe_period_px, fringe_period_mm
    """
    N = len(profile)
    window = np.hanning(N)
    F = np.fft.rfft(profile * window)
    mag = np.abs(F)
    freqs = np.fft.rfftfreq(N)   # cycles per pixel

    # Find dominant peak (exclude DC)
    mag[0] = 0
    peak_idx = np.argmax(mag[1:]) + 1
    dom_freq = freqs[peak_idx]

    fringe_period_px = 1.0 / dom_freq if dom_freq > 0 else 0
    fringe_period_mm = fringe_period_px * PX_TO_MM

    return freqs, mag, dom_freq, fringe_period_px, fringe_period_mm


def ftm_phase(img):
    """
    Fourier Transform Method: extract wrapped phase map from single fringe image.
    Returns phase_map (radians, wrapped), phase_profile (centre row).
    """
    H, W = img.shape
    # FFT of middle row
    row = img[H//2, :]
    F = np.fft.fft(row)
    freqs = np.fft.fftfreq(W)

    # Find positive fundamental frequency peak
    mag = np.abs(F)
    mag[0] = 0
    half = W // 2
    peak = np.argmax(mag[1:half]) + 1

    # Bandpass filter: keep only ±3 bins around peak, shift to DC
    F_filt = np.zeros_like(F)
    bw = max(3, peak // 4)
    F_filt[peak-bw:peak+bw+1] = F[peak-bw:peak+bw+1]
    # Shift to baseband
    F_shifted = np.roll(F_filt, -peak)

    analytic = np.fft.ifft(F_shifted)
    phase_profile = np.angle(analytic)

    # 2-D phase map (same filter applied row-by-row)
    phase_map = np.zeros_like(img)
    for r in range(H):
        Frow = np.fft.fft(img[r, :])
        Frow_f = np.zeros_like(Frow)
        Frow_f[peak-bw:peak+bw+1] = Frow[peak-bw:peak+bw+1]
        Frow_s = np.roll(Frow_f, -peak)
        phase_map[r, :] = np.angle(np.fft.ifft(Frow_s))

    return phase_map, phase_profile


def phase_to_displacement(delta_phase_rad):
    """
    Δφ (radians) → displacement (μm).
    Δx = Δφ · λ / (4π)
    """
    dx_nm = (delta_phase_rad * LAMBDA_NM) / (4 * np.pi)
    return dx_nm / 1000.0   # nm → μm


def analyse(path_or_bytes, ref_phase=None):
    """
    Full analysis pipeline. Returns dict of results + base64-encoded plot PNG.
    ref_phase: reference phase profile (from a reference image) for displacement calc.
    """
    img = load_gray(path_or_bytes)
    profile = intensity_profile(img)
    contrast = fringe_contrast(img)
    freqs, mag, dom_freq, period_px, period_mm = spatial_fft_analysis(profile)
    phase_map, phase_profile = ftm_phase(img)

    # Displacement relative to reference
    displacement_um = None
    mean_delta_phase = None
    if ref_phase is not None:
        min_len = min(len(ref_phase), len(phase_profile))
        delta = phase_profile[:min_len] - ref_phase[:min_len]
        # Unwrap
        delta_unwrapped = np.unwrap(delta)
        mean_delta_phase = float(np.median(delta_unwrapped))
        displacement_um = phase_to_displacement(abs(mean_delta_phase))

    # Fringe count estimate
    fringe_count = int(round(img.shape[1] / period_px)) if period_px > 0 else 0

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Optical Fringe Analysis — Michelson Interferometer', fontsize=13, fontweight='bold')

    # 1. Original image
    axes[0,0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0,0].set_title('Captured Fringe Pattern', fontsize=10)
    axes[0,0].axis('off')

    # 2. Intensity profile
    x_mm = np.arange(len(profile)) * PX_TO_MM
    axes[0,1].plot(x_mm, profile, color='#1565C0', lw=1.3)
    axes[0,1].axhline(0.5, color='red', lw=0.8, ls='--', label='Threshold 0.5')
    axes[0,1].set_xlabel('Position (mm)')
    axes[0,1].set_ylabel('Normalised Intensity')
    axes[0,1].set_title('Horizontal Intensity Profile', fontsize=10)
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    # 3. Spatial FFT spectrum
    freq_mm = freqs / PX_TO_MM   # cycles per mm
    axes[0,2].plot(freq_mm[:len(freq_mm)//2], mag[:len(mag)//2], color='#2E7D32', lw=1.2)
    axes[0,2].axvline(dom_freq/PX_TO_MM, color='red', lw=1.0, ls='--',
                      label=f'Peak: {1/period_mm:.2f} c/mm')
    axes[0,2].set_xlabel('Spatial Frequency (cycles/mm)')
    axes[0,2].set_ylabel('Magnitude')
    axes[0,2].set_title('Spatial FFT Spectrum', fontsize=10)
    axes[0,2].legend(fontsize=8)
    axes[0,2].grid(True, alpha=0.3)

    # 4. Phase map
    pm = axes[1,0].imshow(phase_map, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1,0].set_title('FTM Phase Map (wrapped, radians)', fontsize=10)
    axes[1,0].axis('off')
    fig.colorbar(pm, ax=axes[1,0], fraction=0.046, pad=0.04)

    # 5. Phase profile (centre row)
    x_mm2 = np.arange(len(phase_profile)) * PX_TO_MM
    axes[1,1].plot(x_mm2, phase_profile, color='#BF360C', lw=1.2)
    axes[1,1].set_xlabel('Position (mm)')
    axes[1,1].set_ylabel('Phase (radians)')
    axes[1,1].set_title('Phase Profile — Centre Row', fontsize=10)
    axes[1,1].grid(True, alpha=0.3)

    # 6. Results text panel
    axes[1,2].axis('off')
    axes[1,2].set_facecolor('#f5f5f5')
    results_text = [
        ('Laser wavelength (λ)', f'{LAMBDA_NM:.0f} nm'),
        ('Fringe period', f'{period_px:.1f} px  /  {period_mm:.3f} mm'),
        ('Fringe count (visible)', f'{fringe_count}'),
        ('Fringe contrast (C)', f'{contrast:.4f}'),
        ('Dominant spatial freq', f'{dom_freq/PX_TO_MM:.3f} cycles/mm'),
    ]
    if displacement_um is not None:
        results_text.append(('Δφ (mean)', f'{mean_delta_phase:.4f} rad'))
        results_text.append(('Displacement (Δx)', f'{displacement_um:.4f} μm'))
        results_text.append(('Resolution (λ/2)', '0.325 μm / fringe'))
    axes[1,2].set_title('Measurement Results', fontsize=10, fontweight='bold')
    y = 0.92
    for label, val in results_text:
        axes[1,2].text(0.04, y, label, transform=axes[1,2].transAxes, fontsize=9, color='#333')
        axes[1,2].text(0.62, y, val,   transform=axes[1,2].transAxes, fontsize=9,
                       color='#1565C0', fontweight='bold')
        y -= 0.115

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode()

    return {
        'contrast':         round(contrast, 4),
        'fringe_period_px': round(period_px, 2),
        'fringe_period_mm': round(period_mm, 4),
        'fringe_count':     fringe_count,
        'dom_freq_cpmm':    round(dom_freq / PX_TO_MM, 4),
        'displacement_um':  round(displacement_um, 4) if displacement_um is not None else None,
        'delta_phase_rad':  round(mean_delta_phase, 4) if mean_delta_phase is not None else None,
        'phase_profile':    phase_profile.tolist(),
        'plot_b64':         plot_b64,
    }
