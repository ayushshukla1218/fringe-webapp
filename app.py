"""
app.py — Optical Fringe Analysis Web App
Flask backend for Michelson interferometer fringe image processing.
"""
import os, json, glob
from flask import Flask, render_template, request, jsonify, send_from_directory
from fringe_analysis import analyse, load_gray
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

BASE     = os.path.dirname(__file__)
SAMPLES  = os.path.join(BASE, 'static/sample_images')
UPLOADS  = os.path.join(BASE, 'static/uploads')

# Cache reference phase (fringe_001.png)
_REF_PHASE = None
def get_ref_phase():
    global _REF_PHASE
    if _REF_PHASE is None:
        ref_path = os.path.join(SAMPLES, 'fringe_001.png')
        if os.path.exists(ref_path):
            from fringe_analysis import load_gray, ftm_phase, intensity_profile
            img = load_gray(ref_path)
            _, phase_profile = ftm_phase(img)
            _REF_PHASE = phase_profile
    return _REF_PHASE


@app.route('/')
def index():
    samples = sorted(os.listdir(SAMPLES)) if os.path.exists(SAMPLES) else []
    return render_template('index.html', samples=samples, total=len(samples))


@app.route('/analyse/sample/<filename>')
def analyse_sample(filename):
    path = os.path.join(SAMPLES, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    ref = get_ref_phase()
    result = analyse(path, ref_phase=ref)
    result.pop('phase_profile', None)
    return jsonify(result)


@app.route('/analyse/upload', methods=['POST'])
def analyse_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400
    data = f.read()
    ref = get_ref_phase()
    result = analyse(data, ref_phase=ref)
    result.pop('phase_profile', None)
    return jsonify(result)


@app.route('/batch')
def batch():
    """Analyse all 100 sample images and return summary JSON."""
    paths = sorted(glob.glob(os.path.join(SAMPLES, 'fringe_*.png')))
    ref = get_ref_phase()
    summary = []
    for p in paths:
        try:
            from fringe_analysis import load_gray, fringe_contrast, spatial_fft_analysis, intensity_profile, ftm_phase, phase_to_displacement
            img = load_gray(p)
            profile = intensity_profile(img)
            C = fringe_contrast(img)
            _, _, _, period_px, period_mm = spatial_fft_analysis(profile)
            _, phase_profile = ftm_phase(img)
            disp = None
            if ref is not None:
                min_len = min(len(ref), len(phase_profile))
                delta = np.unwrap(phase_profile[:min_len] - ref[:min_len])
                disp = round(phase_to_displacement(abs(float(np.median(delta)))), 4)
            summary.append({
                'file':           os.path.basename(p),
                'contrast':       round(C, 4),
                'period_px':      round(period_px, 2),
                'period_mm':      round(period_mm, 4),
                'displacement_um': disp,
            })
        except Exception as e:
            summary.append({'file': os.path.basename(p), 'error': str(e)})
    return jsonify(summary)


@app.route('/static/sample_images/<path:filename>')
def serve_sample(filename):
    return send_from_directory(SAMPLES, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
