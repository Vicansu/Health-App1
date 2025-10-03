"""
Blink Voice Tracker — Improved (dlib + webrtcvad)

This single-file Flask app replaces the previous Haar cascade approach with dlib-based
facial landmarks for far more accurate blink detection (EAR). It also improves the audio
processing pipeline by using WebRTC VAD to isolate voiced segments and extracts
robust acoustic features (median F0, jitter, shimmer, HNR, energy) to compute a
heuristic voice fatigue score.

REQUIREMENTS (pip):
flask
opencv-python
numpy
pydub
librosa
soundfile
webrtcvad
dlib
gunicorn
ffmpeg-python

SYSTEM: ffmpeg must be installed on the host (apt / brew / windows binary).
Download dlib's pretrained shape predictor and place it next to this file:
  shape_predictor_68_face_landmarks.dat
  (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

USAGE:
  python app.py
  or with gunicorn: gunicorn app:app --workers=4 --bind=0.0.0.0:$PORT

NOTE: This is still a prototype / heuristic. For clinical use, consult experts and
use validated models/datasets.
"""

import io
import os
import base64
import uuid
import math
from datetime import datetime, timedelta
from collections import defaultdict, deque
import tempfile

from flask import Flask, render_template_string, request, jsonify, make_response

import numpy as np
import cv2

# dlib facial landmark detector (preferred)
try:
    import dlib
    DLIB_AVAILABLE = True
except Exception:
    DLIB_AVAILABLE = False

# Audio processing
from pydub import AudioSegment
import librosa
import soundfile as sf
import webrtcvad

# ------------------ Configuration ------------------
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EAR_CONSEC_FRAMES = 2
EAR_SMOOTH_WINDOW = 5  # median smoothing window
CALIB_OPEN_SEC = 3.0
CALIB_CLOSED_SEC = 3.0

app = Flask(__name__)

# Initialize detectors
if DLIB_AVAILABLE:
    detector = dlib.get_frontal_face_detector()
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        raise FileNotFoundError(
            f"Dlib shape predictor not found at {SHAPE_PREDICTOR_PATH}. Download from http://dlib.net/files/"
        )
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
else:
    # fallback to Haar cascade for eyes (less accurate)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ------------------ Utilities ------------------
LEFT_EYE_POINTS = list(range(36, 42))  # dlib 68-pt indices
RIGHT_EYE_POINTS = list(range(42, 48))

def shape_to_np(shape):
    # dlib shape -> (68,2) numpy array
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye):
    # eye is 6x2 array of points
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3]) + 1e-8
    ear = (A + B) / (2.0 * C)
    return ear

# ------------------ Per-session state ------------------
# store per-session blink timestamps, smoothing buffers, and calibration
sessions = defaultdict(lambda: {
    "blinks": deque(maxlen=1000),
    "ear_buffer": deque(maxlen=EAR_SMOOTH_WINDOW),
    "consec_closed": 0,
    "threshold": 0.22,  # default initial threshold
    "calibrating": False,
    "calib_open_vals": [],
    "calib_closed_vals": []
})

# ------------------ HTML UI (improved) ------------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <title>Eye & Voice Health Monitor — Improved</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen p-6 flex flex-col items-center">
  <div class="w-full max-w-5xl">
    <h1 class="text-3xl font-bold mb-6">Eye Blink & Voice Fatigue Tracker</h1>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div class="bg-white p-4 rounded-2xl shadow">
        <h2 class="text-xl font-semibold mb-2">Eye Health</h2>
        <video id="video" class="w-full rounded-lg border" width="480" height="360" autoplay muted playsinline></video>
        <canvas id="canvas" width="480" height="360" class="hidden"></canvas>
        <div class="mt-3">
          <p>Blinks/min: <strong id="blinkRate">—</strong></p>
          <p>Count (1 min): <strong id="blinkCount">—</strong></p>
          <p>EAR (smoothed): <strong id="earVal">—</strong></p>
          <p class="text-sm text-red-600" id="blinkAlert">—</p>
        </div>
        <div class="mt-3 flex gap-2">
          <button id="calibrateBtn" class="px-4 py-2 rounded-xl bg-blue-600 text-white">Calibrate Blink</button>
          <button id="resetCalib" class="px-4 py-2 rounded-xl bg-gray-200">Reset Calibration</button>
        </div>
        <div id="calibStatus" class="mt-2 text-sm text-gray-600">Calibration: <span id="calibState">not calibrated</span></div>
      </div>

      <div class="bg-white p-4 rounded-2xl shadow">
        <h2 class="text-xl font-semibold mb-2">Voice Health</h2>
        <div>
          <p>Fatigue score: <strong id="fatigueScore">—</strong></p>
          <pre id="audioMetrics" class="text-xs text-gray-700 mt-2">—</pre>
        </div>
        <div class="mt-3 flex gap-2">
          <button id="startRec" class="px-4 py-2 rounded-xl bg-green-600 text-white">Start Monitoring</button>
          <button id="stopRec" class="px-4 py-2 rounded-xl bg-red-600 text-white" disabled>Stop</button>
        </div>
        <div class="mt-3 text-sm text-gray-600">Tips: Ensure microphone access allowed. Speak normally while monitoring.</div>
      </div>
    </div>
  </div>

<script>
(async () => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const blinkRateEl = document.getElementById('blinkRate');
  const blinkCountEl = document.getElementById('blinkCount');
  const earValEl = document.getElementById('earVal');
  const blinkAlertEl = document.getElementById('blinkAlert');
  const calibrateBtn = document.getElementById('calibrateBtn');
  const calibStateEl = document.getElementById('calibState');
  const resetCalib = document.getElementById('resetCalib');
  const startRec = document.getElementById('startRec');
  const stopRec = document.getElementById('stopRec');
  const fatigueScoreEl = document.getElementById('fatigueScore');
  const audioMetricsEl = document.getElementById('audioMetrics');

  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  video.srcObject = stream;

  async function sendFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
    try {
      const res = await fetch('/frame', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ image: dataUrl }) });
      const j = await res.json();
      if (j) {
        if (j.blink_rate_min !== undefined) {
          blinkRateEl.textContent = j.blink_rate_min.toFixed(1);
          blinkCountEl.textContent = j.blink_count;
          earValEl.textContent = j.ear_smoothed ? j.ear_smoothed.toFixed(3) : '—';
          blinkAlertEl.textContent = j.alert || '—';
          calibStateEl.textContent = j.calibrated ? 'calibrated' : 'not calibrated';
        }
      }
    } catch (e) {
      console.error('frame error', e);
    }
    setTimeout(sendFrame, 300);
  }
  sendFrame();

  calibrateBtn.onclick = async () => {
    // client-side capture for open and closed phases
    const capturePhase = async (label, ms) => {
      alert(`Calibration: keep your ${label} for ${ms/1000} seconds. Click OK to start.`);
      const imgs = [];
      const start = performance.now();
      while (performance.now() - start < ms) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        imgs.push(canvas.toDataURL('image/jpeg', 0.7));
        await new Promise(r => setTimeout(r, 120));
      }
      return imgs;
    };

    const openImgs = await capturePhase('eyes open', 3000);
    const closedImgs = await capturePhase('eyes closed (gently)', 3000);

    // send batches to server
    const res = await fetch('/calibrate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ open: openImgs, closed: closedImgs }) });
    const j = await res.json();
    if (j && j.status === 'ok') alert('Calibration saved.');
  };

  resetCalib.onclick = async () => {
    await fetch('/reset_calib', { method:'POST' });
    alert('Calibration reset.');
  };

  // Audio recorder
  let recorder = null;
  startRec.onclick = () => {
    startRec.disabled = true; stopRec.disabled = false;
    recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    recorder.ondataavailable = async e => {
      if (e.data && e.data.size > 0) {
        const arr = await e.data.arrayBuffer();
        const res = await fetch('/audio', { method: 'POST', body: arr });
        const j = await res.json();
        if (j) {
          fatigueScoreEl.textContent = j.fatigue_score !== undefined ? j.fatigue_score.toFixed(3) : '—';
          audioMetricsEl.textContent = JSON.stringify(j.metrics || j, null, 2);
        }
      }
    };
    recorder.start(4000);
  };
  stopRec.onclick = () => {
    startRec.disabled = false; stopRec.disabled = true;
    recorder?.stop();
  };
})();
</script>
</body>
</html>
"""

# ------------------ Helper: sessions ------------------
def get_session():
    sid = request.cookies.get('sid')
    if not sid:
        sid = str(uuid.uuid4())
    return sid

# ------------------ Calibration endpoint ------------------
@app.route('/calibrate', methods=['POST'])
def calibrate():
    sid = get_session()
    s = sessions[sid]
    data = request.get_json(force=True)
    open_imgs = data.get('open', [])
    closed_imgs = data.get('closed', [])

    def imgs_to_ears(img_list):
        ears = []
        for data_url in img_list:
            try:
                header, b64 = data_url.split(',',1)
            except Exception:
                continue
            arr = np.frombuffer(base64.b64decode(b64), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            ear = compute_ear_from_image(img)
            if ear is not None:
                ears.append(ear)
        return ears

    open_ears = imgs_to_ears(open_imgs)
    closed_ears = imgs_to_ears(closed_imgs)

    if len(open_ears) < 5 or len(closed_ears) < 5:
        return jsonify({'status':'error','message':'Not enough valid frames for calibration. Ensure face is visible.'}), 400

    open_mean = float(np.median(open_ears))
    closed_mean = float(np.median(closed_ears))
    # set threshold halfway (weighted) between open and closed
    new_threshold = float((open_mean + closed_mean) / 2.0)
    s['threshold'] = new_threshold
    s['calib_open_vals'] = open_ears
    s['calib_closed_vals'] = closed_ears
    s['calibrating'] = False
    return jsonify({'status':'ok','threshold':new_threshold})

@app.route('/reset_calib', methods=['POST'])
def reset_calib():
    sid = get_session()
    s = sessions[sid]
    s['threshold'] = 0.22
    s['calib_open_vals'] = []
    s['calib_closed_vals'] = []
    return jsonify({'status':'ok'})

# ------------------ Frame analysis ------------------
def compute_ear_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if DLIB_AVAILABLE:
        rects = detector(gray, 0)
        if len(rects) == 0:
            return None
        # pick largest face
        rect = max(rects, key=lambda r: r.width() * r.height())
        shape = predictor(gray, rect)
        coords = shape_to_np(shape)
        leftEye = coords[LEFT_EYE_POINTS]
        rightEye = coords[RIGHT_EYE_POINTS]
        le = eye_aspect_ratio(leftEye.astype('float'))
        re = eye_aspect_ratio(rightEye.astype('float'))
        return float((le + re) / 2.0)
    else:
        # fallback: Haar cascade detection of eyes
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        if len(eyes) < 2:
            return None
        ratios = []
        for (ex,ey,ew,eh) in eyes[:2]:
            ratios.append(eh / float(ew))
        return float(np.mean(ratios))

@app.route('/frame', methods=['POST'])
def frame():
    sid = get_session()
    s = sessions[sid]
    data = request.get_json(force=True)
    img_b64 = data.get('image','').split(',',1)[1] if data.get('image') else None
    if not img_b64:
        return jsonify({'error':'no_image'}), 400
    arr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    ear = compute_ear_from_image(img)
    if ear is not None:
        s['ear_buffer'].append(ear)
        # smoothed ear (median)
        ear_smoothed = float(np.median(list(s['ear_buffer'])))
        # threshold-based blink detection with hysteresis
        if ear_smoothed < s['threshold']:
            s['consec_closed'] += 1
        else:
            if s['consec_closed'] >= EAR_CONSEC_FRAMES:
                s['blinks'].append(datetime.utcnow())
            s['consec_closed'] = 0
    else:
        ear_smoothed = None

    # trim blinks older than 60s
    cutoff = datetime.utcnow() - timedelta(seconds=60)
    while s['blinks'] and s['blinks'][0] < cutoff:
        s['blinks'].popleft()
    blink_count = len(s['blinks'])

    alert = None
    if blink_count < 6:
        alert = 'Low blink rate — possible dryness or prolonged staring.'
    elif blink_count > 30:
        alert = 'High blink rate — possible irritation or discomfort.'

    return jsonify({'ear': ear, 'ear_smoothed': ear_smoothed, 'blink_count': blink_count, 'blink_rate_min': blink_count, 'alert': alert, 'calibrated': bool(s.get('calib_open_vals'))})

# ------------------ Audio processing (improved) ------------------
def extract_voiced_pcm(audio_bytes):
    # audio_bytes come from MediaRecorder blobs (webm). Convert to 16kHz mono pcm16 via pydub
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    # target parameters for webrtcvad
    target_sr = 16000
    audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
    raw_bytes = audio.raw_data
    sample_rate = target_sr
    sample_width = 2

    vad = webrtcvad.Vad(2)
    frame_duration_ms = 30
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * sample_width)

    voiced_bytes = bytearray()
    for offset in range(0, len(raw_bytes), frame_size):
        frame = raw_bytes[offset:offset+frame_size]
        if len(frame) < frame_size:
            break
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech:
            voiced_bytes.extend(frame)

    if len(voiced_bytes) < 16000 * 0.5 * sample_width:
        # less than 0.5s voiced -> return None
        return None, sample_rate

    # return voiced PCM bytes and sample rate
    return bytes(voiced_bytes), sample_rate

def analyze_audio_bytes(blob_bytes):
    # extract voiced-only PCM using VAD
    voiced_pcm, sr = extract_voiced_pcm(blob_bytes)
    if voiced_pcm is None:
        return {'fatigue_score': 0.0, 'message': 'No voiced speech detected (try speaking louder/closer).', 'metrics': {}}

    # write voiced pcm to temp wav for librosa
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
        # AudioSegment to write header + PCM
        seg = AudioSegment(data=voiced_pcm, sample_width=2, frame_rate=sr, channels=1)
        seg.export(tf.name, format='wav')
        tmp_wav = tf.name

    y, sr = librosa.load(tmp_wav, sr=sr)
    if len(y) < 100:
        return {'fatigue_score': 0.0, 'message': 'Voiced segment too short', 'metrics': {}}

    # basic acoustic features
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    std_rms = float(np.std(rms))

    # pitch (yin)
    try:
        f0 = librosa.yin(y, fmin=75, fmax=600, sr=sr)
        f0_voiced = f0[~np.isnan(f0)]
        median_f0 = float(np.median(f0_voiced)) if len(f0_voiced) else 0.0
        f0_std = float(np.std(f0_voiced)) if len(f0_voiced) else 0.0
        jitter = float(np.mean(np.abs(np.diff(f0_voiced))) / (median_f0 + 1e-6)) if len(f0_voiced) > 1 else 0.0
    except Exception:
        median_f0, f0_std, jitter = 0.0, 0.0, 0.0

    # shimmer (amplitude perturbation): successive RMS diffs
    if len(rms) > 1:
        shimmer = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9))
    else:
        shimmer = 0.0

    # HNR approximation using harmonic/percussive separation
    try:
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = 10.0 * math.log10((np.sum(harmonic**2) + 1e-9) / (np.sum(percussive**2) + 1e-9))
    except Exception:
        hnr = 0.0

    # spectral flatness (lower -> more tonal)
    try:
        sf = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    except Exception:
        sf = 1.0

    # Build normalized metrics (heuristic scaling)
    norm_energy = 1.0 - np.tanh(mean_rms * 80.0)  # lower energy -> increase fatigue
    norm_jitter = np.tanh(jitter * 8.0)
    norm_shimmer = np.tanh(shimmer * 8.0)
    # hnr: higher is better -> fatigue contribution = 1 - scaled_hnr
    norm_hnr = 1.0 - (1.0 / (1.0 + np.tanh((hnr - 10.0) / 10.0)))

    # combine weights (tuned heuristically)
    fatigue_score = float(np.clip(0.45 * norm_energy + 0.25 * norm_jitter + 0.15 * norm_shimmer + 0.15 * norm_hnr, 0.0, 1.0))

    # message
    if fatigue_score > 0.7:
        message = 'High voice fatigue likelihood — rest your voice and hydrate.'
    elif fatigue_score > 0.45:
        message = 'Moderate signs of voice fatigue.'
    else:
        message = 'No strong signs of fatigue detected.'

    metrics = {
        'mean_rms': mean_rms,
        'std_rms': std_rms,
        'median_f0': median_f0,
        'f0_std': f0_std,
        'jitter': jitter,
        'shimmer': shimmer,
        'hnr': hnr,
        'spectral_flatness': sf,
        'voiced_duration_s': len(y)/sr
    }

    return {'fatigue_score': fatigue_score, 'message': message, 'metrics': metrics}

@app.route('/audio', methods=['POST'])
def audio():
    try:
        data = request.data
        if not data:
            return jsonify({'error':'no_audio'}), 400
        out = analyze_audio_bytes(data)
        return jsonify(out)
    except Exception as e:
        return jsonify({'error':'exception','message': str(e)}), 500

# ------------------ Routes ------------------
@app.route('/')
def index():
    resp = make_response(render_template_string(INDEX_HTML))
    if not request.cookies.get('sid'):
        resp.set_cookie('sid', str(uuid.uuid4()))
    return resp

# ------------------ Run ------------------
if __name__ == '__main__':
    print('Starting improved Blink Voice Tracker...')
    app.run(host='0.0.0.0', port=5000, debug=True)
