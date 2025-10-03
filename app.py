import io
import base64
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque

import numpy as np
import cv2
import dlib
from flask import Flask, render_template_string, request, jsonify, make_response
import pyttsx3
from pydub import AudioSegment
import librosa, tempfile

# ------------------ Setup ------------------
app = Flask(__name__)

# ------------------ Face & Eye Detection ------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

# ------------------ State ------------------
sessions = defaultdict(lambda: {
    "blinks": deque(maxlen=500),
    "consec_closed": 0,
    "threshold": 0.25,
    "calibrating": False,
    "calib_samples": []
})

tts_engine = pyttsx3.init()

# ------------------ HTML UI ------------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <title>Eye Blink & Voice Fatigue Tracker</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex flex-col items-center p-6">
  <h1 class="text-3xl font-bold mb-6">Eye Blink & Voice Fatigue Tracker</h1>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-6xl">
    <!-- Eye Tracker -->
    <div class="bg-white shadow-lg rounded-2xl p-6">
      <h2 class="text-xl font-semibold mb-4">Eye Blink Tracker</h2>
      <video id="video" class="rounded-lg border mb-2" width="400" height="300" autoplay muted playsinline></video>
      <canvas id="canvas" width="400" height="300" class="hidden"></canvas>
      <div class="space-y-2">
        <p>Blinks/min: <span id="blinkRate">—</span></p>
        <p>Total (1 min): <span id="blinkCount">—</span></p>
        <p class="text-sm text-red-600" id="blinkAlert">—</p>
      </div>
      <button id="calibrateBtn" class="mt-3 bg-blue-500 text-white px-4 py-2 rounded-xl">Calibrate Blink</button>
    </div>

    <!-- Voice Tracker -->
    <div class="bg-white shadow-lg rounded-2xl p-6">
      <h2 class="text-xl font-semibold mb-4">Voice Fatigue Tracker</h2>
      <div class="space-y-2">
        <p>Fatigue Score: <span id="fatigueScore">—</span></p>
        <p id="voiceMsg">—</p>
      </div>
      <div class="mt-3 space-x-2">
        <button id="startRec" class="bg-green-500 text-white px-4 py-2 rounded-xl">Start</button>
        <button id="stopRec" class="bg-red-500 text-white px-4 py-2 rounded-xl" disabled>Stop</button>
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
  const blinkAlertEl = document.getElementById('blinkAlert');
  const calibrateBtn = document.getElementById('calibrateBtn');
  const fatigueScoreEl = document.getElementById('fatigueScore');
  const voiceMsgEl = document.getElementById('voiceMsg');
  const startRecBtn = document.getElementById('startRec');
  const stopRecBtn = document.getElementById('stopRec');

  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  video.srcObject = stream;

  async function sendFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
    const res = await fetch('/frame', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    });
    const j = await res.json();
    if (j.blink_rate_min !== undefined) {
      blinkRateEl.textContent = j.blink_rate_min.toFixed(1);
      blinkCountEl.textContent = j.blink_count;
      blinkAlertEl.textContent = j.alert || '—';
    }
    setTimeout(sendFrame, 300);
  }
  sendFrame();

  calibrateBtn.onclick = async () => {
    await fetch('/calibrate', { method: 'POST' });
    alert('Calibration started. Blink naturally for 5 seconds.');
  };

  let recorder = null;
  startRecBtn.onclick = () => {
    startRecBtn.disabled = true; stopRecBtn.disabled = false;
    recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    recorder.ondataavailable = async e => {
      const buf = await e.data.arrayBuffer();
      const res = await fetch('/audio', { method: 'POST', body: buf });
      const j = await res.json();
      fatigueScoreEl.textContent = j.fatigue_score?.toFixed(2) || '—';
      voiceMsgEl.textContent = j.message || '—';
    };
    recorder.start(4000);
  };
  stopRecBtn.onclick = () => {
    startRecBtn.disabled = false; stopRecBtn.disabled = true;
    recorder?.stop();
  };
})();
</script>
</body>
</html>
"""

# ------------------ Helpers ------------------
def get_session():
    sid = request.cookies.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
    return sid

# ------------------ Routes ------------------
@app.route("/")
def index():
    resp = make_response(render_template_string(INDEX_HTML))
    if not request.cookies.get("sid"):
        resp.set_cookie("sid", str(uuid.uuid4()))
    return resp

@app.route("/calibrate", methods=["POST"])
def calibrate():
    sid = get_session()
    s = sessions[sid]
    s["calibrating"] = True
    s["calib_samples"] = []
    return {"status": "ok"}

@app.route("/frame", methods=["POST"])
def frame():
    sid = get_session()
    s = sessions[sid]

    data = request.get_json()
    img_b64 = data['image'].split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)
    blink_detected = False
    eye_ratio = None

    if faces:
        face = faces[0]
        shape = predictor(gray, face)
        left_eye = [shape.part(i) for i in range(36, 42)]
        right_eye = [shape.part(i) for i in range(42, 48)]

        def eye_aspect_ratio(eye_points):
            a = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - np.array([eye_points[5].x, eye_points[5].y]))
            b = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - np.array([eye_points[4].x, eye_points[4].y]))
            c = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - np.array([eye_points[3].x, eye_points[3].y]))
            return (a + b) / (2.0 * c)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        eye_ratio = (left_ear + right_ear) / 2.0

        if s["calibrating"]:
            s["calib_samples"].append(eye_ratio)
            if len(s["calib_samples"]) > 50:
                s["threshold"] = 0.7 * np.mean(s["calib_samples"])
                s["calibrating"] = False

        if eye_ratio < s["threshold"]:
            s["consec_closed"] += 1
        else:
            if s["consec_closed"] >= 2:
                s["blinks"].append(datetime.utcnow())
            s["consec_closed"] = 0

    cutoff = datetime.utcnow() - timedelta(seconds=60)
    while s["blinks"] and s["blinks"][0] < cutoff:
        s["blinks"].popleft()
    blink_count = len(s["blinks"])
    blink_rate = blink_count

    alert = None
    if blink_rate < 6:
        alert = "Low blink rate — possible dryness."
    elif blink_rate > 30:
        alert = "High blink rate — possible irritation."

    return {"eye_ratio": eye_ratio, "blink_count": blink_count, "blink_rate_min": blink_rate, "alert": alert}

# ------------------ Audio ------------------
def analyze_audio_bytes(blob_bytes):
    audio = AudioSegment.from_file(io.BytesIO(blob_bytes))
    audio = audio.set_frame_rate(22050).set_channels(1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        audio.export(tf.name, format="wav")
        y, sr = librosa.load(tf.name, sr=22050)
    if len(y) < 200: return {"fatigue_score":0, "message":"Audio too short"}
    f0 = librosa.yin(y, fmin=80, fmax=800, sr=sr)
    f0 = f0[~np.isnan(f0)]
    median_f0 = np.median(f0) if len(f0) else 0
    jitter = np.mean(np.abs(np.diff(f0)))/(median_f0+1e-6) if len(f0) > 1 else 0
    fatigue_score = jitter * 100
    message = "Fatigue detected" if fatigue_score > 2 else "Normal"
    return {"fatigue_score": fatigue_score, "message": message}

@app.route("/audio", methods=["POST"])
def audio():
    blob = request.data
    result = analyze_audio_bytes(blob)
    if result["message"] != "Normal":
        tts_engine.say(result["message"])
        tts_engine.runAndWait()
    return jsonify(result)

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
