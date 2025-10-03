import io, base64, uuid, json, tempfile
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, request, make_response, render_template_string
import pyttsx3
from pydub import AudioSegment
import librosa

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

sessions = defaultdict(lambda: {
    "blinks": deque(maxlen=500),
    "consec_closed": 0,
    "threshold": None,
    "calibrating": False,
    "calib_samples": [],
    "eye_ratios": deque(maxlen=300),
    "timestamps": deque(maxlen=300)
})

tts_engine = pyttsx3.init()

INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>Eye Blink & Voice Tracker</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <style>
        .fade-in {
            animation: fadeIn 0.6s ease-in;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
    </style>
</head>
<body class="bg-gray-900 text-gray-200 min-h-screen font-sans">
<div class="container mx-auto p-6">
    <h1 class="text-4xl font-bold text-center mb-6 text-gradient bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">Eye Blink & Voice Fatigue Tracker</h1>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        <!-- Eye Tracker -->
        <div class="bg-gray-800 p-6 rounded-3xl shadow-xl transform hover:scale-105 transition duration-300 ease-in-out fade-in">
            <h2 class="text-2xl font-semibold mb-4">👁️ Eye Health Tracker</h2>
            <video id="video" class="rounded-lg w-full mb-4 border-4 border-gray-700 shadow-lg" autoplay muted playsinline></video>
            <canvas id="canvas" class="hidden"></canvas>
            <div class="flex justify-between mb-2">
                <div>Blinks/min: <span id="blinkRate" class="font-bold">—</span></div>
                <div>Total (1 min): <span id="blinkCount" class="font-bold">—</span></div>
            </div>
            <p id="blinkAlert" class="text-red-400 font-bold mb-2 transition duration-500">—</p>
            <div id="blinkChart" class="w-full h-64 mb-4 rounded-lg"></div>
            <button id="calibrateBtn" class="bg-blue-500 hover:bg-blue-600 px-5 py-2 rounded-xl shadow-lg transition ease-in-out duration-300">Calibrate Blink 👁️</button>
        </div>

        <!-- Voice Tracker -->
        <div class="bg-gray-800 p-6 rounded-3xl shadow-xl transform hover:scale-105 transition duration-300 ease-in-out fade-in">
            <h2 class="text-2xl font-semibold mb-4">🎤 Voice Health Tracker</h2>
            <p class="mb-2">Fatigue Score: <span id="fatigueScore" class="font-bold">—</span></p>
            <p id="voiceMsg" class="mb-4 text-lg">—</p>
            <div class="flex space-x-2 mb-4">
                <button id="startRec" class="bg-green-500 hover:bg-green-600 px-5 py-2 rounded-xl shadow-lg transition ease-in-out duration-300">Start Recording 🎙️</button>
                <button id="stopRec" class="bg-red-500 hover:bg-red-600 px-5 py-2 rounded-xl shadow-lg transition ease-in-out duration-300" disabled>Stop Recording ⏹️</button>
            </div>
            <div id="voiceChart" class="w-full h-64 rounded-lg"></div>
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
            Plotly.newPlot('blinkChart', [{
                x: j.timestamps,
                y: j.eye_ratios,
                type: 'scatter',
                line: {color: '#38bdf8'}
            }], {margin:{t:0}});
        }
        setTimeout(sendFrame, 400);
    }
    sendFrame();

    calibrateBtn.onclick = async () => {
        await fetch('/calibrate', { method: 'POST' });
        calibrateBtn.innerText = "Calibrating...";
        calibrateBtn.disabled = true;
        setTimeout(() => {
            calibrateBtn.innerText = "Calibrate Blink 👁️";
            calibrateBtn.disabled = false;
        }, 5000);
    };

    let recorder = null;
    startRecBtn.onclick = () => {
        startRecBtn.disabled = true;
        stopRecBtn.disabled = false;
        recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        let chunks = [];
        recorder.ondataavailable = e => chunks.push(e.data);
        recorder.onstop = async () => {
            const blob = new Blob(chunks, { type: "audio/webm" });
            const buf = await blob.arrayBuffer();
            const res = await fetch('/audio', { method: 'POST', body: buf });
            const j = await res.json();
            fatigueScoreEl.textContent = j.fatigue_score?.toFixed(2) || '—';
            voiceMsgEl.textContent = j.message || '—';
            Plotly.newPlot('voiceChart', [{
                x: j.timestamps || [],
                y: j.scores || [],
                type: 'scatter',
                line: {color: '#f43f5e'}
            }], {margin:{t:0}});
        };
        recorder.start();
    };
    stopRecBtn.onclick = () => {
        startRecBtn.disabled = false;
        stopRecBtn.disabled = true;
        recorder?.stop();
    };
})();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    resp = make_response(render_template_string(INDEX_HTML))
    if not request.cookies.get("sid"):
        resp.set_cookie("sid", str(uuid.uuid4()))
    return resp

def get_session():
    sid = request.cookies.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
    return sid

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
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    eye_ratio = None
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = [landmarks[i] for i in [33, 133]]
        right_eye = [landmarks[i] for i in [362, 263]]
        left_ratio = np.linalg.norm(
            np.array([left_eye[0].x, left_eye[0].y]) - np.array([left_eye[1].x, left_eye[1].y]))
        right_ratio = np.linalg.norm(
            np.array([right_eye[0].x, right_eye[0].y]) - np.array([right_eye[1].x, right_eye[1].y]))
        eye_ratio = (left_ratio + right_ratio) / 2.0
        s["eye_ratios"].append(eye_ratio)
        s["timestamps"].append(datetime.utcnow().strftime("%H:%M:%S"))
        if s["calibrating"]:
            s["calib_samples"].append(eye_ratio)
            if len(s["calib_samples"]) > 50:
                s["threshold"] = np.mean(s["calib_samples"]) * 0.75
                s["calibrating"] = False
        if s["threshold"] and eye_ratio < s["threshold"]:
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
    return {
        "eye_ratio": eye_ratio,
        "blink_count": blink_count,
        "blink_rate_min": blink_rate,
        "alert": alert,
        "eye_ratios": list(s["eye_ratios"]),
        "timestamps": list(s["timestamps"])
    }

def analyze_audio_bytes(blob_bytes):
    audio = AudioSegment.from_file(io.BytesIO(blob_bytes))
    audio = audio.set_frame_rate(22050).set_channels(1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        audio.export(tf.name, format="wav")
        y, sr = librosa.load(tf.name, sr=22050)
    if len(y) < 200: return {"fatigue_score": 0, "message": "Audio too short"}
    f0 = librosa.yin(y, fmin=80, fmax=800, sr=sr)
    f0 = f0[~np.isnan(f0)]
    median_f0 = np.median(f0) if len(f0) else 0
    jitter = np.mean(np.abs(np.diff(f0))) / (median_f0 + 1e-6) if len(f0) > 1 else 0
    rms = librosa.feature.rms(y=y)[0]
    norm_rms = 1 - np.tanh(np.mean(rms) * 50)
    norm_jitter = np.tanh(jitter * 10)
    score = float(np.clip(0.7 * norm_rms + 0.3 * norm_jitter, 0, 1))
    msg = "Moderate fatigue" if score > 0.45 else "Low fatigue"
    if score > 0.7: msg = "High fatigue"

    tts_engine.say(f"Your fatigue score is {score:.2f}. {msg}")
    tts_engine.runAndWait()

    return {"fatigue_score": score, "message": msg, "timestamps": [datetime.utcnow().strftime("%H:%M:%S")], "scores": [score]}

@app.route("/audio", methods=["POST"])
def audio():
    return analyze_audio_bytes(request.data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
