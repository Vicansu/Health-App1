# app.py
import cv2
import numpy as np
import librosa
import tempfile
import uuid
import time
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Haar cascade files (OpenCV built-in)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg"}

# Blink stats storage
blink_data = {
    "blinks": 0,
    "start_time": time.time(),
    "last_blink": None,
    "trend": []
}


# -------- Voice Fatigue Checker --------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_fatigue(filepath):
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    pitch = float(np.mean(librosa.yin(y, fmin=50, fmax=300, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    score = (rms * 10) + (100 - min(pitch, 300) / 3) - (duration / 10) + (zcr * 100)
    if score < 30:
        fatigue_level = "Low"
        color = "#2ecc71"
    elif score < 60:
        fatigue_level = "Moderate"
        color = "#f1c40f"
    else:
        fatigue_level = "High"
        color = "#e74c3c"

    return {
        "duration_sec": round(duration, 2),
        "avg_rms": round(rms, 4),
        "avg_pitch": round(pitch, 2),
        "zero_crossing_rate": round(zcr, 4),
        "fatigue_score": round(score, 2),
        "fatigue_level": fatigue_level,
        "fatigue_color": color
    }


# -------- Blink Detection (Haar Cascade) --------
def detect_blink(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    blink_detected = False
    eyes_open = 0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eyes_open = len(eyes)
        if eyes_open == 0:
            blink_detected = True

    return blink_detected, eyes_open


# -------- Routes --------
@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <title>AI Wellness Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Arial, sans-serif; background:#f0f2f5; color:#333; text-align:center; margin:0; padding:20px; }
    h1 { color:#2c3e50; margin-bottom:20px; }
    .container { display:flex; flex-wrap:wrap; justify-content:center; }
    .card { flex:1 1 40%; background:white; padding:20px; margin:15px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.1); max-width:500px; }
    button { padding:10px 15px; background:#3498db; color:white; border:none; border-radius:8px; cursor:pointer; margin-top:10px; }
    button:hover { background:#2980b9; }
    pre { text-align:left; background:#ecf0f1; padding:10px; border-radius:8px; height:120px; overflow:auto; font-size:14px; }
    video { border-radius:12px; border:2px solid #ccc; max-width:100%; }
    canvas { margin-top:10px; max-width:100%; }
    .level { font-weight:bold; font-size:18px; padding:5px 10px; border-radius:8px; display:inline-block; margin-top:10px; }
  </style>
</head>
<body>
  <h1>AI Wellness Dashboard</h1>
  <div class="container">

    <!-- Voice -->
    <div class="card">
      <h2>Voice Fatigue Checker</h2>
      <input type="file" id="voiceFile" accept="audio/*">
      <button onclick="uploadVoice()">Analyze Voice</button>
      <div id="voiceLevel" class="level">Awaiting analysis...</div>
      <pre id="voiceOutput">Upload an audio file to see results...</pre>
      <canvas id="voiceChart" height="200"></canvas>
    </div>

    <!-- Eye -->
    <div class="card">
      <h2>Eye Blink Tracker</h2>
      <video id="video" autoplay playsinline></video>
      <pre id="blinkOutput">Initializing camera...</pre>
      <canvas id="blinkChart" height="200"></canvas>
    </div>

  </div>

<script>
    // ---- VOICE ----
    let voiceChart;
    function uploadVoice() {
      const file = document.getElementById("voiceFile").files[0];
      if (!file) return alert("Select audio file");
      const formData = new FormData();
      formData.append("file", file);
      fetch("/upload", { method: "POST", body: formData })
        .then(r => r.json())
        .then(data => {
          document.getElementById("voiceOutput").textContent = JSON.stringify(data, null, 2);
          document.getElementById("voiceLevel").textContent = "Fatigue Level: " + data.fatigue_level;
          document.getElementById("voiceLevel").style.background = data.fatigue_color;
          document.getElementById("voiceLevel").style.color = "white";

          const ctx = document.getElementById("voiceChart").getContext("2d");
          if (voiceChart) voiceChart.destroy();
          voiceChart = new Chart(ctx, {
            type: 'bar',
            data: {
              labels: ["RMS", "Pitch", "ZCR", "Score"],
              datasets: [{
                label: "Voice Features",
                data: [data.avg_rms, data.avg_pitch, data.zero_crossing_rate, data.fatigue_score],
                backgroundColor: ["#3498db","#2ecc71","#f1c40f", data.fatigue_color]
              }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
          });
        });
    }

    // ---- BLINK ----
    let blinkChart;
    let blinkTimes = [];
    let blinkRates = [];

    const video = document.getElementById("video");
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
      video.srcObject = stream;
    });

    setInterval(() => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);
      canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");
        fetch("/blink", { method: "POST", body: formData })
          .then(r => r.json())
          .then(data => {
            document.getElementById("blinkOutput").textContent = JSON.stringify(data, null, 2);

            blinkTimes.push(data.elapsed_minutes);
            blinkRates.push(data.blink_rate_per_min);

            const ctxBlink = document.getElementById("blinkChart").getContext("2d");
            if (blinkChart) blinkChart.destroy();
            blinkChart = new Chart(ctxBlink, {
              type: 'line',
              data: {
                labels: blinkTimes,
                datasets: [{
                  label: "Blink Rate (per min)",
                  data: blinkRates,
                  borderColor: "#e67e22",
                  backgroundColor: "rgba(230, 126, 34, 0.2)",
                  fill: true,
                  tension: 0.2
                }]
              },
              options: { responsive: true, scales: { x: { title: { display: true, text: "Minutes" } }, y: { beginAtZero: true } } }
            });
          });
      }, "image/jpeg");
    }, 2000);
</script>
</body>
</html>
    """)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"})
    file = request.files["file"]
    if file and allowed_file(file.filename):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
            tmp.write(file.read())
            tmp.flush()
            return jsonify(analyze_fatigue(tmp.name))
    return jsonify({"error": "Invalid file type"})


@app.route("/blink", methods=["POST"])
def blink():
    global blink_data
    file = request.files["frame"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    blink_detected, eyes_open = detect_blink(frame)

    if blink_detected:
        if blink_data["last_blink"] is None or (time.time() - blink_data["last_blink"]) > 0.3:
            blink_data["blinks"] += 1
            blink_data["last_blink"] = time.time()

    elapsed_min = (time.time() - blink_data["start_time"]) / 60
    blink_rate = blink_data["blinks"] / elapsed_min if elapsed_min > 0 else 0
    blink_data["trend"].append({"time": elapsed_min, "rate": blink_rate})

    return jsonify({
        "blink_detected": blink_detected,
        "eyes_open": eyes_open,
        "total_blinks": blink_data["blinks"],
        "blink_rate_per_min": round(blink_rate, 2),
        "elapsed_minutes": round(elapsed_min, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
