"""
Baby Soother v3  —  Python 3.13 · Flask · gTTS + librosa voice adaptation · ML cry detection
=============================================================================================
Run:   python app.py
Open:  http://localhost:5050
"""

import os, io, time, json, uuid, base64, logging, threading, shutil, struct, wave
from pathlib import Path
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

# ── Local modules ─────────────────────────────────────────────────────────────
import cry_ml
import voice_engine

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("app")

# ── Dirs ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
SAMPLES   = BASE / "samples"
GENERATED = BASE / "generated"
STATIC_A  = BASE / "static" / "audio"
for d in [SAMPLES, GENERATED, STATIC_A]: d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024

# ── App state ─────────────────────────────────────────────────────────────────
class S:
    samples:    list[str]  = []        # WAV paths
    queue:      list[dict] = []        # generated phrase items
    fingerprint: dict      = {}        # voice analysis result
    is_soothing: bool      = False
    fade_active: bool      = False
    fade_start:  float     = 0.0
    fade_dur:    float     = 10.0
    current_vol: float     = 1.0
    lock = threading.Lock()

# ── Default soothing phrases ──────────────────────────────────────────────────
PHRASES = [
    "Shh, it's okay sweetheart. Mama is right here with you.",
    "You are safe and loved, my little one. Everything is alright.",
    "Hush now, precious baby. Close your eyes and rest.",
    "Sweet dreams, little angel. Mama's got you.",
    "There there, my darling. You are warm, safe, and loved.",
    "Rest now baby. The night is peaceful and all is well.",
    "Shh, mama loves you so much. Sleep now, sweet one.",
    "Everything is okay. You are held and you are loved.",
    "Breathe easy, little one. Mama is watching over you.",
    "Shhh. Close your eyes, my precious baby. All is calm.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "ok": True,
        "samples":       len(S.samples),
        "queue":         len(S.queue),
        "fingerprint":   S.fingerprint,
        "is_soothing":   S.is_soothing,
        "fade_active":   S.fade_active,
        "volume":        round(S.current_vol * 100),
    })


# ── Samples ───────────────────────────────────────────────────────────────────
@app.route("/api/samples/upload", methods=["POST"])
def upload_sample():
    try:
        data = request.data or (request.files.get("audio") and request.files["audio"].read())
        if not data:
            return jsonify({"ok": False, "error": "No audio data"}), 400

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = str(SAMPLES / f"sample_{ts}.webm")
        wav_path = str(SAMPLES / f"sample_{ts}.wav")

        with open(raw_path, "wb") as f:
            f.write(data)

        # Convert webm → wav with pydub if available, else keep raw
        converted = False
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(raw_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format="wav")
            os.remove(raw_path)
            final = wav_path
            converted = True
        except Exception:
            final = raw_path   # serve the webm directly

        with S.lock:
            S.samples.append(final)

        # Re-analyze voice on background thread
        threading.Thread(target=_reanalyze, daemon=True).start()

        return jsonify({
            "ok": True,
            "count":     len(S.samples),
            "name":      Path(final).name,
            "url":       f"/api/samples/play/{len(S.samples)-1}",
            "converted": converted,
        })
    except Exception as e:
        log.error(f"Upload error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


def _reanalyze():
    if S.samples:
        fp = voice_engine.analyze_voice(S.samples)
        with S.lock:
            S.fingerprint = fp
        log.info(f"Voice fingerprint updated: {fp}")


@app.route("/api/samples/play/<int:idx>")
def play_sample(idx):
    """Stream a recorded sample back to the browser."""
    if idx < 0 or idx >= len(S.samples):
        return jsonify({"error": "not found"}), 404
    path = Path(S.samples[idx])
    return send_from_directory(path.parent, path.name)


@app.route("/api/samples/list")
def list_samples():
    items = [{"idx": i, "name": Path(p).name, "url": f"/api/samples/play/{i}"}
             for i, p in enumerate(S.samples)]
    return jsonify({"ok": True, "samples": items, "fingerprint": S.fingerprint})


@app.route("/api/samples/clear", methods=["POST"])
def clear_samples():
    with S.lock:
        S.samples.clear()
        S.fingerprint = {}
    return jsonify({"ok": True})


# ── Phrase generation ─────────────────────────────────────────────────────────
@app.route("/api/generate", methods=["POST"])
def api_generate():
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"ok": False, "error": "No text"}), 400
    return _do_generate(text)


@app.route("/api/generate/defaults", methods=["POST"])
def api_generate_defaults():
    results = []
    for phrase in PHRASES:
        r = _do_generate(phrase)
        results.append(r.get_json())
    return jsonify({"ok": True, "results": results})


def _do_generate(text: str):
    pid      = str(uuid.uuid4())[:8]
    out_path = str(GENERATED / f"phrase_{pid}.wav")

    fp = S.fingerprint or {}
    ok = voice_engine.generate_adapted_phrase(text, out_path, fp)

    # Resolve actual file (might be mp3 if wav conversion failed)
    actual = out_path
    if not Path(out_path).exists():
        for ext in [".mp3", ".webm"]:
            alt = out_path.replace(".wav", ext)
            if Path(alt).exists():
                actual = alt
                break

    if not Path(actual).exists():
        return jsonify({"ok": False, "error": "Generation failed"})

    # Copy to static/audio for browser
    ext         = Path(actual).suffix
    static_name = f"phrase_{pid}{ext}"
    static_path = STATIC_A / static_name
    shutil.copy(actual, static_path)
    url = f"/static/audio/{static_name}"

    item = {"id": pid, "text": text, "path": actual, "url": url}
    with S.lock:
        S.queue.append(item)

    return jsonify({"ok": True, "item": item})


@app.route("/api/queue")
def api_queue():
    q = [{"id": i["id"], "text": i["text"], "url": i["url"]} for i in S.queue]
    return jsonify({"ok": True, "queue": q})


@app.route("/api/queue/clear", methods=["POST"])
def api_clear_queue():
    with S.lock:
        S.queue.clear()
    return jsonify({"ok": True})


# ── ML Cry detection ──────────────────────────────────────────────────────────
@app.route("/api/cry/analyze", methods=["POST"])
def api_cry_analyze():
    """
    Receive a base64-encoded PCM float32 audio chunk from the browser,
    run ML cry detection, return result.

    Body: { "pcm_b64": "...", "sr": 16000 }
    """
    try:
        body   = request.get_json(silent=True) or {}
        pcm_b64 = body.get("pcm_b64", "")
        sr      = int(body.get("sr", 16000))

        if not pcm_b64:
            return jsonify({"ok": False, "error": "No audio data"}), 400

        # Decode Float32 PCM
        raw  = base64.b64decode(pcm_b64)
        arr  = np.frombuffer(raw, dtype=np.float32).copy()

        result = cry_ml.predict(arr, sr)
        return jsonify({"ok": True, **result})

    except Exception as e:
        log.error(f"Cry analysis error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/cry/event", methods=["POST"])
def api_cry_event():
    body   = request.get_json(silent=True) or {}
    crying = body.get("crying", False)
    if crying:
        S.fade_active = False
        S.current_vol = 1.0
        S.is_soothing = True
    else:
        if S.is_soothing and not S.fade_active:
            S.fade_active = True
            S.fade_start  = time.time()
    return jsonify({"ok": True})


@app.route("/api/soothe/stop", methods=["POST"])
def api_stop():
    S.is_soothing = False
    S.fade_active = False
    S.current_vol = 1.0
    return jsonify({"ok": True})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    log.info(f"🌙 Baby Soother v3  →  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
