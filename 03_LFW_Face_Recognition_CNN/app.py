"""
Flask Application — Real-Time Face Recognition
Streams live webcam with bounding-box overlays and exposes an /add_face
page for registering new identities on the fly.
"""

import os
import time
import base64
import threading

import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

from recognizer import FaceRecognizer

# ---------------------------------------------------------------------------
# App & global state
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

recognizer = FaceRecognizer(
    known_faces_dir=os.path.join(BASE_DIR, "known_faces"),
    embeddings_file=os.path.join(BASE_DIR, "embeddings.pkl"),
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Camera management --------------------------------------------------------

_camera = None
_camera_lock = threading.Lock()
_stream_active = threading.Event()   # controls the generate_frames loop
_stream_active.set()


def _get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(0)
        _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        _camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return _camera


def _release_camera():
    """Stop the stream, background recognizer, and release the hardware."""
    global _camera
    _stream_active.clear()          # signal generator to exit
    bg_recognizer.stop()
    time.sleep(0.15)                # let generator loop finish
    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None


# ---------------------------------------------------------------------------
# Background recognition thread
# ---------------------------------------------------------------------------

class BackgroundRecognizer:
    """Runs face detection + DeepFace recognition in a separate daemon thread
    so the video stream is never blocked."""

    def __init__(self):
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._results: list[dict] = []
        self._results_lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def feed_frame(self, frame):
        """Called by the generator to supply the newest frame (non-blocking)."""
        with self._frame_lock:
            self._latest_frame = frame.copy()

    @property
    def results(self) -> list[dict]:
        with self._results_lock:
            return list(self._results)

    def _loop(self):
        """Continuously process the latest frame for recognition."""
        while self._running:
            # Grab the most recent frame
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None   # consume it

            if frame is None:
                time.sleep(0.05)
                continue

            # Detect faces (Haar — fast)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
            )

            new_results = []
            for (x, y, w, h) in faces:
                # Pad crop for better embedding quality
                pad = int(0.1 * max(w, h))
                y1 = max(0, y - pad)
                y2 = min(frame.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(frame.shape[1], x + w + pad)
                face_crop = frame[y1:y2, x1:x2].copy()
                name = recognizer.recognize_face(face_crop)
                new_results.append({"bbox": (x, y, w, h), "name": name})

            with self._results_lock:
                self._results = new_results

            # Small sleep to avoid burning CPU between passes
            time.sleep(0.1)


bg_recognizer = BackgroundRecognizer()


# ---------------------------------------------------------------------------
# Video streaming generator
# ---------------------------------------------------------------------------

def _draw_label(frame, text, x, y, w, color):
    """Draw a filled label rectangle above the bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.65, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 6), font, scale, (255, 255, 255), thickness)


def _center(bbox):
    """Return the center (cx, cy) of a bounding box (x, y, w, h)."""
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def _match_label(bbox, cached_results, max_dist=100):
    """Find the cached recognition result closest to *bbox* by center distance."""
    if not cached_results:
        return None
    cx, cy = _center(bbox)
    best, best_d = None, float("inf")
    for res in cached_results:
        rcx, rcy = _center(res["bbox"])
        d = abs(cx - rcx) + abs(cy - rcy)  # Manhattan distance
        if d < best_d:
            best_d, best = d, res
    return best if best_d < max_dist else None


def generate_frames():
    """Yield JPEG frames at full speed -- recognition happens in background."""
    global _camera
    _stream_active.set()
    bg_recognizer.start()
    frame_count = 0
    DETECT_SCALE = 0.5

    try:
        while _stream_active.is_set():
            try:
                with _camera_lock:
                    cam = _get_camera()
                    ok, frame = cam.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                frame_count += 1

                # Feed frame to background recognizer (every 5th frame)
                if frame_count % 5 == 0:
                    bg_recognizer.feed_frame(frame)

                # --- Detect faces (full resolution, safe params) ----------
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
                )

                # Get cached recognition labels (non-blocking)
                cached = bg_recognizer.results

                # --- Draw a box for every detected face -------------------
                for (x, y, w, h) in faces:
                    match = _match_label((x, y, w, h), cached)
                    if match:
                        name = match["name"]
                        if name == "subject_00":
                            color = (0, 0, 255)       # red -- unknown
                            label = "Unknown"
                        else:
                            color = (0, 230, 118)      # green -- recognised
                            label = name
                    else:
                        color = (180, 180, 180)         # gray -- pending
                        label = "Detecting..."
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    _draw_label(frame, label, x, y, w, color)

                # --- Encode & yield --------------------------------------
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                    )

            except GeneratorExit:
                # Client disconnected — exit cleanly
                break
            except Exception as exc:
                # Catch transient OpenCV / numpy errors — keep streaming
                print(f"[Stream] Frame error (ignored): {exc}")
                time.sleep(0.05)
                continue

    finally:
        # Release camera directly — do NOT call _release_camera() here
        # because it clears _stream_active which would kill the next stream.
        bg_recognizer.stop()
        with _camera_lock:
            if _camera is not None:
                _camera.release()
                _camera = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", page="home")


@app.route("/add_face")
def add_face_page():
    return render_template("index.html", page="add_face")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/release_camera", methods=["POST"])
def release_camera():
    """Free the server-side camera so getUserMedia can access it."""
    _release_camera()
    time.sleep(0.3)   # give OS time to fully release the device
    return jsonify({"success": True})


@app.route("/capture", methods=["POST"])
def capture():
    """Receive a base64 snapshot from the browser and register a new face."""
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    image_data = data.get("image", "")

    if not name:
        return jsonify({"success": False, "message": "Please enter a name."})
    if not image_data:
        return jsonify({"success": False, "message": "No image data received."})

    try:
        # Decode base64 → OpenCV image
        _, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "message": "Invalid image data."})

        # Persist the image
        person_dir = os.path.join(recognizer.known_faces_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        img_path = os.path.join(person_dir, f"img_{ts}.jpg")
        cv2.imwrite(img_path, img)

        # Generate & store embedding
        ok = recognizer.add_embedding(name, img_path)
        if ok:
            return jsonify({"success": True, "message": f'Face registered for "{name}"!'})
        return jsonify({"success": False, "message": "Could not generate embedding."})

    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)})


@app.route("/rebuild", methods=["POST"])
def rebuild():
    """Rebuild all embeddings from the known_faces directory."""
    try:
        recognizer.rebuild_embeddings()
        return jsonify({"success": True, "message": "Embeddings rebuilt."})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  Face Recognition Server starting...")
    print("  Home page   -> http://localhost:5000")
    print("  Add face    -> http://localhost:5000/add_face\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
