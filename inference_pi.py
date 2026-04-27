# ─────────────────────────────────────────────────────────────────
#  inference_pi.py  —  Live inference + FPS logging
#                      Pi 4B | Debian Trixie | rpicam stack
# ─────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import time
import sys
import json
import os
import threading
import subprocess
from datetime import datetime

try:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
    print("[Init] Using ai-edge-litert")
except ImportError:
    print("[Init] ai-edge-litert not found. Run: pip install ai-edge-litert")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH  = '/home/ananya/files/outputs/expression_model.tflite'
CLASSES     = ['angry', 'happy', 'sad', 'neutral']
IMG_SIZE    = 48
CONF_THRESH = 0.50
FRAME_W     = 640
FRAME_H     = 480
TARGET_FPS  = 10

USE_SERIAL  = False
SERIAL_PORT = '/dev/ttyS0'
SERIAL_BAUD = 9600

COLOURS = {
    'angry'  : (0,   0,   220),
    'happy'  : (0,   200, 0  ),
    'sad'    : (220, 80,  0  ),
    'neutral': (180, 180, 180),
}

# ── Log directory ─────────────────────────────────────────────────
LOG_DIR = '/home/ananya/files/logs'
os.makedirs(LOG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  Session logger — records every inference session
# ─────────────────────────────────────────────────────────────────

class SessionLogger:
    def __init__(self, notes: str = ''):
        self.start_time   = datetime.now()
        self.notes        = notes
        self.fps_readings = []
        self.predictions  = []   # (timestamp, label, confidence)
        self.face_counts  = []

    def log_frame(self, fps: float, faces: int,
                  label: str = '', conf: float = 0.0):
        self.fps_readings.append(fps)
        self.face_counts.append(faces)
        if label:
            self.predictions.append({
                'time'      : datetime.now().strftime('%H:%M:%S.%f')[:12],
                'label'     : label,
                'confidence': round(conf, 4),
            })

    def save(self):
        if not self.fps_readings:
            return

        fps_arr  = np.array(self.fps_readings)
        duration = (datetime.now() - self.start_time).seconds

        session = {
            'timestamp'    : self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'notes'        : self.notes,
            'model_path'   : MODEL_PATH,
            'duration_sec' : duration,
            'classes'      : CLASSES,
            'conf_threshold': CONF_THRESH,
            'fps': {
                'mean'   : round(float(fps_arr.mean()), 2),
                'min'    : round(float(fps_arr.min()),  2),
                'max'    : round(float(fps_arr.max()),  2),
                'std'    : round(float(fps_arr.std()),  2),
                'n_frames': len(self.fps_readings),
            },
            'faces': {
                'mean_per_frame': round(float(np.mean(self.face_counts)), 2),
                'frames_with_face': int(sum(f > 0 for f in self.face_counts)),
            },
            'prediction_counts': {
                cls: sum(1 for p in self.predictions
                         if p['label'] == cls)
                for cls in CLASSES
            },
            'predictions_sample': self.predictions[:50],  # first 50
        }

        fname = os.path.join(
            LOG_DIR,
            f"session_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json")

        with open(fname, 'w') as f:
            json.dump(session, f, indent=2)

        # Append to rolling FPS summary CSV
        csv_path = os.path.join(LOG_DIR, 'pi_sessions.csv')
        write_hdr = not os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            if write_hdr:
                f.write('timestamp,notes,duration_sec,fps_mean,'
                        'fps_min,fps_max,frames_with_face,'
                        'angry,happy,sad,neutral\n')
            pc = session['prediction_counts']
            f.write(f"{session['timestamp']},{self.notes},"
                    f"{duration},"
                    f"{session['fps']['mean']},"
                    f"{session['fps']['min']},"
                    f"{session['fps']['max']},"
                    f"{session['faces']['frames_with_face']},"
                    f"{pc.get('angry',0)},{pc.get('happy',0)},"
                    f"{pc.get('sad',0)},{pc.get('neutral',0)}\n")

        print(f"\n[Logger] Session saved -> {fname}")
        print(f"[Logger] FPS mean={session['fps']['mean']}  "
              f"min={session['fps']['min']}  max={session['fps']['max']}")
        print(f"[Logger] Prediction counts: {session['prediction_counts']}")


# ─────────────────────────────────────────────────────────────────
#  Camera — rpicam-vid pipe
# ─────────────────────────────────────────────────────────────────

def open_camera():
    W, H = FRAME_W, FRAME_H
    frame_size = int(W * H * 1.5)

    cmd = ['rpicam-vid', '-t', '0',
           '--width', str(W), '--height', str(H),
           '--framerate', str(TARGET_FPS),
           '--codec', 'yuv420',
           '--nopreview', '-o', '-']

    print("[Camera] Starting rpicam-vid pipe...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)

    def drain_stderr(pipe):
        try:
            for _ in pipe:
                pass
        except Exception:
            pass

    threading.Thread(target=drain_stderr,
                     args=(proc.stderr,), daemon=True).start()
    open_camera._proc = proc

    def read_frame():
        try:
            raw = b''
            while len(raw) < frame_size:
                chunk = proc.stdout.read(frame_size - len(raw))
                if not chunk:
                    return False, None
                raw += chunk
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                int(H * 1.5), W)
            return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        except Exception as e:
            print(f"[Camera] Read error: {e}")
            return False, None

    print("[Camera] Warming up...")
    for _ in range(10):
        ok, frame = read_frame()
        if ok and frame is not None:
            print(f"[Camera] Ready — {W}x{H} @ {TARGET_FPS}fps")
            return read_frame
        time.sleep(0.2)

    proc.terminate()
    print("[Camera] Failed to get first frame.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────

def load_model(path: str):
    interpreter = TFLiteInterpreter(model_path=path)
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']
    size_kb = os.path.getsize(path) / 1024
    print(f"[Model] Loaded  <- {path}  ({size_kb:.1f} KB)")
    return interpreter, inp_idx, out_idx


def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32) / 255.0
    return face.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict(interpreter, inp_idx, out_idx, tensor):
    interpreter.set_tensor(inp_idx, tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(out_idx)[0]
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs


# ─────────────────────────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────────────────────────

def draw_face_box(frame, x, y, w, h, label, conf, colour):
    cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
    text = f"{label}  {conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x, y-th-12), (x+tw+8, y), (0,0,0), -1)
    cv2.putText(frame, text, (x+4, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)


def draw_prob_bars(frame, probs, classes):
    bx, by = FRAME_W - 180, 10
    bwm, bh, gap = 160, 18, 6
    for i, (cls, p) in enumerate(zip(classes, probs)):
        y   = by + i * (bh + gap)
        col = COLOURS.get(cls, (200,200,200))
        cv2.rectangle(frame, (bx,y),(bx+bwm,y+bh),(40,40,40),-1)
        cv2.rectangle(frame, (bx,y),(bx+int(p*bwm),y+bh),col,-1)
        cv2.putText(frame, f"{cls} {p*100:.0f}%",
                    (bx+4, y+bh-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255,255,255), 1)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, FRAME_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)


# ─────────────────────────────────────────────────────────────────
#  Serial
# ─────────────────────────────────────────────────────────────────

def init_serial():
    import serial
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    print(f"[Serial] {SERIAL_PORT} @ {SERIAL_BAUD}")
    return ser


def send_expression(ser, label):
    try:
        ser.write((label.upper() + '\n').encode())
    except Exception as e:
        print(f"[Serial] Error: {e}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--notes', default='',
                        help='Experiment notes for this session log')
    args = parser.parse_args()

    print("\n[Init] Facial Expression Recognition")
    print(       "       Pi 4B | Trixie | rpicam")
    print(       "       Classes:", CLASSES)
    print(       "       Press Q to quit\n")

    interpreter, inp_idx, out_idx = load_model(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    read_frame = open_camera()
    ser        = init_serial() if USE_SERIAL else None
    session    = SessionLogger(notes=args.notes)

    last_label  = ''
    fps_timer   = time.time()
    frame_count = 0
    fps         = 0.0

    while True:
        ret, frame = read_frame()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60,60))

        detected_label = ''
        detected_conf  = 0.0

        for (x, y, w, h) in faces:
            tensor             = preprocess_face(gray[y:y+h, x:x+w])
            label, conf, probs = predict(interpreter, inp_idx,
                                         out_idx, tensor)
            if conf >= CONF_THRESH:
                colour = COLOURS.get(label, (255,255,255))
                draw_face_box(frame, x, y, w, h, label, conf, colour)
                draw_prob_bars(frame, probs, CLASSES)
                detected_label = label
                detected_conf  = conf

                if ser and label != last_label:
                    send_expression(ser, label)
                    last_label = label

        # FPS
        if frame_count % 10 == 0:
            fps       = 10 / (time.time() - fps_timer)
            fps_timer = time.time()
            print(f"[Main] FPS: {fps:.1f} | faces: {len(faces)}"
                  + (f" | {detected_label} {detected_conf*100:.0f}%"
                     if detected_label else ""))

        session.log_frame(fps, len(faces),
                          detected_label, detected_conf)

        draw_fps(frame, fps)
        cv2.imshow('Expression Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Main] Quit.")
            break

    # Cleanup + save session log
    if hasattr(open_camera, '_proc'):
        open_camera._proc.terminate()
    if ser:
        ser.close()
    cv2.destroyAllWindows()
    session.save()
    print("[Main] Stopped.")


if __name__ == '__main__':
    main()