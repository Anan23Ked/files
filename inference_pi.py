# ─────────────────────────────────────────────────────────────────
#  inference_pi.py  —  Live inference on Raspberry Pi with Camera V2
#
#  Run on Pi:
#    pip install tflite-runtime opencv-python numpy --break-system-packages
#    python inference_pi.py
# ─────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import time

# tflite-runtime is a lightweight package for Pi (not full TF)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback if running full TF on Pi
    import tensorflow.lite as tflite

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH  = 'expression_model.tflite'
CLASSES     = ['angry', 'happy', 'sad', 'neutral']
IMG_SIZE    = 48
CONF_THRESH = 0.55        # minimum confidence to display a label

# ── ESP32 serial (set USE_SERIAL = True when ESP32 is wired up) ───
USE_SERIAL   = False
SERIAL_PORT  = '/dev/ttyS0'   # or /dev/ttyAMA0 depending on Pi model
SERIAL_BAUD  = 9600

# ── Colour map per expression ─────────────────────────────────────
COLOURS = {
    'angry'  : (0,   0,   255),   # red
    'happy'  : (0,   255, 0  ),   # green
    'sad'    : (255, 100, 0  ),   # blue-ish
    'neutral': (200, 200, 200),   # grey
}


# ── Load TFLite model ─────────────────────────────────────────────

def load_model(path: str):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]['index']
    out = interpreter.get_output_details()[0]['index']
    print(f"[Inference] Model loaded ← {path}")
    return interpreter, inp, out


# ── Preprocess face ROI ───────────────────────────────────────────

def preprocess(face_gray: np.ndarray) -> np.ndarray:
    """Resize, normalise, and reshape to model input format."""
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32) / 255.0
    return face.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# ── Run inference ─────────────────────────────────────────────────

def predict(interpreter, inp_idx: int, out_idx: int,
            face_tensor: np.ndarray) -> tuple[str, float]:
    interpreter.set_tensor(inp_idx, face_tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(out_idx)[0]
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])


# ── Serial output to ESP32 ────────────────────────────────────────

def init_serial(port: str, baud: int):
    import serial
    ser = serial.Serial(port, baud, timeout=1)
    print(f"[Inference] Serial open → {port} @ {baud}")
    return ser


def send_expression(ser, label: str):
    """Send newline-terminated label to ESP32."""
    try:
        ser.write((label.upper() + '\n').encode())
    except Exception as e:
        print(f"[Inference] Serial error: {e}")


# ── Main loop ─────────────────────────────────────────────────────

def main():
    interpreter, inp_idx, out_idx = load_model(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)   # Pi Camera V2 via CSI
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. "
                           "Check 'sudo raspi-config' → Interface → Camera.")

    ser = init_serial(SERIAL_PORT, SERIAL_BAUD) if USE_SERIAL else None

    last_label  = ''
    fps_timer   = time.time()
    frame_count = 0

    print("[Inference] Running — press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Inference] Frame read failed.")
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_tensor = preprocess(gray[y:y+h, x:x+w])
            label, conf = predict(interpreter, inp_idx, out_idx, face_tensor)

            colour = COLOURS.get(label, (255, 255, 255))

            if conf >= CONF_THRESH:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)

                # Label with confidence
                text = f'{label}  {conf*100:.0f}%'
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2)

                # Send to ESP32 only when expression changes
                if ser and label != last_label:
                    send_expression(ser, label)
                    last_label = label

        # FPS counter
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_timer)
            fps_timer = time.time()
            print(f"[Inference] FPS: {fps:.1f}")

        cv2.imshow('Facial Expression Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()
    print("[Inference] Stopped.")


if __name__ == '__main__':
    main()
