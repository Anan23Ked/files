# ─────────────────────────────────────────────────────────────────
#  convert.py  —  Convert trained weights -> quantised TFLite model
# ─────────────────────────────────────────────────────────────────

import os
import numpy as np
import tensorflow as tf

from config import IMG_SIZE, CHANNELS, WEIGHTS_PATH, TFLITE_PATH, NUM_CLASSES
from model  import ExpressionCNN


def convert(weights_path: str = WEIGHTS_PATH,
            tflite_path: str = TFLITE_PATH):

    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)

    h5_path = weights_path.replace('.npy', '.weights.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Weights not found at '{h5_path}'. Run train.py first.")

    print("[Convert] Loading model weights...")
    cnn = ExpressionCNN(num_classes=NUM_CLASSES)
    cnn.load_weights(weights_path)

    # Build a concrete function for TFLite export
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, IMG_SIZE, IMG_SIZE, CHANNELS],
                      dtype=tf.float32)])
    def inference_fn(x):
        logits = cnn.forward(x, training=False)
        return tf.nn.softmax(logits)

    print("[Convert] Running TFLite conversion with quantisation...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [inference_fn.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"[Convert] Saved  -> {tflite_path}")
    print(f"[Convert] Size   : {size_kb:.1f} KB")

    _sanity_check(tflite_path)


def _sanity_check(tflite_path: str):
    print("[Convert] Running sanity check...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, CHANNELS).astype(np.float32)
    interpreter.set_tensor(inp_idx, dummy)
    interpreter.invoke()
    probs = interpreter.get_tensor(out_idx)[0]
    print(f"[Convert] Probabilities sum: {probs.sum():.4f}  (should be ~1.0) OK")
    print("[Convert] Conversion complete.")


if __name__ == '__main__':
    convert()