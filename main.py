# ─────────────────────────────────────────────────────────────────
#  main.py  —  Full pipeline entry point
#
#  Usage:
#    python main.py              # train + evaluate + convert
#    python main.py --skip-train # evaluate + convert (needs weights)
# ─────────────────────────────────────────────────────────────────

import os
import sys
import json
import argparse

from config   import (TRAIN_DIR, TEST_DIR, NUM_CLASSES,
                      WEIGHTS_PATH, TFLITE_PATH)
from dataset  import build_pipelines
from model    import ExpressionCNN
from train    import train
from evaluate import evaluate
from convert  import convert


def parse_args():
    parser = argparse.ArgumentParser(
        description='Facial Expression Recognition — training pipeline')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training, load existing weights')
    parser.add_argument('--skip-eval',  action='store_true',
                        help='Skip evaluation plots')
    parser.add_argument('--skip-convert', action='store_true',
                        help='Skip TFLite conversion')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs('outputs', exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 1 — Load dataset")
    print("=" * 55)
    train_ds, test_ds, class_counts = build_pipelines(
        TRAIN_DIR, TEST_DIR, NUM_CLASSES)

    print("\nClass distribution (train set):")
    for cls, cnt in class_counts.items():
        bar = '█' * (cnt // 10)
        print(f"  {cls:<12} {cnt:>4}  {bar}")

    # ── 2. Model ──────────────────────────────────────────────────
    model = ExpressionCNN(num_classes=NUM_CLASSES)
    model.summary()

    # ── 3. Train ──────────────────────────────────────────────────
    history = None

    if not args.skip_train:
        print("=" * 55)
        print("  STEP 2 — Train")
        print("=" * 55)
        history = train(model, train_ds, test_ds)

        # Persist history for evaluate.py to reuse
        with open('outputs/history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print("[Main] History saved → outputs/history.json")

    else:
        print("[Main] Skipping training — loading existing weights.")
        if os.path.exists(WEIGHTS_PATH):
            model.load_weights(WEIGHTS_PATH)
        else:
            print(f"[Main] ERROR: No weights at '{WEIGHTS_PATH}'. "
                  "Remove --skip-train to train first.")
            sys.exit(1)

        # Try to reload history for plots
        hist_path = 'outputs/history.json'
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)

    # ── 4. Evaluate ───────────────────────────────────────────────
    if not args.skip_eval:
        print("\n" + "=" * 55)
        print("  STEP 3 — Evaluate")
        print("=" * 55)
        evaluate(model, test_ds, history)

    # ── 5. Convert to TFLite ──────────────────────────────────────
    if not args.skip_convert:
        print("\n" + "=" * 55)
        print("  STEP 4 — Convert to TFLite for Raspberry Pi")
        print("=" * 55)
        convert(WEIGHTS_PATH, TFLITE_PATH)

    # ── Done ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Weights  → {WEIGHTS_PATH}")
    print(f"  TFLite   → {TFLITE_PATH}")
    print("  Copy expression_model.tflite + inference_pi.py to your Pi")
    print("=" * 55)


if __name__ == '__main__':
    main()
