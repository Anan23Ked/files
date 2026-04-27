# ─────────────────────────────────────────────────────────────────
#  config.py  —  All hyperparameters, paths, and settings
# ─────────────────────────────────────────────────────────────────

# ── Dataset ───────────────────────────────────────────────────────
TRAIN_DIR   = 'dataset/train'
TEST_DIR    = 'dataset/test'
CLASSES     = ['angry', 'happy', 'sad', 'neutral']
NUM_CLASSES = len(CLASSES)

# ── Image ─────────────────────────────────────────────────────────
IMG_SIZE    = 48          # pixels (height and width)
CHANNELS    = 1           # 1 = grayscale

# ── Training ──────────────────────────────────────────────────────
BATCH_SIZE  = 32
EPOCHS      = 50
LR_INITIAL  = 0.001
LR_DECAY    = 0.5         # multiply LR by this on plateau
LR_PATIENCE = 5           # epochs before reducing LR
ES_PATIENCE = 10          # epochs before early stopping

# ── Augmentation ──────────────────────────────────────────────────
AUG_FLIP        = True
AUG_BRIGHTNESS  = 0.2     # max_delta
AUG_CONTRAST    = (0.8, 1.2)
AUG_ROTATION    = 10      # degrees

# ── Paths ─────────────────────────────────────────────────────────
WEIGHTS_PATH    = 'outputs/best_weights.npy'
TFLITE_PATH     = 'outputs/expression_model.tflite'
HISTORY_PLOT    = 'outputs/training_history.png'
CM_PLOT         = 'outputs/confusion_matrix.png'
