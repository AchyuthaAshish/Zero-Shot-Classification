"""Configuration and paths for Local Machine Learning classification.

Conforms to Free Local ML Classifier specifications.
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data" / "training"
TRAINING_CSV_PATH = DATA_DIR / "defect_training.csv"
EVALUATION_DATASET_PATH = BASE_DIR / "tests" / "fixtures" / "evaluation_cases.json"

# Model artifact directories
MODELS_DIR = BASE_DIR / "models"
EMBEDDING_MODEL_DIR = MODELS_DIR / "multilingual_embedding_classifier"
TFIDF_SVM_MODEL_DIR = MODELS_DIR / "tfidf_svm"

# Sentence Transformer model
PRETRAINED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Initial engineering confidence threshold
DEFAULT_LOCAL_CONFIDENCE_THRESHOLD = 0.70

# Training defaults
RANDOM_SEED = 42
VALIDATION_SPLIT_RATIO = 0.20

# Strict 8 approved categories
APPROVED_CATEGORIES = [
    "Mechanical Fault",
    "Electrical Fault",
    "Sensor Fault",
    "Temperature Fault",
    "Software Fault",
    "Power Supply Fault",
    "Communication Fault",
    "Unknown"
]
