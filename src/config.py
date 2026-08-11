from pathlib import Path 
import tensorflow as tf 

# Caminhos do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
MODELS_PATH = PROJECT_ROOT / "models"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"

# Configuraoes do modelo
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
VALIDATION_SPLIT = 0.2

# TensorFlow
AUTOTUNE = tf.data.AUTOTUNE