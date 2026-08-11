import tensorflow as tf
from tensorflow.keras import mixed_precision
import config

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Erro na GPU: {e}")

mixed_precision.set_global_policy('mixed_float16')


def load_data():
    """Carrega os datasets e extrai a lista de classes."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_PATH,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATASET_PATH,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE
    )

    # Extrai os nomes das classes ANTES do prefetch
    class_names = train_ds.class_names

    # Otimização do pipeline
    train_ds = train_ds.shuffle(100).prefetch(buffer_size=config.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=config.AUTOTUNE)

    # Retorna os datasets e a lista de classes
    return train_ds, val_ds, class_names


if __name__ == "__main__":
    print("Iniciando o carregamento dos dados...")
    train, val, classes = load_data()
    print(f"Dados carregados com sucesso! {len(classes)} classes encontradas.")