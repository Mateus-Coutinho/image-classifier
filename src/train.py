import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import mixed_precision

import config
from data_loader import load_data

# 1. Configura a GPU antes de qualquer alocação
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Erro na GPU: {e}")

# 2. Usa float16 para poupar VRAM na RTX 3050
mixed_precision.set_global_policy('mixed_float16')


def build_model(num_classes):
    """Constrói o modelo usando MobileNetV2 via Transfer Learning."""
    base_model = MobileNetV2(
        input_shape=(*config.IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*config.IMAGE_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    # Saída obrigatória em float32 para estabilidade com mixed_precision
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs, outputs)
    return model


def main():
    print("--- 1. Carregando dados ---")
    train_ds, val_ds, class_names = load_data()

    # Identifica a quantidade de classes com base nos dados carregados
    num_classes = len(class_names)
    print(f"Número de classes detectadas: {num_classes}")

    print("\n--- 2. Construindo o modelo MobileNetV2 ---")
    model = build_model(num_classes)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Cria a pasta de modelos se não existir
    config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.MODELS_PATH / "plant_village_mobilenet.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
    ]

    print("\n--- 3. Iniciando o treinamento na RTX 3050 ---")
    EPOCHS = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"\nTreinamento concluído! Modelo salvo em: {checkpoint_path}")


if __name__ == "__main__":
    main()