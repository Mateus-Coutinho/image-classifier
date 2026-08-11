import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

import config
from data_loader import load_data


def predict_image(image_path):
    """Carrega o modelo e faz a predição para uma única imagem."""
    
    # 1. Carrega o modelo treinado que foi salvo na pasta models/
    model_path = config.MODELS_PATH / "plant_village_mobilenet.keras"
    if not model_path.exists():
        print(f"Erro: O modelo não foi encontrado em {model_path}")
        return

    print("Carregando o modelo salvo...")
    model = load_model(model_path)

    # 2. Carrega as classes para saber o nome da doença
    _, _, class_names = load_data()

    # 3. Carrega e pré-processa a imagem informada
    print(f"Processando imagem: {image_path}")
    img = load_img(image_path, target_size=config.IMAGE_SIZE)
    img_array = img_to_array(img)
    
    # Adiciona a dimensão do batch: (224, 224, 3) -> (1, 224, 224, 3)
    img_batch = np.expand_array = np.expand_dims(img_array, axis=0)

    # 4. Realiza a previsão
    predictions = model.predict(img_batch)
    
    # Pega o índice da maior probabilidade
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100

    print("\n" + "=" * 40)
    print(f"Resultado da Análise:")
    print(f"Diagnóstico: {predicted_class}")
    print(f"Confiança:   {confidence:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    # Permite passar o caminho da imagem via linha de comando
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predict_image(img_path)
    else:
        print("Uso correto: python3 src/predict.py <caminho_da_imagem>")
        print("Exemplo: python3 src/predict.py data/raw/PlantVillage/Tomato___Healthy/000146db-922b-4d69-8620-835917300390___GH_HL Leaf 259.1.JPG")