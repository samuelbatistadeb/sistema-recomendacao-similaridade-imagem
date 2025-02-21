import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from pycocotools.coco import COCO
import os
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gdown

# Baixar o arquivo de anotações COCO
url = 'https://drive.google.com/uc?export=download&id=1KMp6sd0kfyH8UO27V70mf8VX8PshvmFQ'
output_file = '/content/instances_train2017.json'

response = requests.get(url)
if response.status_code == 200:
    with open(output_file, 'wb') as f:
        f.write(response.content)
    print(f"Arquivo baixado com sucesso: {output_file}")
else:
    print(f"Erro ao baixar o arquivo. Status: {response.status_code}")

# Agora carregue o arquivo de anotações COCO
coco = COCO(output_file)  # Certifique-se de usar o caminho correto

# ID das classes de interesse
classes = ['bicycle', 'boat', 'oven', 'bed']
class_ids = []

# Buscar os IDs das categorias
for cls in classes:
    cat_ids = coco.getCatIds(catNms=[cls])
    if cat_ids:
        class_ids.append(cat_ids[0])  # Se a classe existe, adicionar o ID
    else:
        print(f"A classe '{cls}' não foi encontrada no dataset.")

# Encontrar todas as imagens dessas classes
img_ids = []
for class_id in class_ids:
    img_ids += coco.getImgIds(catIds=[class_id])

# Remover duplicatas
img_ids = list(set(img_ids))

# Criar uma lista para armazenar as URLs das imagens e suas classes
image_paths = []
image_labels = []

# Iterar sobre as imagens e acessar suas anotações
for image_id in img_ids:
    img_info = coco.loadImgs(image_id)[0]
    image_url = img_info['coco_url']

    # Buscar as anotações dessa imagem para determinar as categorias
    annotation_ids = coco.getAnnIds(imgIds=[image_id])
    annotations = coco.loadAnns(annotation_ids)

    # Obter todas as categorias da imagem
    labels = [coco.loadCats(ann['category_id'])[0]['name'] for ann in annotations]
    
    # Armazenar as URLs e as classes para cada imagem
    image_paths.append(image_url)
    image_labels.append(labels)

# Criar um DataFrame com as imagens e suas classes
df = pd.DataFrame({"image_path": image_paths, "label": image_labels})

# Amostra aleatória de 100 imagens para acelerar o processo
sample_size = 100
df_sampled = df.sample(n=sample_size, random_state=42)

# Definindo o modelo MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False  # Congelar as camadas convolucionais

# Função para extrair características de uma imagem
def extract_features_from_input_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Tamanho de entrada do MobileNetV2
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona a dimensão batch
    img_array = preprocess_input(img_array)  # Pré-processa a imagem para o MobileNetV2

    # Extrair as características (sem a camada de classificação final)
    features = base_model.predict(img_array)
    features = features.flatten()  # Achatar para um vetor 1D
    return features

# Função para baixar uma imagem e carregá-la como um array
def download_image(image_url, image_id, save_dir="downloaded_images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Definir o caminho para salvar a imagem localmente
    local_image_path = os.path.join(save_dir, f"{image_id}.jpg")

    try:
        # Fazer o download da imagem
        response = requests.get(image_url)
        response.raise_for_status()  # Verificar se houve erro no download

        # Abrir a imagem com PIL
        img = Image.open(BytesIO(response.content))

        # Salvar a imagem localmente
        img.save(local_image_path)

        return local_image_path

    except Exception as e:
        print(f"Erro ao baixar a imagem {image_url}: {e}")
        return None

# Função para extrair características de uma imagem local
def extract_features_from_local_image(image_path):
    # Carregar a imagem com o tamanho correto para o MobileNetV2
    img = load_img(image_path, target_size=(224, 224))  # MobileNetV2 espera imagens de 224x224
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona a dimensão batch
    img_array = preprocess_input(img_array)  # Pré-processa a imagem

    # Extrair as características (apenas camadas convolucionais)
    features = base_model.predict(img_array)
    features = features.flatten()  # Achata para vetor 1D
    return features

# Função para extrair características de múltiplas imagens
def extract_batch_features(image_urls, image_ids):
    features = []
    for image_url, image_id in zip(image_urls, image_ids):
        # Baixar a imagem e obter o caminho local
        local_image_path = download_image(image_url, image_id)
        if local_image_path:
            # Extrair as características da imagem local
            feature = extract_features_from_local_image(local_image_path)
            features.append(feature)
    return np.array(features)

# Função para calcular a similaridade entre a imagem de entrada e as imagens no dataset
def find_similar_images(input_image_path, all_features, image_paths, top_k=4):
    # Extrair características da imagem de entrada
    input_features = extract_features_from_input_image(input_image_path)

    # Calcular a similaridade de cosseno entre a imagem de entrada e todas as imagens
    similarities = cosine_similarity([input_features], all_features)

    # Ordenar as imagens com base na similaridade (do maior para o menor)
    similar_indices = np.argsort(similarities[0])[::-1]

    # Obter as top_k imagens mais similares
    top_similar_images = [image_paths[i] for i in similar_indices[:top_k]]
    return top_similar_images

# Exemplo de URLs de imagens e IDs
image_urls_sampled = df_sampled['image_path'].tolist()  # URLs das imagens da amostra
image_ids_sampled = df_sampled['image_path'].apply(lambda x: x.split('/')[-1].split('.')[0]).tolist()  # IDs das imagens (usando a parte final da URL)

# Extração das características das imagens
all_features = extract_batch_features(image_urls_sampled, image_ids_sampled)

# Exemplo de uso:
input_image_path = 'path_to_input_image.jpg'  # Caminho da imagem de entrada

# Encontrar as 4 imagens mais similares
similar_images = find_similar_images(input_image_path, all_features, image_urls_sampled, top_k=4)

print("As 4 imagens mais similares são:")
for img in similar_images:
    print(img)

# Salvar as características extraídas em um arquivo .npy
np.save('extracted_features.npy', all_features)

# Salvar os rótulos das imagens (classes) em um arquivo .csv
df_sampled['labels'] = df_sampled['label']
df_sampled[['image_path', 'labels']].to_csv('image_labels.csv', index=False)

print("Características e rótulos salvos com sucesso!")
