# !pip install pycocotools

from pycocotools.coco import COCO
import requests
import os
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# URL das anotações do COCO (se você quiser todas as anotações de treinamento, por exemplo)
annotation_file = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
image_dir = "/content/coco_images"

# Baixar o arquivo de anotações diretamente
os.makedirs(image_dir, exist_ok=True)

# Baixar o arquivo de anotações
# !wget {annotation_file} -O annotations.zip
# !unzip annotations.zip -d /content/coco_annotations/

from pycocotools.coco import COCO

# Carregar o arquivo de anotações COCO
coco = COCO('/content/coco_annotations/annotations/instances_train2017.json')

# IDs das classes que você quer filtrar
classes = ['bicycle', 'boat', 'oven', 'bed']
class_ids = []

# Buscando os IDs das categorias de interesse
for cls in classes:
    cat_ids = coco.getCatIds(catNms=[cls])
    if cat_ids:
        class_ids.append(cat_ids[0])  # Se a classe existe, adicionar o ID
    else:
        print(f"A classe '{cls}' não foi encontrada no dataset.")

# Verificando os IDs das classes
print(f"IDs das classes: {class_ids}")

# Encontrar todas as imagens dessas classes
img_ids = []
for class_id in class_ids:
    img_ids += coco.getImgIds(catIds=[class_id])

# Remover duplicatas
img_ids = list(set(img_ids))
print(f"Encontradas {len(img_ids)} imagens com as classes especificadas.")

# Função para exibir uma imagem a partir do COCO
def show_image(image_id):
    img_info = coco.loadImgs(image_id)[0]
    image_url = img_info['coco_url']
    image = Image.open(BytesIO(requests.get(image_url).content))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Exibir algumas imagens de exemplo
for image_id in img_ids[:5]:  # Mostrar as 5 primeiras imagens
    show_image(image_id)
