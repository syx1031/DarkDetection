import os
import numpy as np
from PIL import Image
from typing import List
import torch
from transformers import CLIPModel, CLIPProcessor
import chromadb
from chromadb.config import Settings
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CLIPImageEmbeddings:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"Using device: {self.device}")

    def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
        """对图像列表进行嵌入编码"""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_single_image(self, image: Image.Image) -> List[float]:
        """对单张图像进行嵌入编码"""
        return self.embed_images([image])[0]


def create_image_index(folder_path: str, index_path: str, collection_name: str = "image_collection"):
    """创建图像索引数据库"""
    print(f"Creating index from folder: {folder_path}")

    # 初始化 CLIP 嵌入器
    embedder = CLIPImageEmbeddings()

    # 获取所有图像文件路径
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(image_extensions):
            full_path = os.path.join(folder_path, f)
            image_paths.append(full_path)

    if not image_paths:
        print("No images found in the specified folder!")
        return

    print(f"Found {len(image_paths)} images")

    # 加载并处理图像
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            continue

    if not images:
        print("No valid images could be loaded!")
        return

    print(f"Successfully loaded {len(images)} images")

    # 生成嵌入向量
    print("Generating embeddings...")
    embeddings = embedder.embed_images(images)

    # 创建 ChromaDB 客户端
    client = chromadb.PersistentClient(path=index_path)

    # 删除已存在的集合（如果有）
    try:
        client.delete_collection(name=collection_name)
    except:
        pass

    # 创建新集合
    collection = client.create_collection(name=collection_name)

    # 准备数据
    ids = [f"img_{i}" for i in range(len(valid_paths))]
    metadatas = [{"path": path, "filename": os.path.basename(path)} for path in valid_paths]
    documents = [f"Image: {os.path.basename(path)}" for path in valid_paths]

    # 添加到集合中
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully indexed {len(valid_paths)} images to {index_path}")


def query_similar_images(query_image_path: str, index_path: str, top_k: int = 3,
                         collection_name: str = "image_collection"):
    """查询与目标图像最相似的图片"""
    print(f"Querying similar images for: {query_image_path}")

    # 初始化 CLIP 嵌入器
    embedder = CLIPImageEmbeddings()

    # 加载查询图像
    try:
        query_image = Image.open(query_image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading query image: {e}")
        return []

    # 生成查询图像的嵌入向量
    query_embedding = embedder.embed_single_image(query_image)

    # 连接到数据库
    try:
        client = chromadb.PersistentClient(path=index_path)
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return []

    # 执行相似度搜索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # 解析结果
    similar_images = []
    if results['metadatas'] and results['distances']:
        for i in range(len(results['metadatas'][0])):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            # ChromaDB 返回的是距离，我们转换为相似度分数（1 - distance）
            similarity_score = 1 - distance
            similar_images.append((metadata['path'], similarity_score))

    return similar_images


def print_database_info(index_path: str, collection_name: str = "image_collection"):
    """打印数据库信息"""
    try:
        client = chromadb.PersistentClient(path=index_path)
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        print(f"Database contains {count} images")

        # 获取前几个条目作为示例
        if count > 0:
            sample = collection.get(limit=3)
            print("Sample entries:")
            for i, (doc, metadata) in enumerate(zip(sample['documents'], sample['metadatas'])):
                print(f"  {i + 1}. {metadata['filename']} - {doc}")
    except Exception as e:
        print(f"Error reading database: {e}")


def generate_yolo_data(yolo_database):
    # 1. 配置参数
    template_file = 'watch_ad_1.png'
    project_dir = yolo_database
    num_train = 10
    num_val = 2

    # 2. 创建目录结构
    images_train = os.path.join(project_dir, 'images/train')
    images_val = os.path.join(project_dir, 'images/val')
    labels_train = os.path.join(project_dir, 'labels/train')
    labels_val = os.path.join(project_dir, 'labels/val')
    os.makedirs(images_train, exist_ok=True)
    os.makedirs(images_val, exist_ok=True)
    os.makedirs(labels_train, exist_ok=True)
    os.makedirs(labels_val, exist_ok=True)

    # 3. 获取图像尺寸
    with Image.open(template_file) as img:
        width, height = img.size

    # 4. YOLO 标签（整个图像为目标：中心在0.5，宽高为1.0）
    label = "0 0.5 0.5 1.0 1.0\n"

    # 5. 生成训练图像和标签
    for i in range(num_train):
        dst_img = os.path.join(images_train, f'template_{i}.png')
        dst_txt = os.path.join(labels_train, f'template_{i}.txt')
        shutil.copy(template_file, dst_img)
        with open(dst_txt, 'w') as f:
            f.write(label)

    # 6. 生成验证图像和标签
    for i in range(num_val):
        dst_img = os.path.join(images_val, f'template_val_{i}.png')
        dst_txt = os.path.join(labels_val, f'template_val_{i}.txt')
        shutil.copy(template_file, dst_img)
        with open(dst_txt, 'w') as f:
            f.write(label)

    # 7. 写入 template.yaml 配置文件
    yaml_content = f"""path: {project_dir}
    train: images/train
    val: images/val
    nc: 1
    names: ['template']
    """
    with open(os.path.join(project_dir, 'template.yaml'), 'w') as f:
        f.write(yaml_content)


def train_yolo(folder_path, model_path):
    model = YOLO('yolov8s.yaml')
    model.train(data=)


# 主函数示例
def main():
    folder_path = "local_database/watch_ad_buttons"
    index_path = "local_database/chroma_clip_index"
    query_path = "local_database/query.png"
    model_path = "local_database/yolo"

    train_yolo(folder_path, model_path)

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist!")
        return

    # 创建索引目录
    # os.makedirs(index_path, exist_ok=True)

    # 创建图像索引
    # create_image_index(folder_path, index_path)

    # 打印数据库信息
    print_database_info(index_path)

    # 查询相似图像
    if os.path.exists(query_path):
        results = query_similar_images(query_path, index_path, top_k=3)

        print(f"\nTop {len(results)} similar images:")
        for i, (path, score) in enumerate(results, 1):
            print(f"{i}. Path: {path}")
            print(f"   Similarity Score: {score:.4f}")
            print()
    else:
        print(f"Query image {query_path} not found!")


if __name__ == "__main__":
    main()
