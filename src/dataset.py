import os
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from src.config import CLASSES

def preprocess_image(image):
    """Tiền xử lý ảnh: resize, chuyển grayscale, chuẩn hóa pixel"""
    image = cv2.resize(image, (224, 224))  # Resize về 224x224
    image = image / 255.0  # Chuẩn hóa về [0, 1]
    return image

class VSLR_Dataset(Dataset):
    def __init__(self, root_path, transform=None, total_image_per_class=500, ratio_train=0.8, ratio_val=0.1, mode="train"):
        self.root_path = root_path
        self.num_class = len(CLASSES)
        self.transform = transform

        ratio_test = 1 - (ratio_train + ratio_val)
        self.num_image_per_class = {
            "train": math.ceil(total_image_per_class * ratio_train),
            "val": math.ceil(total_image_per_class * ratio_val),
            "test": math.ceil(total_image_per_class * ratio_test),
        }[mode]

        self.image_paths = []
        self.labels = []
        
        for class_name in CLASSES:
            class_dir = os.path.join(self.root_path, class_name)
            if not os.path.exists(class_dir):
                continue

            image_files = os.listdir(class_dir)
            total_images = len(image_files)
            random.shuffle(image_files)

            train_end = math.ceil(total_images * ratio_train)
            val_end = train_end + math.ceil(total_images * ratio_val)

            if mode == "train":
                selected_files = image_files[:train_end]
            elif mode == "val":
                selected_files = image_files[train_end:val_end]
            else:  # mode == "test"
                selected_files = image_files[val_end:]

            for image_name in selected_files:
                image_path = os.path.join(class_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(CLASSES.index(class_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh grayscale
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")

        image = preprocess_image(image)
        image = np.expand_dims(image, axis=-1)  # Thêm chiều kênh (1, H, W)
        image = np.repeat(image, 3, axis=-1)  # Chuyển thành ảnh có 3 kênh để phù hợp với ResNet
        image = Image.fromarray((image * 255).astype(np.uint8))  # Chuyển numpy array thành PIL Image

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

if __name__ == '__main__':
    root_path = "../dataset/alphabet"
    
    training_set = VSLR_Dataset(root_path, mode="train")
    validation_set = VSLR_Dataset(root_path, mode="val")
    test_set = VSLR_Dataset(root_path, mode="test")

    plt.tight_layout()
    plt.show()
