import os 
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import random

class BrainTumourDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32,
                target_size=(224, 224), num_classes=4,
                add_noise=False, augment=False, shuffle=True,
                seed=None):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.add_noise = add_noise
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed

        if self.seed is not None: 
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.on_epoch_end()

    def __len__(self):
        length = int(np.ceil(len(self.image_paths) / self.batch_size))
        return length
    
    def on_epoch_end(self):
        if self.shuffle:
            if self.seed is not None:
                rng = np.random.default_rng(self.seed)
                indices = np.arange(len(self.image_paths))
                rng.shuffle(indices)
                self.image_paths = [self.image_paths[i] for i in indices]
                self.labels = [self.labels[i] for i in indices]
            else:
                zipped = list(zip(self.image_paths, self.labels))
                random.shuffle(zipped)
                self.image_paths, self.labels = zip(*zipped)

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError(F"Index {index} out of range for generator with length {self.__len__()}")
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        if len(batch_paths) == 0:
            raise ValueError(F"Empty batch at index {index}. This should never happen.")

        batch_images = []
        for path in batch_paths:
            img = self.preprocess_image(path)
            batch_images.append(img)

        if len(batch_images) == 0:
            print(F"Warning: batch_images is empty at index {index}")

        X = np.stack(batch_images, axis=0)
        X = np.repeat(X[..., np.newaxis], 3, axis=-1) #greyscale to rgb
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)

        return X, y
    
    def preprocess_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.target_size)

        if self.augment: 
            img = self.random_augment(img)

        img = img.astype(np.float32)

        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / (std + 1e-8)

        if self.add_noise:
            noise = np.random.normal(0, 0.05, img.shape)
            img += noise
            img = np.clip(img, -3, 3)

        return img

    def random_augment(self, img):
        h, w = img.shape

        # random shift +/- 10%
        max_shift = int(0.1 * h)
        dx, dy = np.random.randint(-max_shift, max_shift + 1, size=2)
        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)

        #random rotation +/- 15 deg
        angle = np.random.uniform(-15, 15)
        M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        img = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT)

        #random zoom
        zoom_factor = np.random.uniform(0.95, 1.05)
        new_size = int(h * zoom_factor)
        zoomed = cv2.resize(img, (new_size, new_size))

        if new_size > h:
            start = (new_size - h) // 2
            img = zoomed[start:start + h, start:start + w]
        else:
            pad_h = (h - new_size) // 2
            pad_w = (w - new_size) // 2
            img = cv2.copyMakeBorder(
                zoomed, pad_h, h - new_size - pad_h, pad_w, w - new_size - pad_w,
                borderType=cv2.BORDER_REFLECT)
            
        return img
    

def get_image_paths_and_labels(base_path):
    image_paths = []
    labels = []
    label_map = {}

    for idx, class_name in enumerate(sorted(os.listdir(base_path))):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue
        label_map[class_name] = idx
        for file in os.listdir(class_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(class_path, file))
                labels.append(idx)

    return image_paths, labels, label_map