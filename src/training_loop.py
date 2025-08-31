import numpy as np
import os
from DataGenerator import get_image_paths_and_labels
from model_builders import build_model
from training import train_with_cv_generator

architectures = ["VGG16", "ResNet101", "DenseNet121", "InceptionV3", "Xception"]
classifier_heads = ["softmax", "svm", "rf"]
results = []

#load only image paths and labels
image_paths, labels, label_map = get_image_paths_and_labels("../data/Brain Cancer Dataset/Training")
labels = np.array(labels)

#zero base the labels
labels -= 1
label_map_zero_base = {k: v - 1 for k, v in label_map.items()}
inverse_label_map = {v: k for k, v in label_map_zero_base.items()}

num_classes = len(label_map_zero_base)

# loop through each architecture and classifier head
for arch in architectures:
    for head in classifier_heads:
        print(f"=== Training {arch} with {head} head ===")
        try:
            build_fun = lambda: build_model(arch, num_classes=num_classes, classifier_head=head, dropout_rate=0.3)
            models, df_metrics, histories = train_with_cv_generator(
                build_fn=build_fun,
                image_paths=image_paths,
                labels=labels,
                classifier_head=head,
                folds=10,
                epochs=10,
                batch_size=32,
                arch_name=arch,
                augment=True,
                add_noise=True,
                target_size=(224, 224),
                seed=3
            )
        except Exception as e:
            print(f"Error training {arch} with {head}: {e}")
            
