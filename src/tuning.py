from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from DataGenerator import get_image_paths_and_labels
from model_builders import build_model
from training import train_all_data
import joblib

cv = KFold(n_splits=10, shuffle=True, random_state=3)

top_models = [
    ("DenseNet121", "svm"), 
    ("DenseNet121", "rf"), 
    ("VGG16", "svm"),
    ("Xception", "svm"),
    ("InceptionV3", "svm")
]

params_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
            }

params_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
            }

#load only image paths and labels
image_paths, labels, label_map = get_image_paths_and_labels("../data/Brain Cancer Dataset/Training")
labels = np.array(labels)

#zero base the labels
labels -= 1
label_map_zero_base = {k: v - 1 for k, v in label_map.items()}
inverse_label_map = {v: k for k, v in label_map_zero_base.items()}

num_classes = len(label_map_zero_base)

# loop through each architecture and classifier head
for arch, head in top_models:
    print(f"=== Training {arch} with {head} head ===")
    try:
        build_fun = lambda: build_model(arch, num_classes=num_classes, classifier_head=head, dropout_rate=0.3)
        features, y_labels = train_all_data(
            build_fn=build_fun,
            image_paths=image_paths,
            labels=labels,
            classifier_head=head,
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
        continue

    if head == "svm":
        print("Running GridSearchCV for SVM...")
        grid_svm = GridSearchCV(SVC(), params_svm, scoring='accuracy', cv=cv)
        grid_svm.fit(features, y_labels)
        print(f"Best params for {arch} + SVM:", grid_svm.best_params_)
        print("Best CV accuracy:", grid_svm.best_score_)
        best_clf = grid_svm.best_estimator_
    elif head == "rf":
        print("Running GridSearchCV for Random Forest...")
        grid_rf = GridSearchCV(RandomForestClassifier(random_state=3), params_rf, scoring='accuracy', cv=cv)
        grid_rf.fit(features, y_labels)
        print(f"Best params for {arch} + RF:", grid_rf.best_params_)
        print("Best CV accuracy:", grid_rf.best_score_)
        best_clf = grid_rf.best_estimator_
    
    os.makedirs("../models/Classifiers", exist_ok=True)
    model_file = f"../models/Classifiers/{arch}_{head}_best_model.joblib"
    joblib.dump(best_clf, model_file)
    print(f"Best model saved to {model_file}")