from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import gc
import os
import time
import json
from DataGenerator import BrainTumourDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

def calculate_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity_per_class = []
    for i in range(len(labels)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1)) # TN
        fp = np.sum(np.delete(cm, i, axis=0)[:, i]) # FP
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    return np.mean(specificity_per_class)

def train_softmax_model(model, X, y, epochs=10, batch_size=32):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size)
    return model, history

def plot_fold_histories(histories, arch_name="Model"):
    valid_histories = [h for h in histories if h is not None]
    if not valid_histories:
        print("No valid histories to plot.")
        return
    
    num_epochs = len(valid_histories[0].history["accuracy"])

    #agg all metrics
    train_acc = np.array([h.history['accuracy'] for h in valid_histories])
    val_acc = np.array([h.history['val_accuracy'] for h in valid_histories])
    train_loss = np.array([h.history['loss'] for h in valid_histories])
    val_loss = np.array([h.history['val_loss'] for h in valid_histories])

    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    #accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc.mean(axis=0), label="Avg Training Accuracy")
    plt.plot(epochs_range, val_acc.mean(axis=0), label="Avg Validation Accuracy")
    plt.title(f"{arch_name} Accuracy across folds")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    #loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss.mean(axis=0), label="Avg Training Loss")
    plt.plot(epochs_range, val_loss.mean(axis=0), label="Avg Validation Loss")
    plt.title(f"{arch_name} Loss across folds")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_with_cv(build_fn, X, y, classifier_head="softmax", folds=10, epochs=10, batch_size=32, plot_histories=False, arch_name=None):
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=3)
    metrics = []
    fold_models = []
    histories = []
    labels = np.unique(y)

    os.makedirs("histories", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold + 1}/{folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_or_extractor = build_fn()

        start_time = time.time()

        if classifier_head == "softmax":
            model, history = train_softmax_model(model_or_extractor, X_train, y_train, epochs, batch_size)
            histories.append(history)

            # save history to JSON
            if arch_name:
                hist_path = f"histories/history_{arch_name}_{classifier_head}_fold{fold}.json"
                with open(hist_path, "w") as f:
                    json.dump(history.history, f)

            y_pred_probs = model.predict(X_val)
            y_pred = np.argmax(y_pred_probs, axis=1)

        elif classifier_head == "svm":
            features_train = model_or_extractor.predict(X_train)
            features_val = model_or_extractor.predict(X_val)
            clf = SVC(probability=True)
            clf.fit(features_train, y_train)
            y_pred_probs = clf.predict_proba(features_val)
            y_pred = clf.predict(features_val)
            model = (model_or_extractor, clf)  # Save both feature extractor and classifier

        elif classifier_head == "rf":
            features_train = model_or_extractor.predict(X_train)
            features_val = model_or_extractor.predict(X_val)
            clf = RandomForestClassifier(random_state=3)
            clf.fit(features_train, y_train)
            y_pred_probs = clf.predict_proba(features_val)
            y_pred = clf.predict(features_val)
            model = (model_or_extractor, clf)

        end_time = time.time()
        train_duration = end_time - start_time

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
        spec = calculate_specificity(y_val, y_pred, labels)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        try:
            roc = roc_auc_score(y_val, y_pred_probs, multi_class="ovr", average="macro")
        except:
            roc = np.nan
        
        metrics.append({
            "Architecture": arch_name,
            "Classifier": classifier_head,
            "Fold": fold + 1,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Specificity": spec,
            "F1 Score": f1,
            "ROC AUC": roc,
            "Training Time (s)": train_duration
        })

        fold_models.append(model)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
                    # ⚠️ Clear memory here, at end of each fold
        del model
        if classifier_head == "softmax":
            del history
        K.clear_session()
        gc.collect()
    
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(f"results_{arch_name}_{classifier_head}.csv", index=False)

    return fold_models, df_metrics, histories

def train_with_cv_generator(build_fn, image_paths, labels, classifier_head="softmax", folds=10, epochs=10, batch_size=32, 
                            arch_name=None, augment=False, add_noise=False, target_size=(224, 224), seed=3):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    metrics = []
    fold_models = []
    histories = []
    label_set = np.unique(labels)

    os.makedirs("histories", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths, labels)):
        print(f"\n Fold {fold + 1}/{folds}")

        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        # build generators
        train_gen = BrainTumourDataGenerator(train_paths, train_labels, batch_size=batch_size, 
                                            target_size=target_size, num_classes=len(label_set), 
                                            add_noise=add_noise, augment=augment, shuffle=False, seed=seed)
        
        val_gen = BrainTumourDataGenerator(val_paths, val_labels, batch_size=batch_size, 
                                            target_size=target_size, num_classes=len(label_set), 
                                            add_noise=False, augment=False, shuffle=False, seed=seed)

        model_or_extractor = build_fn()

        start_time = time.time()

        if classifier_head == "softmax":
            model_or_extractor.compile(
                optimizer="adam",
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model_or_extractor.fit(train_gen, validation_data=val_gen,
                                            epochs=epochs, callbacks=[es], verbose=1)
            histories.append(history)

            if arch_name:
                hist_path = f"histories/history_{arch_name}_{classifier_head}_fold{fold}.json"
                with open(hist_path, "w") as f:
                    json.dump(history.history, f)

            y_true, y_pred, y_probs = [], [], []
            for x_batch, y_batch in val_gen:
                probs = model_or_extractor.predict(x_batch)
                preds = np.argmax(probs, axis=1)
                y_pred.extend(preds)
                y_true.extend(np.argmax(y_batch, axis=1))
                y_probs.extend(probs)

        elif classifier_head in ["svm", "rf"]:
            #extract features\
            train_gen_no_shuffle = BrainTumourDataGenerator(train_paths, train_labels, batch_size=batch_size, 
                                            target_size=target_size, num_classes=len(label_set), 
                                            add_noise=add_noise, augment=augment, shuffle=False, seed=seed)
            val_gen_no_shuffle = BrainTumourDataGenerator(val_paths, val_labels, batch_size=batch_size, 
                                            target_size=target_size, num_classes=len(label_set), 
                                            add_noise=add_noise, augment=augment, shuffle=False, seed=seed)

            features_train = model_or_extractor.predict(train_gen_no_shuffle)
            features_val = model_or_extractor.predict(val_gen_no_shuffle)

            y_train_array = np.array(train_labels)
            y_val_array = np.array(val_labels)

            np.savez(f"features_{arch_name}_{classifier_head}_fold{fold}.npz", 
            X_train=features_train, y_train=y_train_array,
            X_val=features_val, y_val=y_val_array)

            scaler = StandardScaler()
            features_train = scaler.fit_transform(features_train)
            features_val = scaler.transform(features_val)

            if classifier_head == "svm":
                clf = SVC(probability=True)
            else:
                clf = RandomForestClassifier(random_state=3)

            clf.fit(features_train, y_train_array)
            y_probs = clf.predict_proba(features_val)
            y_pred = clf.predict(features_val)
            y_true = y_val_array
            model_or_extractor = (model_or_extractor, clf)

        #eval
        end_time = time.time()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        spec = calculate_specificity(y_true, y_pred, label_set)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        try:
            roc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
        except:
            roc = np.nan

        metrics.append({
            "Architecture": arch_name,
            "Classifier": classifier_head,
            "Fold": fold + 1,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Specificity": spec,
            "F1 Score": f1,
            "ROC AUC": roc,
            "Training Time (s)": end_time - start_time
        })

        fold_models.append(model_or_extractor)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

        # clean up
        del model_or_extractor
        if classifier_head == "softmax":
            del history
        K.clear_session()
        gc.collect()

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(f"results_{arch_name}_{classifier_head}.csv", index=False)

    return fold_models, df_metrics, histories

def train_all_data(build_fn, image_paths, labels, classifier_head,
                epochs=10, batch_size=32, arch_name=None, 
                augment=False, add_noise=False, target_size=(224, 224), seed=3):
    import os 
    import time
    import numpy as np
    from DataGenerator import BrainTumourDataGenerator

    assert classifier_head in ["svm", "rf"], "This function is only for SVM or RF classifier heads."

    # set up the generator for feature extraction
    train_gen = BrainTumourDataGenerator(
        image_paths, labels,
        batch_size=batch_size, 
        target_size=target_size,
        num_classes=len(np.unique(labels)),
        add_noise=add_noise, 
        augment=augment,
        shuffle=False, # ensure order for label alignments
        seed=seed)

    #build the CNN feature extractor
    feature_extractor=build_fn()

    print(f"Extracting features for {arch_name} with {classifier_head} head...")
    start = time.time()
    features = feature_extractor.predict(train_gen, verbose=1)
    end = time.time()
    print(f"Extraction completed in {(end - start):.2f} seconds")

    # save extracted features and corresponding labels
    os.makedirs("features", exist_ok=True)
    fname = f"features/features_{arch_name}_{classifier_head}_fulltrain.npz"
    np.savez(fname, X=features, y=labels)
    print(f"Saved to {fname}")

    os.makedirs("../models/Feature Extractors", exist_ok=True)
    model_path = f"../models/Feature Extractors/{arch_name}_{classifier_head}_feature_extractor.h5"
    feature_extractor.save(model_path)
    print(f"Saved CNN feature extractor model to {model_path}")

    return features, labels

def extract_features_for_dataset(build_fn, image_paths, batch_size=32, target_size=(224,224)):
    from DataGenerator import BrainTumourDataGenerator
    import numpy as np

    test_gen = BrainTumourDataGenerator(
        image_paths,
        labels=None,  # no labels needed for feature extraction
        batch_size=batch_size,
        target_size=target_size,
        num_classes=None,
        shuffle=False,
        augment=False,
        add_noise=False
    )
    model = build_fn()
    features = model.predict(test_gen, verbose=1)
    return features
