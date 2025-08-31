from tensorflow.keras.applications import VGG16, ResNet101, DenseNet121, InceptionV3, Xception
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


ARCHITECTURES = {
    "VGG16": VGG16,
    "ResNet101": ResNet101,
    "DenseNet121": DenseNet121,
    "InceptionV3": InceptionV3,
    "Xception": Xception,
}

def build_feature_extractor(arch_name, input_shape):
    base_model = ARCHITECTURES[arch_name](weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False
    return base_model.input, base_model.output

def build_model(arch_name, num_classes, classifier_head="softmax", input_shape=(224, 224, 3), dropout_rate=0.3):
    inputs, features = build_feature_extractor(arch_name, input_shape)
    
    if classifier_head == "softmax":
        x = GlobalAveragePooling2D()(features)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    elif classifier_head in ["svm", "rf"]:
        x = GlobalAveragePooling2D()(features)
        # Return feature extractor to be used with sklearn classifier
        return Model(inputs=inputs, outputs=x)

    else:
        raise ValueError("Unsupported classifier head")

