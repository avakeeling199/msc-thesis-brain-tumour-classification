import matplotlib.pyplot as plt
import numpy as np
from DataGenerator import get_image_paths_and_labels
shap_values = np.load("../results/shap_values.npy", allow_pickle=True)
print(shap_values.shape)
breakpoint()
shap_per_class = np.abs(shap_values).mean(axis=1)
num_classes = shap_values.shape[2]
top_n = 10
sample_idx = 0

_, _, label_map = get_image_paths_and_labels("../data/Brain Cancer Dataset/Training")
class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

plt.figure(figsize=(6, num_classes*2))

for i in range(num_classes):
    class_shap = np.abs(shap_values[sample_idx, :, i])
    top_idx = np.argsort(class_shap)[-top_n:]
    top_values = class_shap[top_idx]
    plt.subplot((num_classes + 1) // 2, 2, i + 1)
    plt.barh(range(top_n), top_values, color='indigo')
    plt.yticks(range(top_n), top_idx)
    plt.title(class_names[i])
    plt.xlabel("SHAP value (importance)")
plt.tight_layout()
plt.show()