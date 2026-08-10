import numpy as np
from PIL import Image
from joblib import load
import matplotlib.pyplot as plt
from pathlib import Path

pixel_dims = 128
# model = load("model/cat_model.joblib")
# model = load("model/cat_model_L2_regularized.joblib")
model = load("model/cat_model_dropout_regularized.joblib")

folder = Path("separate_images")
all_files = list(folder.iterdir())
total = len(all_files)
correct = 0

# ================== test with separate images =======================
for img_path in all_files:
    img = np.array(Image.open(img_path).resize((pixel_dims, pixel_dims)).convert("RGB"))
    img_usable = ((img.reshape(1, -1) - model.mew) / np.sqrt(model.sigma)).T

    # plt.imshow(img)
    # plt.show()

    pred = model.predict(img_usable)[0, 0]

    if pred == 1:
        print("Model predicts cat")
    else:
        print("model predicts not cat")

    if "cat" in img_path.name:
        if pred == 1:
            print("Model is correct")
            correct += 1
        else:
            print("Model is incorrect")
    else:
        if pred == 0:
            print("Model is correct")
            correct += 1
        else:
            print("Model is incorrect")

    print()

print(f"Correct: {correct}")
print(f"Accuracy: {correct/total}")
