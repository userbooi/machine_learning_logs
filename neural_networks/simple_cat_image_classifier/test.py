import numpy as np
from PIL import Image
from joblib import load
import matplotlib.pyplot as plt

pixel_dims = 128
model = load("models/cat_model.joblib")

# check some images
# plt.imshow(X_raw[52])
# plt.show()

# ================== test with separate images =======================
img1 = np.array(Image.open("separate_images/kawhi_leonard.jpeg").resize((pixel_dims, pixel_dims)).convert("RGB"))
img1_usable = img1.reshape(-1, 1) / 255
plt.imshow(img1)
plt.show()
print(model.predict(img1_usable)) # 0 - correct

img2 = np.array(Image.open("separate_images/one_more_cat.jpg").resize((pixel_dims, pixel_dims)).convert("RGB"))
img2_usable = img2.reshape(-1, 1) / 255
plt.imshow(img2)
plt.show()
print(model.predict(img2_usable)) # 1 - correct

img3 = np.array(Image.open("separate_images/final_cat.jpg").resize((pixel_dims, pixel_dims)).convert("RGB"))
img3_usable = img3.reshape(-1, 1) / 255
plt.imshow(img3)
plt.show()
print(model.predict(img3_usable)) # 1 - correct

img4 = np.array(Image.open("separate_images/fried_chicken.jpeg").resize((pixel_dims, pixel_dims)).convert("RGB"))
img4_usable = img4.reshape(-1, 1) / 255
plt.imshow(img4)
plt.show()
print(model.predict(img4_usable)) # 0 - correct

img5 = np.array(Image.open("separate_images/sideways_cat.jpg").resize((pixel_dims, pixel_dims)).convert("RGB"))
img5_usable = img5.reshape(-1, 1) / 255
plt.imshow(img5)
plt.show()
print(model.predict(img5_usable)) # 1 - correct