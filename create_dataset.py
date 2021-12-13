# importing required libraries

# usage: python create_dataset.py

import numpy as np
from config import *
from tqdm import tqdm
import os
import cv2

print("[Info] loading Images and Masks from 3 Folds")

fold1Images = np.load(os.path.join(ROOT_DIR, DATA_DIR, "Fold 1/Images/images.npy"), mmap_mode="r")
fold1Masks = np.load(os.path.join(ROOT_DIR, DATA_DIR, "Fold 1/Masks/masks.npy"), mmap_mode="r")

fold2Images = np.load(os.path.join(ROOT_DIR, DATA_DIR, "Fold 2/Images/images.npy"), mmap_mode="r")
fold2Masks = np.load(os.path.join(ROOT_DIR, DATA_DIR, "Fold 2/Masks/masks.npy"), mmap_mode="r")

fold3Images = np.load(os.path.join(ROOT_DIR, DATA_DIR, "Fold 3/Images/images.npy"), mmap_mode="r")
fold3Masks = np.load(os.path.join(ROOT_DIR, DATA_DIR, "Fold 3/Masks/masks.npy"), mmap_mode="r")

color_values = {0: [255, 255, 255],
                1: [207, 63, 83],
                2: [98, 115, 102],
                3: [85, 20, 232],
                4: [84, 138, 133],
                5: [84, 163, 247]}

def onehot_to_rgb(onehot, colormap = color_values):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,))
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

def rgb_to_onehot(rgb_arr, color_dict = color_values):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.float32 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

# create images and mask directory
if not os.path.exists(os.path.join(ROOT_DIR, DATA_DIR, "Fold 1/Images/Imgs")):
    os.makedirs(os.path.join(ROOT_DIR, DATA_DIR, "Fold 1/Images/Imgs"))
    print("[Info] Created folder ", os.path.join(ROOT_DIR, DATA_DIR, "Fold 1/Images/Imgs"))
if not os.path.exists(os.path.join(ROOT_DIR, DATA_DIR, "Fold 1/Masks/Msks")):
    os.makedirs(os.path.join(ROOT_DIR, DATA_DIR, "Fold 1/Masks/Msks"))
    print("[Info] Created folder ", os.path.join(ROOT_DIR, DATA_DIR, "Fold 1/Masks/Msks"))
if not os.path.exists(os.path.join(ROOT_DIR, DATA_DIR, "Fold 2/Images/Imgs")):
    os.makedirs(os.path.join(ROOT_DIR, DATA_DIR, "Fold 2/Images/Imgs"))
    print("[Info] Created folder ", os.path.join(ROOT_DIR, DATA_DIR, "Fold 2/Images/Imgs"))
if not os.path.exists(os.path.join(ROOT_DIR, DATA_DIR, "Fold 2/Masks/Msks")):
    os.makedirs(os.path.join(ROOT_DIR, DATA_DIR, "Fold 2/Masks/Msks"))
    print("[Info] Created folder ", os.path.join(ROOT_DIR, DATA_DIR, "Fold 2/Masks/Msks"))
if not os.path.exists(os.path.join(ROOT_DIR, DATA_DIR, "Fold 3/Images/Imgs")):
    os.makedirs(os.path.join(ROOT_DIR, DATA_DIR, "Fold 3/Images/Imgs"))
    print("[Info] Created folder ", os.path.join(ROOT_DIR, DATA_DIR, "Fold 3/Images/Imgs"))
if not os.path.exists(os.path.join(ROOT_DIR, DATA_DIR, "Fold 3/Masks/Msks")):
    os.makedirs(os.path.join(ROOT_DIR, DATA_DIR, "Fold 3/Masks/Msks"))
    print("[Info] Created folder ", os.path.join(ROOT_DIR, DATA_DIR, "Fold 3/Masks/Msks"))

with tqdm(total = fold1Images.shape[0]) as pbar:
    for i, (image, mask) in enumerate(zip(fold1Images, fold1Masks)):
        cv2.imwrite(os.path.join(ROOT_DIR, DATA_DIR, "Fold 1/Images/Imgs/image_{}.png".format(i)), image.astype("uint8"))
        mask = onehot_to_rgb(mask)
        cv2.imwrite(os.path.join(ROOT_DIR, DATA_DIR, "Fold 1/Masks/Msks/image_{}.png".format(i)), mask.astype("uint8"))
        pbar.update(1)

print("[Info] Fold 1 Images and Masks are Created")

with tqdm(total = fold2Images.shape[0]) as pbar:
    for i, (image, mask) in enumerate(zip(fold2Images, fold2Masks)):
        cv2.imwrite(os.path.join(ROOT_DIR, DATA_DIR, "Fold 2/Images/Imgs/image_{}.png".format(i)), image.astype("uint8"))
        mask = onehot_to_rgb(mask)
        cv2.imwrite(os.path.join(ROOT_DIR, DATA_DIR, "Fold 2/Masks/Msks/image_{}.png".format(i)), mask.astype("uint8"))
        pbar.update(1)

print("[Info] Fold 2 Images and Masks are Created")

with tqdm(total = fold3Images.shape[0]) as pbar:
    for i, (image, mask) in enumerate(zip(fold3Images, fold3Masks)):
        cv2.imwrite(os.path.join(ROOT_DIR, DATA_DIR, "Fold 3/Images/Imgs/image_{}.png".format(i)), image.astype("uint8"))
        mask = onehot_to_rgb(mask)
        cv2.imwrite(os.path.join(ROOT_DIR, DATA_DIR, "Fold 3/Masks/Msks/image_{}.png".format(i)), mask.astype("uint8"))
        pbar.update(1)

print("[Info] Fold 3 Images and Masks are Created")
