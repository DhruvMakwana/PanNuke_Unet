# importing required libraries

# usage: python download_dataset.py

import tensorflow as tf
from config import *
import os

# download dataset
fold1 = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip"
fold2 = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip"
fold3 = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip"

fold1 = tf.keras.utils.get_file(origin = fold1, fname=os.path.join(ROOT_DIR, DATA_DIR, "fold1.zip"), extract=True, archive_format="zip", cache_dir = os.path.join(ROOT_DIR, DATA_DIR, "fold1"), cache_subdir = os.path.join(ROOT_DIR, DATA_DIR, "fold1"))
fold2 = tf.keras.utils.get_file(origin = fold2, fname=os.path.join(ROOT_DIR, DATA_DIR, "fold2.zip"), extract=True, archive_format="zip", cache_dir = os.path.join(ROOT_DIR, DATA_DIR, "fold2"), cache_subdir = os.path.join(ROOT_DIR, DATA_DIR, "fold2"))
fold3 = tf.keras.utils.get_file(origin = fold3, fname=os.path.join(ROOT_DIR, DATA_DIR, "fold3.zip"), extract=True, archive_format="zip", cache_dir = os.path.join(ROOT_DIR, DATA_DIR, "fold3"), cache_subdir = os.path.join(ROOT_DIR, DATA_DIR, "fold3"))
print("\nFold1 is downloaded at ", fold1)
print("Fold2 is downloaded at ", fold2)
print("Fold3 is downloaded at ", fold3)