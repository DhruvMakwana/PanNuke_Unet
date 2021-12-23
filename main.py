# importing required libraries

import os
import tensorflow as tf
from dataloader import DataLoader
from config import *
import matplotlib.pyplot as plt
import numpy as np
from model import FANet
from deeplab import DeeplabV3Plus
from keras_unet_collection import models, utils
from losses import Loss

def main():
    
    # dataloader
    train_image_paths = sorted([os.path.join(ROOT_DIR, DATA_DIR, TRAIN_IMAGES, x) for x in os.listdir(os.path.join(ROOT_DIR, DATA_DIR, TRAIN_IMAGES)) if x.endswith('.png')])
    train_mask_paths = sorted([os.path.join(ROOT_DIR, DATA_DIR, TRAIN_MASKS, x) for x in os.listdir(os.path.join(ROOT_DIR, DATA_DIR, TRAIN_MASKS)) if x.endswith('.png')])

    valid_image_paths = sorted([os.path.join(ROOT_DIR, DATA_DIR, VALID_IMAGES, x) for x in os.listdir(os.path.join(ROOT_DIR, DATA_DIR, VALID_IMAGES)) if x.endswith('.png')])
    valid_mask_paths = sorted([os.path.join(ROOT_DIR, DATA_DIR, VALID_MASKS, x) for x in os.listdir(os.path.join(ROOT_DIR, DATA_DIR, VALID_MASKS)) if x.endswith('.png')])

    test_image_paths = sorted([os.path.join(ROOT_DIR, DATA_DIR, TEST_IMAGES, x) for x in os.listdir(os.path.join(ROOT_DIR, DATA_DIR, TEST_IMAGES)) if x.endswith('.png')])
    test_mask_paths = sorted([os.path.join(ROOT_DIR, DATA_DIR, TEST_MASKS, x) for x in os.listdir(os.path.join(ROOT_DIR, DATA_DIR, TEST_MASKS)) if x.endswith('.png')])

    trainDS = DataLoader(image_paths = train_image_paths,
                        mask_paths = train_mask_paths,
                        image_size = (256, 256),
                        # crop_percent = 0.8,
                        palette = [[255, 255, 255], [83, 63, 207], [102, 115, 98], [232, 20, 85], [133, 138, 84], [247, 163, 84]],
                        channels = (3, 3),
                        augment = False,
                        compose = False,
                        seed = 47)
                        
    # Parse the images and masks, and return the data in batches, augmented optionally.
    trainDS = trainDS.data_batch(batch_size = BATCH_SIZE, shuffle = False)
    
    # Initialize the dataloader object
    validDS = DataLoader(image_paths = valid_image_paths,
                        mask_paths = valid_mask_paths,
                        image_size = (256, 256),
                        # crop_percent = 0.8,
                        palette = [[255, 255, 255], [83, 63, 207], [102, 115, 98], [232, 20, 85], [133, 138, 84], [247, 163, 84]],
                        channels = (3, 3),
                        augment = False,
                        compose = False,
                        seed = 47)
                        
    # Parse the images and masks, and return the data in batches, augmented optionally.
    validDS = validDS.data_batch(batch_size = BATCH_SIZE, shuffle = False)
    
    # Initialize the dataloader object
    testDS = DataLoader(image_paths = test_image_paths,
                        mask_paths = test_mask_paths,
                        image_size = (256, 256),
                        # crop_percent = 0.8,
                        palette = [[255, 255, 255], [83, 63, 207], [102, 115, 98], [232, 20, 85], [133, 138, 84], [247, 163, 84]],
                        channels = (3, 3),
                        augment = False,
                        compose = False,
                        seed = 47)
                        
    # Parse the images and masks, and return the data in batches, augmented optionally.
    testDS = testDS.data_batch(batch_size = BATCH_SIZE, shuffle = False)
    print("[Info] Train Dataset: ", trainDS)
    print("[Info] Valid Dataset: ", validDS)
    print("[Info] Test Dataset: ", testDS)

    # model building
    #model = FANet.build(width = 256, height = 256, channel = 3, classes=6)
    #model = DeeplabV3Plus(image_size=256, num_classes=6)
    model = models.att_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], n_labels=6, 
                           stack_num_down=2, stack_num_up=2, activation='ReLU', 
                           atten_activation='ReLU', attention='add', output_activation='Softmax', 
                           batch_norm=True, pool=False, unpool=False, 
                           backbone='VGG16', weights='imagenet', 
                           freeze_backbone=False, freeze_batch_norm=True, 
                           name='attunet')

    print("[Info] Model Summary: \n")
    print(model.summary())

    # compile
    print("[Info] compiling model")
    panopticLoss = Loss()
    loss = panopticLoss.slicesPanopticLoss
    metric = [panopticLoss.diceCoef, panopticLoss.slicesPanopticScore]
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = loss, metrics=metric)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, min_lr=0.00001, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('FANet.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    ]
    model.fit(trainDS, epochs = NUMBER_EPOCHS, validation_data = validDS, callbacks = callbacks)

if __name__ == "__main__":
    main()