seed = 123

import numpy as np

np.random.seed(seed)
import tensorflow as tf

tf.set_random_seed(seed)

import random

random.seed(seed)

import os

import sys

import time

from my_crypts_dataset import CryptsDataset, CryptsConfig

import model as modellib
from model import log

import numpy as np
from imgaug import augmenters as iaa

import skimage.io

#######################################################################################
## SET UP CONFIGURATION

bowl_config = CryptsConfig()
bowl_config.display()
#######################################################################################

# Root directory of the project
ROOT_DIR = os.getcwd()

## Change this dir to the stage 1 training data
train_dir = os.path.join(ROOT_DIR, 'dataset/new_crypts/train/Images')
print(train_dir)

# Get train IDs
train_ids = next(os.walk(train_dir))[2]

# Training dataset
dataset_train = CryptsDataset()
dataset_train.load_bowl(train_ids, 'dataset/new_crypts/train')
dataset_train.prepare()

# # Validation dataset, same as training.. will use pad64 on this one
val_dir = os.path.join(ROOT_DIR, 'dataset/new_crypts/valid/Images/')
valid_ids = next(os.walk(val_dir))[2]
dataset_val = CryptsDataset()
dataset_val.load_bowl(valid_ids, 'dataset/new_crypts/valid')
dataset_val.prepare()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
## https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

model = modellib.MaskRCNN(mode="training", config=bowl_config,
                          model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

import time

start_time = time.time()

## Augment True will perform flipud fliplr and 90 degree rotations on the 512x512 images

# Image augmentation
# http://imgaug.readthedocs.io/en/latest/source/augmenters.html

# ## This should be the equivalente version of my augmentations using the imgaug library
# ## However, there are subtle differences so I keep my implementation
augmentation = iaa.Sequential([
    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Sometimes(0.5,iaa.Affine(rotate=(-180,180))),
    iaa.Sometimes(0.5,iaa.CropAndPad(percent=(-0.25, 0.25))),
    iaa.Sometimes(0.5, iaa.AddElementwise((-10, 10), per_channel=0.5)),
], random_order=True)

# augmentation = False

model.train(dataset_train, dataset_val,
            learning_rate=bowl_config.LEARNING_RATE,
            epochs=30,
            augmentation=augmentation,
            # augment=True,
            layers="all")

model.train(dataset_train, dataset_val,
            learning_rate=bowl_config.LEARNING_RATE / 10,
            epochs=50,
            augmentation=augmentation,
            # augment=True,
            layers="all")

model.train(dataset_train, dataset_val,
            learning_rate=bowl_config.LEARNING_RATE / 30,
            epochs=75,
            augmentation=augmentation,
            # augment=True,
            layers="all")

end_time = time.time()
ellapsed_time = (end_time - start_time) / 3600

print(model.log_dir)
model_path = os.path.join(model.log_dir, 'final.h5')
model.keras_model.save_weights(model_path)
