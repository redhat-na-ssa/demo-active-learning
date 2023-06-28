# %%
# setup parameters
scratch = "../scratch/"

# %%
import os, random, pathlib, warnings, itertools, math, json

import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageEnhance

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)

# %%
K.clear_session()

# %%
# setup paths for data
dataset = scratch + "Vegetable Images"

train_folder = os.path.join(dataset, "train")
test_folder = os.path.join(dataset, "validation")
validation_folder = os.path.join(dataset, "test")

# %%
IMAGE_SIZE = [224, 224]

inception = InceptionV3(
    input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False
)

for layer in inception.layers:
    layer.trainable = False

x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)

prediction = Dense(15, activation="softmax")(x)

model = Model(inputs=inception.input, outputs=prediction)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

train_datagen = image.ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = image.ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    train_folder, target_size=(224, 224), batch_size=64, class_mode="categorical"
)

test_set = test_datagen.flow_from_directory(
    test_folder, target_size=(224, 224), batch_size=64, class_mode="categorical"
)

class_map = training_set.class_indices
print(class_map)

# %%
model_metadata = "_inceptionV3_epoch5"
path_to_model = scratch + "model" + model_metadata + ".h5"

# %%
if not os.path.exists(path_to_model):
    print("Model: Training...")
    r = model.fit(
        training_set,
        validation_data=test_set,
        epochs=5,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set),
    )
    print("Model: Trained")

    print("Model: Saving...")
    model.save(path_to_model)
    print("Model: Saved")

else:
    print("Model: Already Exists")
