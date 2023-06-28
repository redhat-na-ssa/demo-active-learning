import os
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_single_tag_keys, get_choice

logger = logging.getLogger(__name__)


class VegetableClassifier(LabelStudioMLBase):
    def __init__(self, trainable=False, batch_size=32, epochs=3, **kwargs):
        super(ActiveVegetableClassifier, self).__init__(**kwargs)

        self.image_width, self.image_height = 224, 224
        self.trainable = trainable
        self.batch_size = batch_size
        self.epochs = epochs

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema["to_name"][0]
        self.labels = schema["labels"]

        (
            self.from_name,
            self.to_name,
            self.value,
            self.labels_in_config,
        ) = get_single_tag_keys(self.parsed_label_config, "Choices", "Image")

        print(self.labels_in_config)
        self.labels = tf.convert_to_tensor(sorted(self.labels_in_config))

        num_classes = len(self.labels_in_config)
        self.model = self.load_model_from_local_file()

        self.category = {
            0: "Bean",
            1: "Bitter_Gourd",
            2: "Bottle_Gourd",
            3: "Brinjal",
            4: "Broccoli",
            5: "Cabbage",
            6: "Capsicum",
            7: "Carrot",
            8: "Cauliflower",
            9: "Cucumber",
            10: "Papaya",
            11: "Potato",
            12: "Pumpkin",
            13: "Radish",
            14: "Tomato",
        }

        if self.train_output:
            model_file = self.train_output["model_file"]
            logger.info("Restore model from " + model_file)
            # Restore previously saved weights
            self.labels = self.train_output["labels"]
            self.model.load_weights(self.train_output["model_file"])

        print(kwargs)

    def load_model_from_local_file(self):
        path_to_model = os.environ.get("MODEL_PATH", "/data/models/model.h5")
        print("Model: Loading...")
        model = load_model(path_to_model)
        print("Model: Loaded")
        return model

    def predict(self, tasks, **kwargs):
        try:
            self.model.summary()
        except:
            self.model = self.load_model_from_local_file()

        predictions = []
        # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions

        for task in tasks:
            print(task)
            url = task["data"]["image"]
            print(url)
            image_dir = os.getenv(
                "IMAGE_UPLOADED_DIR",
                "/Users/arunhariharan/Library/Application Support/label-studio/media/upload",
            )
            image_path = get_image_local_path(url, None, None, image_dir)
            print(image_path)
            img_ = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img_)
            img_processed = np.expand_dims(img_array, axis=0)
            img_processed /= 255.0
            prediction = self.model.predict(img_processed)
            index = np.argmax(prediction)
            predicted_value = self.category[index]
            print(predicted_value)
            # for each task, return classification results in the form of "choices" pre-annotations
            predictions.append(
                {
                    "result": [
                        {
                            "from_name": self.from_name,
                            "to_name": self.to_name,
                            "type": "choices",
                            "value": {"choices": [predicted_value]},
                        }
                    ],
                    "score": float(1),
                }
            )
        return predictions

    def fit(self, tasks, workdir=None, **kwargs):
        if "data" in kwargs:
            annotations = []
            print(f"*******************")
            print(f"Inside Training now")
            print(f"*******************")
            completion = kwargs
            url = completion["data"]["task"]["data"]["image"]
            image_dir = os.getenv(
                "IMAGE_UPLOADED_DIR",
                "/Users/arunhariharan/Library/Application Support/label-studio/media/upload",
            )
            image_path = get_image_local_path(url, None, None, image_dir)
            image_label = completion["data"]["annotation"]["result"][0]["value"][
                "choices"
            ][0]
            annotations.append((image_path, image_label))

            print(annotations)
            # Create dataset
            ds = tf.data.Dataset.from_tensor_slices(annotations)

            def prepare_item(item):
                print(f"Item Value :  {tf.data.AUTOTUNE} {self.labels} {item}")
                label = tf.argmax(item[1] == self.labels)
                img = tf.io.read_file(item[0])
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [self.image_height, self.image_width])
                return img, label

            ds = ds.map(prepare_item, num_parallel_calls=tf.data.AUTOTUNE)
            ds = (
                ds.cache()
                .shuffle(buffer_size=1000)
                .batch(self.batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE)
            )
            print(self.model)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["acc"],
            )

            self.model.fit(ds, epochs=self.epochs)
            model_file = os.path.join(workdir, "checkpoint")
            self.model.save_weights(model_file)
            print("Training Completed and saved to checkpoint")
            return {"model_file": model_file}

    def get_choice_single(completion):
        return completion["annotation"]["result"][0]["value"]["choices"][0]
