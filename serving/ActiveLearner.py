import os
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_single_tag_keys, get_choice

from s3 import S3Images

logger = logging.getLogger(__name__)


class VegetableClassifier(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(VegetableClassifier, self).__init__(**kwargs)

        self.image_width, self.image_height = 224, 224

        print(kwargs)
        print(self.parsed_label_config)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema["to_name"][0]
        self.labels = schema["labels"]

    def default(self, **kwargs):

        self.trainable = False
        self.batch_size = 32
        self.epochs = 3

        print(self.labels_in_config)
        self.labels = tf.convert_to_tensor(sorted(self.labels_in_config))

        num_classes = len(self.labels_in_config)
        self.model = self.load_model_from_local_file()

        if self.train_output:
            model_file = self.train_output["model_file"]
            logger.info("Restore model from " + model_file)
            
            # restore previously saved weights
            self.labels = self.train_output["labels"]
            self.model.load_weights(self.train_output["model_file"])

    def load_model_from_local_file(self):
        path_to_model = os.environ.get("MODEL_PATH", "model_inceptionV3_epoch5.h5")
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

        print(kwargs)

        for task in tasks:
            print(task)
            
            url = task["data"]["image"]
            print(url)
            
            access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
            s3_endpoint = os.getenv("AWS_S3_ENDPOINT", "minio:9000").lstrip("http://")
            bucket_name = os.getenv("AWS_S3_BUCKET", "data")

            s3_image = S3Images(s3_endpoint, access_key, secret_key, secure=False)
            image = s3_image.from_s3(bucket_name, 'Vegetable Images/test/Bean/0001.jpg' )

            image = image.resize((self.image_width, self.image_height))
            image = np.array(image) / 255.0
            result = self.model.predict(image[np.newaxis, ...])
            predicted_label_idx = np.argmax(result[0], axis=-1)
            predicted_label_score = result[0][predicted_label_idx]
            predicted_label = self.labels[predicted_label_idx]

            print(predicted_label)

            # for each task, return classification results in the form of "choices" pre-annotations
            predictions.append(
                {
                    "result": [
                        {
                            "from_name": self.from_name,
                            "to_name": self.to_name,
                            "type": "choices",
                            "value": {"choices": [predicted_label]}
                        }
                    ],
                    "score": float(predicted_label_score)
                }
            )
        print(predictions[0])
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
