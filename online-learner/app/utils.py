from abc import ABC, abstractmethod
import numpy as np
from io import BytesIO
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from vowpalwabbit import pyvw

class FeatureProcessor(ABC):
    def __init__(self,
                 batch_size,
                 flattened_size,
                 feature_file_format
                ):
        super().__init__()
        self.batch_size = batch_size
        self.flattened_size = flattened_size
        self.feature_file_format = feature_file_format

    @staticmethod
    def create(
        feature_processor_name,
        batch_size,
        feature_file_format
    ):
        if feature_processor_name == "MobileNet":
            flattened_size = 7 * 7 * 1280
            return MobileNetFeatureProcessor(batch_size, flattened_size, feature_file_format)

    def initialize_output_processor(self, labels, feature_file_path):
        if self.feature_file_format == "npy":
            self.output_processor = NpyOutput(labels,
            self.flattened_size, self.batch_size, feature_file_path)
        elif self.feature_file_format == "csv":
            self.output_processor = CsvOutput(labels,
            self.batch_size, feature_file_path)

    @abstractmethod
    def process_image(self):
        pass

    @abstractmethod
    def create_features(self):
        pass

class MobileNetFeatureProcessor(FeatureProcessor):
    def __init__(self, batch_size, flattened_size, feature_file_format):
        super().__init__(batch_size, flattened_size, feature_file_format)
        self.model = MobileNetV2(weights="imagenet",
                                 include_top=False, 
                                 input_shape=(224, 224, 3)
                    )
        self.name = "mobile"

    def process_image(self, image_path):
        image = load_img(
            image_path,
            target_size=(224, 224)
        )
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        return image

    def process_in_memory_image(self, image, dsize=(224, 224)):
        # See: https://stackoverflow.com/questions/55873174/how-do-i-return-an-image-in-fastapi
        image = cv2.resize(image, dsize=dsize)
        print('... about to convert to array')        
        image = img_to_array(image)        
        image = np.expand_dims(image, axis=0)
        print('... about to preprocess')                
        image = imagenet_utils.preprocess_input(image)
        print('about to return image...')
        return image
    
    def create_features(self, batch_images):
        features = self.model.predict(
            batch_images,
            batch_size= self.batch_size
        )
        features = features.reshape(
            (features.shape[0], self.flattened_size)
        )
        return features

    def create_features_for_an_image(self, the_image):
        print('about to call mobilenet model directly')
        features = self.model.predict(
            the_image
        )
        print('reshaping mobilenet featurs...')
        features = features.reshape(
            (features.shape[0], self.flattened_size)
        )
        return features

def construct_vw_example(label, features):
    prefix = ''
    if label is not None:
        prefix = f"{label} | " 
    the_feature_vector = np.array2string(
        features,
        precision=4,
        separator=' ',
        suppress_small=False
    )[2:-2] # kinda shocked this leaves in brackets
    vw = get_online_learner()
    return vw.example(
        f"{prefix}{the_feature_vector}"
    )

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

# so ideally i'd have a pool of these, use this with
# Celery or something that matched sessions to a pool of
# vowpalwabbits
vw = pyvw.vw(quiet=True)

def get_online_learner(session_key=None):
    learner = None
    if session_key is None:
        # for now
        learner = vw

    return learner



