from tensorflow.keras.applications import MobileNet, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

## modify and train as needed

def load_model():
    model = MobileNet(weights='imagenet',include_top=True)
    return model

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image





