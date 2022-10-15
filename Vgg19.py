from tensorflow import keras
from keras import layers
from tensorflow.keras.applications import VGG19
from keras.activations import softmax, relu, sigmoid

class Vgg19:

    def __init__(self, num_classes):
        self.num_classes = num_classes

        self.model = keras.Sequential(
            [
                VGG19(input_shape=(28, 28), include_top=False),  # weights='imagenet'
                layers.GlobalAveragePooling2D(),
                layers.Dense(500, activation=relu),
                layers.Dense(500, activation=relu),
                layers.Dense(4, activation=softmax)
            ])
