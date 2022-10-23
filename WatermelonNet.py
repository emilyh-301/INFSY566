from tensorflow import keras
from keras import layers

class WatermelonNet:

    def __init__(self, num_classes):
        self.num_classes = num_classes

        self.model = keras.Sequential([
            layers.Conv2D(filters=6, kernel_size=(2, 2), activation='sigmoid', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=12, kernel_size=(2, 2), activation='sigmoid'),
            #layers.Dropout(.25),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=12, kernel_size=(2,2), activation='sigmoid'),
            #layers.Dropout(.25),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(units=120, activation='relu'),
            layers.Dense(units=84, activation='relu'),
            layers.Dense(units=num_classes, activation='softmax')
        ])

