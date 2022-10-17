from tensorflow import keras
from keras import layers


class LeNet5:
    '''
    creates a LeNet5 arch CNN
    '''
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = keras.Sequential([
            layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28)),
            layers.AveragePooling2D(),
            layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
            layers.AveragePooling2D(),
            layers.Flatten(),
            layers.Dense(units=120, activation='relu'),
            layers.Dense(units=84, activation='relu'),
            layers.Dense(units=num_classes, activation='softmax')
        ])

