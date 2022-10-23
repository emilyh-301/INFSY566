# Emily Haigh

from tensorflow import keras

class AlexNet:
    '''
    creates an AlexNet arch CNN
    '''
    def __init__(self, num_classes):
        '''
        :param num_classes: number of classes for classification
        '''
        self.num_classes = num_classes

        self.model = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(6,6), strides=(2,2), activation='relu', input_shape=(28, 28)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=256, kernel_size=(4,4), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])



