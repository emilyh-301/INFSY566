# Emily Haigh

# Project for INFSY 566
# Note, in the code EmilyNet is called WatermelonNet

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
#tf.config.list_physical_devices("GPU")
#[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import WatermelonNet
import performance
import AlexNet
import LeNet5
import confusion

# load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
# normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0
input_size = (28, 28, 1)
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=0, mode='auto')

# constants
EPOCHS = 50
num_classes = 10
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Watermelon / EmilyNet
try:
    watermelon_model = WatermelonNet.WatermelonNet(num_classes).model
    watermelon_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # watermelon_model.summary()
    history = watermelon_model.fit(x_train, y_train, validation_split=.2, epochs=EPOCHS, verbose=True)
    performance.plot_performance(history, 'Watermelon-plot-28')
    score = watermelon_model.evaluate(x_test, y_test)
    f = open("results.txt", "a")
    f.write("Test Loss for WatermelonNet: " + str(score[0]) + "\nTest Accuracy for WatermelonNet: " + str(score[1]) + "\n")
    f.close()
    watermelon_model.save('watermelon-model')
except:
    f = open("results.txt", "a")
    f.write("Watermelon threw an error" + "\n")
    f.close()

# AlexNet
try:
    alexNet_model = AlexNet.AlexNet(num_classes).model
    alexNet_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # alexNet_model.summary()
    history = alexNet_model.fit(x_train, y_train, validation_split=.2, epochs=EPOCHS, verbose=True)
    # To print the loss and accuracy graphs
    performance.plot_performance(history, 'AlexNet-plot-190-80-noBN')
    score = alexNet_model.evaluate(x_test, y_test)
    f = open("results.txt", "a")

    f.write("Test Loss for AlexNet: " + str(score[0]) + "\nTest Accuracy for AlexNet: " + str(score[1]) + "\n")
    f.close()
    alexNet_model.save('alexNet-model')
except:
   print('alex net error')
   f = open("results.txt", "a")
   f.write("AlexNet threw an error" + "\n")
   f.close()

# LeNet
try:
    leNet5_model = LeNet5.LeNet5(num_classes).model
    leNet5_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # leNet5_model.summary()
    history = leNet5_model.fit(x_train, y_train, validation_split=.2, epochs=EPOCHS, verbose=False)

    performance.plot_performance(history, 'LeNet5-plot')
    score = leNet5_model.evaluate(x_test, y_test)
    f = open("results.txt", "a")
    f.write("Test Loss for LeNet5: " + str(score[0]) + "\nTest Accuracy for LeNet5: " + str(score[1]) + "\n")
    f.close()
    leNet5_model.save('leNet5_model')
except:
    f = open("results.txt", "a")
    f.write("LeNet threw an error" + "\n")
    f.close()

