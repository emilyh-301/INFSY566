# Emily Haigh

# Project for INFSY 566

import tensorflow as tf
from tensorflow import keras
import performance
import os
import time
import AlexNet

loss_funcs = []
opt_funcs = []
num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

alexNet_model = AlexNet.AlexNet(num_classes).model

alexNet_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.01), metrics=['accuracy'])
alexNet_model.summary()

history = alexNet_model.fit(x_train, y_train,
          epochs=50,
          #callbacks=callback  TODO: any callbacks?
          )

# To print the loss and accuracy graphs
performance.plot_performance(history, 'AlexNet-plot')

alexNet_model.evaluate(x_test)


