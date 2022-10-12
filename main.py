# Emily Haigh

# Project for INFSY 566

import tensorflow as tf
import performance
import AlexNet
import LeNet5

loss_funcs = []
opt_funcs = []
num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
input_size = (28, 28, 1)


alexNet_model = AlexNet.AlexNet(num_classes).model
alexNet_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.02), metrics=['accuracy'])
#alexNet_model.summary()
history = alexNet_model.fit(x_train, y_train, epochs=50, verbose=False)
# To print the loss and accuracy graphs
performance.plot_performance(history, 'AlexNet-plot')
score = alexNet_model.evaluate(x_test)
f = open("results.txt", "a")
f.write("Test Loss for AlexNet: " + str(score[0]) + "\nTest Accuracy for AlexNet: " + str(score[1]) + "\n")
f.close()

leNet5_model = LeNet5.LeNet5(num_classes).model
leNet5_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.02), metrics=['accuracy'])
#leNet5_model.summary()
history = leNet5_model.fit(x_train, y_train, epochs=50, verbose=False)
performance.plot_performance(history, 'LeNet5-plot')
score = leNet5_model.evaluate(x_test)
f = open("results.txt", "a")
f.write("Test Loss for LeNet5: " + str(score[0]) + "\nTest Accuracy for LeNet5: " + str(score[1]) + "\n")
f.close()




