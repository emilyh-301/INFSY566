from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_test = x_test / 255

def con_matrix1(preds, labels):
    print(confusion_matrix(labels, preds, normalize='pred'))

def con_matrix2(preds, labels):
    preds = np.argmax(preds, axis=1)
    y_test = np.argmax(labels, axis=1)
    # Create confusion matrix and normalizes it over predicted (columns)
    result = confusion_matrix(y_test, preds, normalize='pred')
    print(result)

model = keras.models.load_model('alexNet-model')
preds = model.predict(x_test)
print('1\n')
con_matrix1(preds, y_test)
print('\n2\n\n')

model = keras.models.load_model('leNet5_model')
preds = model.predict(x_test)
print('1\n')
con_matrix1(preds, y_test)
print('\n2\n\n')

model = keras.models.load_model('watermelon-model')
preds = model.predict(x_test)
print('1\n')
con_matrix1(preds, y_test)
print('\n2\n\n')

