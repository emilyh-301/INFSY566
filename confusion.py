from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow import keras
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_test = x_test / 255
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def con_matrix1(preds, labels):
    print(confusion_matrix(labels, preds, normalize='pred'))

def con_matrix2(preds, labels, title):
    preds = np.argmax(preds, axis=1)
    y_test = np.argmax(labels, axis=1)
    # Create confusion matrix and normalizes it over predicted (columns)
    result = confusion_matrix(y_test, preds, normalize='pred')
    plot(result, title)

def plot(array, title):
    df_cm = pd.DataFrame(array, index=class_names, columns=class_names)
    # plt.figure(figsize=(10,10))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.savefig(title)

plt.show()

model = keras.models.load_model('alexNet-model')
preds = model.predict(x_test)
con_matrix2(preds, y_test, 'alex-confusion')
print('\n\n')

model = keras.models.load_model('leNet5_model')
preds = model.predict(x_test)
con_matrix2(preds, y_test, 'lenet-confusion')
print('\n2\n\n')

model = keras.models.load_model('watermelon-model')
preds = model.predict(x_test)
con_matrix2(preds, y_test, 'watermelon-confusion')
print('\n\n')
