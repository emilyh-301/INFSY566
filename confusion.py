from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow import keras
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# generating the confusion matrices

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_test = x_test / 255
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def con_matrix2(preds, labels, title):
    predictions = get_pred(preds)
    # Make confusion matrix and normalize it over the predictions (columns)
    result = confusion_matrix(y_true=labels, y_pred=predictions, normalize='pred')
    f = open(title + ".txt", "a")
    for row in range(len(result)):
        for col in range(10):
            result[row][col] = round(result[row][col], 2)
            f.write(str(result[row][col]) + '\t')
        f.write('\n')
    f.close()
    plot(result, title)

def plot(array, title):
    df_cm = pd.DataFrame(array, index=class_names, columns=class_names)
    plt.figure(figsize=(100,100))
    sn.set(font_scale=1) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}) # font size
    plt.savefig(title)

def get_pred(preds):
    result = []
    for x in preds:
        a = np.where(x == x.max())
        result.append(a[0][0])
    return result

model = keras.models.load_model('alexNet-model')
preds = model.predict(x_test)
con_matrix2(preds, y_test, 'alex-confusion')
print('\n\n')

model = keras.models.load_model('leNet5_model')
preds = model.predict(x_test)
con_matrix2(preds, y_test, 'lenet-confusion')
print('\n\n')

model = keras.models.load_model('watermelon-model')
preds = model.predict(x_test)
con_matrix2(preds, y_test, 'watermelon-confusion')
print('\n\n')
