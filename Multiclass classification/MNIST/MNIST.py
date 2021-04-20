from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# data loading
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# data preprocessing
W, H = X_train.shape[1:]
X_train = X_train.reshape(-1, W, H, 1) / 255.0
X_test = X_test.reshape(-1, W, H, 1) / 255.0
num_classes = 10
Y_train_ = to_categorical(Y_train, num_classes)
Y_test_ = to_categorical(Y_test, num_classes)

# model definition
input_shape = [W, H, 1]  # as a shape of image


def build_model():
    model = models.Sequential()
    # first convolutional layer
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                            input_shape=input_shape))
    # second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # max-pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Fully connected MLP
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    # compile
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
