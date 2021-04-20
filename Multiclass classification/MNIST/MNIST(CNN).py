import time

from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Data preparation
W, H = X_train.shape[1:]
X_train = X_train.reshape(-1, W, H, 1) / 255.0
X_test = X_test.reshape(-1, W, H, 1) / 255.0
num_classes = 10  # 0 ~ 9
Y_train_ = to_categorical(Y_train, num_classes)
Y_test_ = to_categorical(Y_test, num_classes)

# model definition
input_shape = [W, H, 1]  # as a shape of image


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                            input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Learning model
start_time = time.time()
model = build_model()
history = model.fit(X_train, Y_train_, validation_split=0.2,
                    epochs=20, batch_size=100, verbose=1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=1)])
train_loss, train_acc = model.evaluate(X_train, Y_train_)
test_loss, test_acc = model.evaluate(X_test, Y_test_)
print('train_acc: ', train_acc)
print('test_acc: ', test_acc)
print('elapsed time (in sec): ', time.time() - start_time)
