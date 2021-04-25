import plaidml.keras    # for gpu
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import models, layers, optimizers

import matplotlib.pyplot as plt

plaidml.keras.install_backend()     # for gpu
# data loading
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# data preprocessing
X_train = X_train / 255.0
X_test = X_test / 255.0
num_classes = 10
Y_train_ = to_categorical(Y_train, num_classes)
Y_test_ = to_categorical(Y_test, num_classes)

# model definition
L, W, H, C = X_train.shape
input_shape = [W, H, C]  # as a shape of image


def build_model():
    model = models.Sequential()
    # first convolutional layer
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                            input_shape=input_shape))
    # second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # max-pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Dropout layer
    model.add(layers.Dropout(0.25))
    # copy-and-paste 2 conv2d layers, max pooling layer, dropout layer
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    # Fully connected MLP
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    # Dropout layer
    model.add(layers.Dropout(0.5))
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    # compile
    model.compile(optimizer=optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


import time

start_time = time.time()
num_epochs = 100
model = build_model()
history = model.fit(X_train, Y_train_, validation_split=0.2,
                    epochs=num_epochs, batch_size=32, verbose=2)
train_loss, train_acc = model.evaluate(X_train, Y_train_, verbose=2)
test_loss, test_acc = model.evaluate(X_test, Y_test_, verbose=2)
print('train_acc:', train_acc)
print('test_acc:', test_acc)
print("elapsed time (in sec): ", time.time() - start_time)


# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


plot_loss(history)
plt.savefig('cifar10_loss.png')
plt.clf()
plot_acc(history)
plt.savefig('cifar10_accuracy.png')
