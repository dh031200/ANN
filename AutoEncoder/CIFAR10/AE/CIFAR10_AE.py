from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np


class AE(models.Model):
    def __init__(self, x_nodes=784, z_dim=36):
        x_shape = (x_nodes,)
        x = layers.Input(shape=x_shape)
        z = layers.Dense(z_dim, activation='relu')(x)
        y = layers.Dense(x_nodes, activation='sigmoid')(z)

        # Essential parts
        super().__init__(x, y)
        self.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # Optional parts: they are for Encoder and Decoder
        self.x = x
        self.z = z
        self.z_dim = z_dim

    def Encoder(self):
        return models.Model(self.x, self.z)

    def Decoder(self):
        z_shape = (self.z_dim,)
        z = layers.Input(shape=z_shape)
        y_layer = self.layers[-1]
        y = y_layer(z)
        return models.Model(z, y)


# Data load
def data_load():
    (X_train, _), (X_test, _) = cifar10.load_data()

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    return X_train, X_test


# For plot
def plot_Accuracy(h, title="Accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


# For plot
def plot_Loss(h, title="Loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def show_ae(autoencoder, X_test):
    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()

    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X_test[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.stem(encoded_imgs[i].reshape(-1), use_line_collection=True)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def main():
    x_nodes = 32 * 32 * 3
    z_dim = 36
    (X_train, X_test) = data_load()
    autoencoder = AE(x_nodes, z_dim)
    autoencoder.summary()
    history = autoencoder.fit(X_train, X_train,
                              epochs=20,
                              batch_size=256,
                              shuffle=True,
                              validation_data=(X_test, X_test))

    plot_Loss(history)
    plt.savefig('ae_cifar10_36.loss.png')
    plt.clf()
    plot_Accuracy(history)
    plt.savefig('ae_cifar10_36.acc.png')

    show_ae(autoencoder, X_test)
    plt.savefig('ae_cifar10_36.predicted.png')
    plt.show()


if __name__ == '__main__':
    main()
