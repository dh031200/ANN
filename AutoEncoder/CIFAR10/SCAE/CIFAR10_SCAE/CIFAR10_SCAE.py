from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np


def Conv2D(filters, kernel_size, padding='same', activation='relu'):
    return layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)


class AE(models.Model):
    def __init__(self, org_shape=(32, 32, 3)):
        # Input
        original = layers.Input(shape=org_shape)

        # encoding-1
        x = Conv2D(4, (3, 3))(original)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        # encoding-2
        x = Conv2D(8, (3, 3))(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        # encoding-3
        z = Conv2D(1, (7, 7))(x)

        # decoding-1
        y = Conv2D(16, (3, 3))(z)
        y = layers.UpSampling2D((2, 2))(y)

        # decoding-2
        y = Conv2D(8, (3, 3))(y)
        y = layers.UpSampling2D((2, 2))(y)

        # decoding-3
        y = Conv2D(4, (3, 3))(y)

        # decoding & Output
        decoded = Conv2D(3, (3, 3), activation='sigmoid')(y)

        # Essential parts:
        super().__init__(original, decoded)
        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Optional parts:
        self.x = x
        self.y = y
        self.z = z
        self.summary()
        self.original = original

    def Encoder(self):
        return models.Model(self.original, self.z)

    def Decoder(self):
        z_shape = (self.z_dim[1],)
        z = layers.Input(shape=z_shape)
        h = self.layers[-6](z)
        h = self.layers[-5](h)
        h = self.layers[-4](h)
        h = self.layers[-3](h)
        h = self.layers[-2](h)
        h = self.layers[-1](h)
        return models.Model(z, h)


def data_load():
    (X_train, _), (X_test, _) = cifar10.load_data()

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    return X_train, X_test


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

    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = autoencoder.predict(X_test)

    n = 10
    plt.figure(figsize=(20, 6))

    for i in range(n):
        ax = plt.subplot(4, n, i + 1)
        plt.imshow(X_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + n)
        plt.stem(encoded_imgs[i].reshape(-1), use_line_collection=True)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + n * 2)
        plt.imshow(encoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i + 1 + n * 3)
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def main(epochs=20, batch_size=128):
    input_shape = [32, 32, 3]
    (X_train, X_test) = data_load()
    autoencoder = AE(input_shape)

    history = autoencoder.fit(X_train, X_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(X_test, X_test))

    plot_Loss(history)
    plt.savefig('scae_cifar10.loss.png')
    plt.clf()

    show_ae(autoencoder, X_test)
    plt.savefig('scae_cifar10.predicted.png')
    plt.show()


if __name__ == '__main__':
    main()
