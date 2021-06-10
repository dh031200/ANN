from matplotlib import pyplot as plt
from tensorflow.keras import datasets, models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, UpSampling2D


class DATA():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        self.x_train_in = DATA.RGB2gray(x_train)
        self.x_test_in = DATA.RGB2gray(x_test)
        self.x_train_out = x_train
        self.x_test_out = x_test
        img_rows, img_cols, n_ch = self.x_train_in.shape[1:]
        self.input_shape = (img_rows, img_cols, n_ch)

    def RGB2gray(X):
        R = X[..., 0:1]
        G = X[..., 1:2]
        B = X[..., 2:3]
        return 0.299 * R + 0.587 * G + 0.114 * B


class UNET(models.Model):
    def conv(x, n_f, mp_flag=True):
        x = MaxPooling2D((2, 2), padding='same')(x) if mp_flag else x
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Dropout(0.05)(x)
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        return x

    def deconv_unet(x, n_f):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        return x

    def __init__(self, org_shape):
        # Input
        original = Input(shape=org_shape)

        # Encoding
        c1 = UNET.conv(original, 16, mp_flag=False)
        c2 = UNET.conv(c1, 32)

        # Encoded vector
        encoded = UNET.conv(c2, 64)

        # Decoding
        x = UNET.deconv_unet(encoded, 32)
        y = UNET.deconv_unet(x, 16)

        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(y)

        super().__init__(original, decoded)
        self.compile(optimizer='adadelta', loss='mse')


def plot_loss(h, title="Loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def show_images(data, unet):
    x_test_in = data.x_test_in
    x_test_out = data.x_test_out
    decoded_imgs = unet.predict(x_test_in)

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test_in[i, :, :, 0], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n * 2)
        plt.imshow(x_test_out[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def main(epochs=100, batch_size=128, fig=True):
    data = DATA()
    unet = UNET(data.input_shape)
    unet.summary()

    history = unet.fit(data.x_train_in, data.x_train_out,
                       epochs=epochs, batch_size=batch_size,
                       shuffle=True, validation_data=(data.x_test_in, data.x_test_out))

    if fig:
        plot_loss(history)
        plt.savefig('SCAE_loss.png')
        plt.show()
        plt.clf()
        show_images(data, unet)
        plt.savefig('SCAE_pred.png')
        plt.show()


if __name__ == '__main__':
    main()
