import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, datasets


class UNET(models.Model):
    def conv(x, n_f, mp_flag=True):
        x = layers.MaxPooling2D((2, 2), padding='same')(x) if mp_flag else x
        x = layers.Conv2D(n_f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)  # learning booster
        x = layers.Activation('tanh')(x)
        x = layers.Dropout(0.05)(x)  # preventing overfitting
        x = layers.Conv2D(n_f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)  # learning booster
        x = layers.Activation('tanh')(x)
        return x

    def deconv_unet(x, e, n_f):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate(axis=3)([x, e])  # most important
        x = layers.Conv2D(n_f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)  # learning booster
        x = layers.Activation('tanh')(x)
        x = layers.Conv2D(n_f, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)  # learning booster
        x = layers.Activation('tanh')(x)
        return x

    def __init__(self, org_shape):
        # Input
        original = layers.Input(shape=org_shape)

        # Encoding
        c1 = UNET.conv(original, 128, mp_flag=False)
        c2 = UNET.conv(c1, 256)

        # Encoded vector
        encoded = UNET.conv(c2, 64)

        # Decoding
        x = UNET.deconv_unet(encoded, c2, 32)
        y = UNET.deconv_unet(x, c1, 16)

        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(y)

        super().__init__(original, decoded)
        self.compile(optimizer='adadelta', loss='mse',metrics='accuracy')


class DATA():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        self.x_train_in = x_train
        self.x_test_in = x_test
        self.x_train_out = x_train
        self.x_test_out = x_test

        img_rows, img_cols, n_ch = self.x_train_in.shape[1:]
        self.input_shape = (img_rows, img_cols, n_ch)


def show_images(data, unet):
    x_test_in = data.x_test_in
    x_test_out = data.x_test_out
    decoded_imgs = unet.predict(x_test_in)

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test_in[i])
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


def main(in_ch=1, epochs=10, batch_size=512, fig=True):
    data = DATA()
    unet = UNET(data.input_shape)
    unet.summary()

    history = unet.fit(data.x_train_in, data.x_train_out,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(data.x_test_in, data.x_test_out))

    unet.save('unet.h5')
    if fig:
        plot_Loss(history)
        plt.savefig('unet_8x8x.loss.png')
        plt.clf()
        plot_Accuracy(history)
        plt.savefig('unet_8x8x.acc.png')
        plt.clf()
        show_images(data, unet)
        plt.savefig('unet_8x8x.pred.png')


if __name__ == '__main__':
    import argparse
    from distutils import util

    parser = argparse.ArgumentParser(description='UNET for Cifar-10')
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128')
    parser.add_argument('--fig', type=lambda x: bool(util.strtobool(x)),
                        default=True, help='flag to show figures (default: True)')
    args = parser.parse_args()
    print("args:", args)

    main(epochs=100, batch_size=256, fig=True)
    # main(epochs=args.epochs, batch_size=args.batch_size, fig=args.fig)
