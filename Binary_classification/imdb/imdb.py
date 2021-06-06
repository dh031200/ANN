import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# data loading (most frequent 10000 words only)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# To check the contents of the review
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 0,1,2 -> 'padding', 'start of sequence', and 'unknown'
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# Similar to multi-hot encoding
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Vectorize
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# Model definition
def build_model():
    x = layers.Input(shape=(10000,))
    dense1 = layers.Dense(16)(x)
    batchNorm1 = layers.BatchNormalization()(dense1)
    act1 = layers.Activation(activation='relu')(batchNorm1)

    dense2 = layers.Dense(16)(act1)
    batchNorm2 = layers.BatchNormalization()(dense2)
    act2 = layers.Activation(activation='relu')(batchNorm2)

    y = layers.Dense(1, activation='sigmoid')(act2)

    model = models.Model(x, y)
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])
    return model


# For plot
def plot_loss(h, title='loss'):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def plot_acc(h, title='accuracy'):
    plt.plot(h.history['binary_accuracy'])
    plt.plot(h.history['val_binary_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


# Split data for train, valid
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Model learning
model = build_model()
histroy = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=1)])

# Model evaluation & plot
results = model.evaluate(x_test, y_test)
print('test loss: ', results[0])
print('test accuracy: ', results[1])
plot_acc(histroy)
plt.savefig('imdb_acc.png')
plt.clf()
plot_loss(histroy)
plt.savefig('imdb_loss.png')

# Model prediction
print(model.predict(x_test))
