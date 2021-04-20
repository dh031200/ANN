from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Data preparation
L, W, H = X_train.shape
X_train = X_train.reshape((-1, W * H))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((-1, W * H))
X_test = X_test.astype('float32') / 255

Y_train_ = to_categorical(Y_train)
Y_test_ = to_categorical(Y_test)

# model definition
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(W * H,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting
model.fit(X_train, Y_train_, epochs=20, batch_size=128, validation_split=0.2,
          callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

# Evaluation
test_loss, test_acc = model.evaluate(X_test, Y_test_)
print('test_acc: ', test_acc)
