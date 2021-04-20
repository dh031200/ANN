import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing

# 2015270434 김동훈 HW2

# Data Preparation
data = np.loadtxt("breast-cancer-wisconsin.data", delimiter=",", dtype=np.float32)  # load data
data = data[:, 1:]  # Drop first column

# Attribute Domain
# x : 1 - 10
# y : 2 for benign, 4 for malignant

# Split input & output
x = data[:, :-1]
y = data[:, -1] * 0.5 - 1  # Set the output variable 0: benign 1: malignant

# Split train & test
x_test = x[:100]
x_train = x[100:]

# Normalize the input variables
Scaler = preprocessing.StandardScaler()
Scaler.fit(x_train)
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)

# Data split
# next 100 for validation and others for train
x_val = x_train[:100]
x_train = x_train[100:]

y_test = y[:100]
y_val = y[100:200]
y_train = y[200:]


# Model implementation
def build_model(act, neurons=10):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation=act, input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def model_fit(model):
    model.fit(x_train, y_train, epochs=200, batch_size=10,
              validation_data=(x_val, y_val),
              callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
              verbose=0)


def print_mean_sd(tr_l, tr_a, te_l, te_a):
    print(f'train_loss: {np.mean(tr_l):.5f} ± {np.std(tr_l):.5f}')
    print(f'train_acc: {np.mean(tr_a):.5f} ± {np.std(tr_a):.5f}')
    print(f'test_loss: {np.mean(te_l):.5f} ± {np.std(te_l):.5f}')
    print(f'test_acc: {np.mean(te_a):.5f} ± {np.std(te_a):.5f}\n')


# Q2 Train 5 time and print each acc, loss
print("\n---- Q2 ----")
for i in range(5):
    model = build_model('relu')
    model_fit(model)
    eval_train = model.evaluate(x_train, y_train, verbose=0)
    eval_test = model.evaluate(x_test, y_test, verbose=0)
    print("Trial #", i + 1)
    print(f'Training loss: {eval_train[0]:.5f}  Training accuracy: {eval_train[1]:.5f}')
    print(f'Test loss: {eval_test[0]:.5f}  Test accuracy: {eval_test[1]:.5f}')

# Q3 Train models by different activation function
print("\n---- Q3 ----")
funcs = [None, 'relu', 'sigmoid', 'tanh']
for i in funcs:
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    for j in range(10):
        model = build_model(i)
        model_fit(model)
        eval_train = model.evaluate(x_train, y_train, verbose=0)
        eval_test = model.evaluate(x_test, y_test, verbose=0)
        train_loss.append(eval_train[0])
        train_acc.append(eval_train[1])
        test_loss.append(eval_test[0])
        test_acc.append(eval_test[1])
    print("activation function: ", i)
    print_mean_sd(train_loss, train_acc, test_loss, test_acc)

# Q4 Train models by different number of neurons
print("\n---- Q4 ----")
num_of_neurons = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
for i in num_of_neurons:
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    for j in range(5):
        model = build_model('relu', i)
        model_fit(model)
        eval_train = model.evaluate(x_train, y_train, verbose=0)
        eval_test = model.evaluate(x_test, y_test, verbose=0)
        train_loss.append(eval_train[0])
        train_acc.append(eval_train[1])
        test_loss.append(eval_test[0])
        test_acc.append(eval_test[1])
    print("number of hidden neurons: ", i)
    print_mean_sd(train_loss, train_acc, test_loss, test_acc)

# Q E2 Get weights and bias
print("\n---- Q E2 ----")
model = models.Sequential()
model.add(layers.Dense(1, activation='sigmoid', input_shape=(x_train.shape[1],)))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model_fit(model)
w = model.get_weights()[0]
w = w.reshape(w.shape[0])
b = model.get_weights()[1]
print("weight: ", w)
print("bias: ", b)
