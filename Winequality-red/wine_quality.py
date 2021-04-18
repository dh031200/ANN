import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, regularizers
from sklearn import preprocessing


# Model implementation
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),  # Regularized
                           activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                           activation='relu'))
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                           activation='relu'))
    # model.add(layers.Dense(512, activation='relu', input_shape=(train_data.shape[1],)))   # Non-regularized
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Data Preparation
data = np.loadtxt("winequality-red.csv", delimiter=",", skiprows=1, dtype=np.float32)  # Load data

# Split data and targets
targets = data[:, -1]
data = data[:, :-1]

# Split train data & test data
train_data = data[:1000]
train_targets = targets[:1000]

test_data = data[1000:]
test_targets = targets[1000:]

# Min-Max Normalization
scaler = preprocessing.MinMaxScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Split train data & validation data
val_data = train_data[:200]
partial_train_data = train_data[200:]

val_targets = train_targets[:200]
partial_train_targets = train_targets[200:]

# Train model
model = build_model()
history = model.fit(partial_train_data, partial_train_targets,
                    validation_data=(val_data, val_targets),
                    epochs=500, verbose=1)

# Evaluation
train_MAE = model.evaluate(partial_train_data, partial_train_targets)[1]
val_MAE = model.evaluate(val_data, val_targets)[1]
test_MAE = model.evaluate(test_data, test_targets)[1]

# Evaluation in the training set, validation set & test set
print("\ntrain_MAE: ", train_MAE)
print("val_MAE: ", val_MAE)
print("test_MAE: ", test_MAE)


# Learning curve
def plot_MAE(h, title="MAE"):
    plt.plot(h.history['mae'])
    plt.plot(h.history['val_mae'])
    plt.title(title)
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


# Plot
plot_MAE(history)
plt.show()

# Prediction in the test set
predict_test10 = model.predict(test_data)[:10].reshape(10)
gap = abs(test_targets[:10] - predict_test10)
print("\nPrediction 결과 : ", predict_test10)
print("실제 결과 : ", test_targets[:10])

# additional test
print("Prediction과 실제 결과의 차이 : ", end='')
print(gap, sep=' ')
print("오차 최대값: ", max(gap), "오차 최소값: ", min(gap), "오차 평균값: ", round(sum(gap) / 10, 6))
