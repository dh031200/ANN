# Kaggle titanic
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Data Preparation
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
# y_test = pd.read_csv('./titanic/gender_submission.csv')

# Drop missing values
x_train = train.dropna()
# x_test = test.dropna()
x_test = test.replace(pd.NA, 0)

# Split y_train from train
y_train = x_train.Survived

# Drop unnecessary features
x_train = x_train.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])
x_test = x_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
# y_test = y_test.drop(columns=['PassengerId'])

# Label Encoding for Sex, Embarked
embarked_le = preprocessing.LabelEncoder()
sex_le = preprocessing.LabelEncoder()
x_train.Embarked = embarked_le.fit_transform(x_train.Embarked)
x_train.Sex = sex_le.fit_transform(x_train.Sex)
x_test.Embarked = embarked_le.transform(x_test.Embarked)
x_test.Sex = sex_le.transform(x_test.Sex)

# # Predict NA Age
# y_train_Age = x_train.Age
# x_train_Age = x_train.drop(columns=['Age'])
# Scaler_age = preprocessing.MinMaxScaler()
# Scaler_age.fit(x_train_Age)
# x_train_Age = Scaler_age.transform(x_train_Age)


# Data scaling
Scaler = preprocessing.MinMaxScaler()
Scaler.fit(x_train)
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)

# Split Validation data
x_val = x_train[:100]
x_partial_train = x_train[100:]
y_val = y_train[:100]
y_partial_train = y_train[100:]

# x_train_Age_val = x_train_Age[:100]
# y_train_Age_val = y_train_Age[:100]
# x_train_Age = x_train_Age[100:]
# y_train_Age = y_train_Age[100:]

# Reset index
y_partial_train = y_partial_train.values
y_val = y_val.values


# y_test = y_test.values


def build_model(act, neurons=10):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation=act, input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def model_fit(model):
    model.fit(x_train, y_train, epochs=200, batch_size=10,
              validation_data=(x_val, y_val),
              callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
              verbose=1)


model = build_model('relu')
model_fit(model)
result = model.predict(x_test).reshape(test.shape[0],)
for i in range(len(result)):
    if result[i] > 0.5:
        result[i] = 1
    else:
        result[i] = 0
result = result.astype('int64')
# result = result.reshape(result.shape[0],)

output = pd.DataFrame({'PassengerId': test.PassengerId,
                       'Survived': result})
output.to_csv('titanic.csv', index=False)

print(result)
# print("\n---- Test ----")
# for i in range(5):
#     model = build_model('relu')
#     model_fit(model)
#     # eval_train = model.evaluate(x_train, y_train, verbose=0)
#     # eval_test = model.evaluate(x_val, y_val, verbose=0)
#     # print("Trial #", i + 1)
#     # print(f'Training loss: {eval_train[0]:.5f}  Training accuracy: {eval_train[1]:.5f}')
#     # print(f'Training loss: {eval_test[0]:.5f}  Training accuracy: {eval_test[1]:.5f}')
