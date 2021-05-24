import pandas as pd
import numpy as np
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

# Load data
decode_data = pd.read_csv('./dataset/train_job/tags.csv', index_col='tagID')
user_tags = pd.read_csv('./dataset/train_job/user_tags.csv')
job_tags = pd.read_csv('./dataset/train_job/job_tags.csv')
train_data = pd.read_csv('./dataset/train_job/train.csv')
test_data = pd.read_csv('./dataset/test_job.csv')

# Set train data
train_y = np.array(train_data['applied'], dtype='float32')
train_data = train_data.drop(columns='applied')

# Get tagID for labeling
labels = np.unique(np.append(LabelEncoder().fit(user_tags['tagID']).classes_,
                             LabelEncoder().fit(job_tags['tagID']).classes_))

# Labeling
tag_label = LabelEncoder().fit(labels)
user_tags.tagID = tag_label.transform(user_tags['tagID'])
job_tags.tagID = tag_label.transform(job_tags['tagID'])

# Grouping
user_tags = user_tags.groupby('userID')['tagID'].apply(list)
job_tags = job_tags.groupby('jobID')['tagID'].apply(list)


# Vectorization (multi-hot-encoding)
def vectorize_sequences(sequences, dimension=labels.shape[0]):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Data preprocessing (common : count for same job between user and company)
train_user = [user_tags.loc[i] for i in train_data.userID]
train_job = [job_tags.loc[i] for i in train_data.jobID]
test_user = [user_tags.loc[i] for i in test_data.userID]
test_job = [job_tags.loc[i] for i in test_data.jobID]

# common = [len(list(set(user_tags[train_data.loc[i].userID]).intersection(job_tags[train_data.loc[i].jobID]))) for i in
#           range(len(train_data))]
# common = np.array(common) / 10

# Vectorize data
train_user = vectorize_sequences(train_user)
train_job = vectorize_sequences(train_job)
test_user = vectorize_sequences(test_user)
test_job = vectorize_sequences(test_job)

# i = 0
# b_size = 0
# while train_user.shape[1] != b_size:
#     if i >= train_user.shape[1]:
#         break
#     if not train_user[:, i].any() and not test_user[:, i].any() and not train_job[:, i].any() and not test_job[:, i].any():
#         b_size = train_user.shape[1]
#         train_user = np.delete(train_user, i, axis=1)
#         train_job = np.delete(train_job, i, axis=1)
#         test_user = np.delete(test_user, i, axis=1)
#         test_job = np.delete(test_job, i, axis=1)
#         i -= 1
#     i += 1

# # Way 1
# train_user *= 0.5
# train_job *= 0.5
# train_x = train_user + train_job
# for i in train_x:
#     for j in range(len(i)):
#         if i[j] == 0.5:
#             i[j] = 0

# Add common data
# train_x = [np.append(common[i], train_x[i]) for i in range(len(train_x))]
train_x = np.concatenate((train_user, train_job), axis=1)
test_x = np.concatenate((test_user, test_job), axis=1)


# train_x = np.array(train_x)


# Model structure
def build_model():
    x = layers.Input(shape=(train_x.shape[1],))
    dense1 = layers.Dense(128)(x)
    batch1 = layers.BatchNormalization()(dense1)
    act1 = layers.Activation(activation='relu')(batch1)

    # dense2 = layers.Dense(64)(act1)
    # batch2 = layers.BatchNormalization()(dense2)
    # act2 = layers.Activation(activation='relu')(batch2)

    # dense3 = layers.Dense(100)(act2)
    # batch3 = layers.BatchNormalization()(dense3)
    # act3 = layers.Activation(activation='relu')(batch3)

    # dense4 = layers.Dense(8)(act3)
    # batch4 = layers.BatchNormalization()(dense4)
    # act4 = layers.Activation(activation='relu')(batch4)
    # dense1 = layers.Dense(16)(x)
    # dense2 = layers.Dense(16)(dense1)
    # dropout1 = layers.Dropout(0.25)(act2)
    y = layers.Dense(1, activation='sigmoid')(act1)

    model = models.Model(x, y)
    opt = optimizers.SGD(learning_rate=1e-6, decay=1e-8, momentum=0.9, nesterov=True)
    # opt = optimizers.RMSprop(learning_rate=1e-6)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# For plot
def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


# # Normalization
# scaler = MinMaxScaler()
# scaler.fit(train_x)
# train_x = scaler.transform(train_x)

# Fit
model = build_model()
history = model.fit(train_x, train_y, epochs=500, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
plot_loss(history)
plt.savefig('loss.png')
