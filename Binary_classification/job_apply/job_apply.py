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
common = [len(list(set(user_tags[train_data.loc[i].userID]).intersection(job_tags[train_data.loc[i].jobID]))) for i in
          range(len(train_data))]
common = np.array(common) / 10

# Vectorize data
train_user = vectorize_sequences(train_user)
train_job = vectorize_sequences(train_job)
train_user *= 0.5
train_job *= (-1)
train_x = train_user + train_job
for i in train_x:
    for j in range(len(i)):
        if i[j] == -0.5:
            i[j] = 1

# Add common data
train_x = [np.append(common[i], train_x[i]) for i in range(len(train_x))]


# Model structure
def build_model():
    x = layers.Input(shape=(train_x.shape[1],))
    dense1 = layers.Dense(128)(x)
    batch1 = layers.BatchNormalization()(dense1)
    act1 = layers.Activation(activation='relu')(batch1)

    dense2 = layers.Dense(64)(act1)
    batch2 = layers.BatchNormalization()(dense2)
    act2 = layers.Activation(activation='relu')(batch2)

    dense3 = layers.Dense(16)(act2)
    batch3 = layers.BatchNormalization()(dense3)
    act3 = layers.Activation(activation='relu')(batch3)

    # dropout1 = layers.Dropout(0.25)(act2)
    y = layers.Dense(1, activation='sigmoid')(act3)

    model = models.Model(x, y)

    model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-5, momentum=0.05), loss='binary_crossentropy',
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


# Normalization
scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)

# Fit
model = build_model()
history = model.fit(train_x, train_y, epochs=100, batch_size=16, validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
plot_loss(history)
plt.savefig('loss.png')
