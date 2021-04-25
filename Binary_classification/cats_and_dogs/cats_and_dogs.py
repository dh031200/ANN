import os
import shutil
import time
import plaidml.keras  # for gpu
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from keras.applications import InceptionV3
import matplotlib.pyplot as plt
import pandas as pd

plaidml.keras.install_backend()  # for gpu


def mk_dir(path, n=False):
    os.mkdir(path)
    if n:
        os.mkdir(path + 'dogs')
        os.mkdir(path + 'cats')


# directory structure : datasets
#                           ├─  train   (Essential)
#                           └─  cats_and_dogs_small(big) ─┬─ train ─┬─  cats
#                                (Automatic copy)         │         └─  dogs
#                                                         ├─ valid ─┬─  cats
#                                                         │         └─  dogs
#                                                         └─  test ─┬─  cats
#                                                                   └─  dogs.

# Set path
root_path = './datasets/'
data_path = root_path + 'train/'
# small_path = root_path + 'cats_and_dogs_small/'
# train_path = small_path + 'train/'
# valid_path = small_path + 'validation/'
# test_path = small_path + 'test/'
big_path = root_path + 'cats_and_dogs_big/'
train_path = big_path + 'train/'
valid_path = big_path + 'validation/'
test_path = big_path + 'test/'
submission_path = root_path + 'test/'

# Check directory (create it if not)
if not os.path.isdir(big_path):
    mk_dir(big_path)  # Make directory & Data split
    mk_dir(train_path, True)
    mk_dir(valid_path, True)
    mk_dir(test_path, True)

    class_name = ['dog', 'cat']
    for name in class_name:
        f_names = [name + '.{}.jpg'.format(i) for i in range(8000)]  # split train data
        count = 0
        for f_name in f_names:
            src = os.path.join(data_path, f_name)
            dst = os.path.join(train_path + name + 's', f_name)
            shutil.copyfile(src, dst)
            count += 1
        print(name + ': ', count, ' train data moved')

        f_names = [name + '.{}.jpg'.format(i) for i in range(8000, 10000)]  # split valid data
        count = 0
        for f_name in f_names:
            src = os.path.join(data_path, f_name)
            dst = os.path.join(valid_path + name + 's', f_name)
            shutil.copyfile(src, dst)
            count += 1
        print(name + ': ', count, ' valid data moved')

        f_names = [name + '.{}.jpg'.format(i) for i in range(10000, 12500)]  # split test data
        count = 0
        for f_name in f_names:
            src = os.path.join(data_path, f_name)
            dst = os.path.join(test_path + name + 's', f_name)
            shutil.copyfile(src, dst)
            count += 1
        print(name + ': ', count, ' test data moved')

# Set image generators
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20, shear_range=0.1,
                                   width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

submission_data_gen = ImageDataGenerator(rescale=1. / 255)

# Data augmentation
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
    valid_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
submission_test_generator = submission_data_gen.flow_from_directory(
    submission_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode=None,
    shuffle=False
)

# model definition
input_shape = [150, 150, 3]  # as a shape of image


def build_model():
    model = models.Sequential()
    conv_base = InceptionV3(weights='imagenet',
                            include_top=False,
                            input_shape=input_shape)
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


# Pretrain start
start_time = time.time()
num_epochs = 50
model = build_model()
history = model.fit_generator(train_generator, epochs=num_epochs, steps_per_epoch=8000 // 32, verbose=1,
                              validation_data=validation_generator, validation_steps=2000 // 32)

# Saving the model
model.save('cats_and_dogs_pretrained.h5')

# Plot
plot_loss(history)
plt.savefig('cats_and_dogs_pretrain_loss.png')
plt.clf()
plot_acc(history)
plt.savefig('cats_and_dogs_pretrain_acc.png')

# Evaluation
train_loss, train_acc = model.evaluate_generator(train_generator, verbose=1)
test_loss, test_acc = model.evaluate_generator(test_generator, verbose=1)
print('train_acc:', train_acc)
print('test_acc:', test_acc)
print("elapsed time (in sec): ", time.time() - start_time)

# Predict
predict = model.predict_generator(submission_test_generator, verbose=1)
print("elapsed time (in sec): ", time.time() - start_time)

# Organize predict & make csv
predict_id = []
predict = predict.reshape(predict.shape[0])
for i in range(predict.shape[0]):
    predict_id.append(i + 1)
    if predict[i] > 0.5:
        predict[i] = 0
    else:
        predict[i] = 1

output = pd.DataFrame({'id': predict_id,
                       'label': predict})
output.to_csv('cats_and_dogs_pretrained.csv', index=False)

# Fine-tune start line -----
# Load InceptionV3
conv_inception = model.layers[0]

# Change trainable setting for Fine-tune top 2 inception blocks
for layer in conv_inception.layers[:249]:
    layer.trainable = False
for layer in conv_inception.layers[249:]:
    layer.trainable = True

# Compile
model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

# reset time & fit
start_time = time.time()
history = model.fit_generator(train_generator, epochs=num_epochs, steps_per_epoch=8000 // 32, verbose=1,
                              validation_data=validation_generator, validation_steps=2000 // 32)

# Plot
plot_loss(history)
plt.savefig('cats_and_dogs_final_loss.png')
plt.clf()
plot_acc(history)
plt.savefig('cats_and_dogs_final_acc.png')

# Predict test data
predict = model.predict_generator(submission_test_generator, verbose=1)
print("elapsed time (in sec): ", time.time() - start_time)

# Organize predict & make csv
predict_id = []
predict = predict.reshape(predict.shape[0])
for i in range(predict.shape[0]):
    predict_id.append(i + 1)
    if predict[i] > 0.5:
        predict[i] = 0
    else:
        predict[i] = 1

output = pd.DataFrame({'id': predict_id,
                       'label': predict})
output.to_csv('cats_and_dogs_final.csv', index=False)
