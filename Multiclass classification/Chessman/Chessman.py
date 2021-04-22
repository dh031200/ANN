import os
import shutil
import time

from matplotlib import pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set path
data_path = './Chessman-image-dataset/Chess/'
train_path = './Chessman-image-dataset/train/'
test_path = './Chessman-image-dataset/test/'
valid_path = './Chessman-image-dataset/valid/'

class_list = os.listdir(data_path)
class_list.sort()


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        for name in class_list:
            os.mkdir(path + name)


# Data split
if not os.path.isdir(train_path):
    make_dir(train_path)
    make_dir(test_path)
    make_dir(valid_path)

    for class_name in class_list:
        img_list = os.listdir(data_path + class_name)
        img_num = len(img_list)
        train_index = int(img_num * 0.64)
        valid_index = int(img_num * 0.8)
        for img in img_list[:train_index]:  # 64% for train
            shutil.copy(data_path + class_name + "/" + img,
                        train_path + class_name + "/" + img)
        for img in img_list[train_index:valid_index]:  # 16% for valid
            shutil.copy(data_path + class_name + "/" + img,
                        valid_path + class_name + "/" + img)
        for img in img_list[valid_index:]:  # 20% for test
            shutil.copy(data_path + class_name + "/" + img,
                        test_path + class_name + "/" + img)

print("Data segmentation is complete.")

# Set image generators
train_data_gen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_data_gen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

test_data_gen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_data_gen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

valid_data_gen = ImageDataGenerator(rescale=1. / 255)
valid_generator = valid_data_gen.flow_from_directory(
    valid_path,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

# Check batch size
for data_batch, labels_batch in train_generator:
    print("batch data size: ", data_batch.shape)
    print("batch label size: ", labels_batch.shape)
    break

# Model definition
input_shape = [150, 150, 3]


def build_model():
    model = models.Sequential()
    conv_base = VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=input_shape)
    conv_base.trainable = False
    model.add(conv_base)
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)) # delete for transfer learning
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Pretrained model
start_time = time.time()
model = build_model()
history = model.fit_generator(train_generator, epochs=50, validation_data=valid_generator,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3),
                               ModelCheckpoint('Chessman_pretrained.h5', monitor='val_loss', verbose=1,
                                               save_best_only=True, mode='min')])

# Evaluation (pretrained model)
train_loss, train_acc = model.evaluate(train_generator)
test_loss, test_acc = model.evaluate(test_generator)
print('train_acc: ', train_acc)
print('test_acc: ', test_acc)
print('elapsed time (in sec): ', time.time() - start_time)

# transfer learning
tf_model = models.load_model('Chessman_pretrained.h5')
conv_base = tf_model.layers[0]
for layer in conv_base.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True

tf_model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                 loss='categorical_crossentropy', metrics=['accuracy'])

tf_history = tf_model.fit(train_generator, epochs=50, validation_data=valid_generator,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                                     ModelCheckpoint('Chessman.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                                     mode='min')])

# Evaluation (Transfer model)
train_acc = tf_model.evaluate(train_generator)[1]
test_acc = tf_model.evaluate(test_generator)[1]
print('train_acc: ', train_acc)
print('test_acc: ', test_acc)
print('elapsed time (in sec): ', time.time() - start_time)

# Evaluation (Best model)
best_model = models.load_model('Chessman.h5')
train_loss, train_acc = best_model.evaluate(train_generator)
test_loss, test_acc = best_model.evaluate(test_generator)
print('train_acc: ', train_acc)
print('train_loss: ', train_loss)
print('test_acc: ', test_acc)
print('test_loss: ', test_loss)
print('elapsed time (in sec): ', time.time() - start_time)


# For plot
def plot_Accuracy(h, title="Accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


# Plot
plot_Accuracy(tf_history)
plt.show()
