import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

root_path = "./chest_xray/"
train_path = root_path + "train/"
valid_path = root_path + "val/"
test_path = root_path + "test/"

train_data_generator = ImageDataGenerator(rescale=1. / 255)
valid_data_generator = ImageDataGenerator(rescale=1. / 255)
test_data_generator = ImageDataGenerator(rescale=1. / 255)
input_shape = [256, 256]
train_generator = train_data_generator.flow_from_directory(train_path, target_size=input_shape, batch_size=20,
class_mode='binary')
valid_generator = valid_data_generator.flow_from_directory(valid_path, target_size=input_shape, batch_size=20,
class_mode='binary')
test_generator = test_data_generator.flow_from_directory(test_path, target_size=input_shape, batch_size=20,
class_mode='binary')


def build_model():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    conv_base.trainable = False
    gap = layers.GlobalAvgPool2D()(conv_base.layers[-1].output)
    dropout_1 = layers.Dropout(0.25)(gap)
    dense1 = layers.Dense(512)(dropout_1)
    batchNorm = layers.BatchNormalization()(dense1)
    activation = layers.Activation(activation='relu')(batchNorm)
    dropout_2 = layers.Dropout(0.25)(activation)
    dense2 = layers.Dense(128, activation='relu')(dropout_2)
    dropout_3 = layers.Dropout(0.25)(dense2)
    y = layers.Dense(1, activation='sigmoid')(dropout_3)
    model = models.Model(conv_base.inputs, y)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
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


model = build_model()
model.summary()
history = model.fit(train_generator, epochs=100, verbose=0, validation_data=valid_generator)

model.save('HW3_Q3_256_pt.h5')

train_loss, train_acc = model.evaluate(train_generator,verbose=0)
test_loss, test_acc = model.evaluate(test_generator,verbose=0)

print('train acc: ', train_acc)
print('train loss: ', train_loss)
print('test acc: ', test_acc)
print('test loss: ', test_loss)

plt.clf()
plot_loss(history)
plt.savefig('Q3_256_pt_loss.png')
plt.clf()
plot_acc(history)
plt.savefig('Q3_256_pt_accuracy.png')

# --- fine tuning ---
tf_model = models.load_model('HW3_Q3_256_pt.h5')
for layer in tf_model.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True

tf_model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-5), loss='binary_crossentropy',
metrics=['accuracy'])

tf_history = tf_model.fit(train_generator, epochs=100, validation_data=valid_generator, verbose=0)

model.save('HW3_Q3_256_ft.h5')

train_loss, train_acc = tf_model.evaluate(train_generator,verbose=0)
test_loss, test_acc = tf_model.evaluate(test_generator,verbose=0)

print('train acc: ', train_acc)
print('train loss: ', train_loss)
print('test acc: ', test_acc)
print('test loss: ', test_loss)

plt.clf()
plot_loss(tf_history)
plt.savefig('Q3_256_ft_loss.png')
plt.clf()
plot_acc(tf_history)
plt.savefig('Q3_256_ft_accuracy.png')