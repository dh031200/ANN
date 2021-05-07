from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import models, layers, optimizers

root_path = "./chest_xray/"
train_path = root_path + "train/"
valid_path = root_path + "val/"
test_path = root_path + "test/"

generator = ImageDataGenerator(rescale=1. / 255)
input_shape = [128, 128]
train_generator = generator.flow_from_directory(train_path, target_size=input_shape, batch_size=20, class_mode='binary')
valid_generator = generator.flow_from_directory(valid_path, target_size=input_shape, batch_size=20, class_mode='binary')
test_generator = generator.flow_from_directory(test_path, target_size=input_shape, batch_size=20, class_mode='binary')


def build_model():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    conv_base.trainable = False
    gap = layers.GlobalAvgPool2D()(conv_base.layers[-1].output)
    dense1 = layers.Dense(512)(gap)
    batchNorm = layers.BatchNormalization()(dense1)
    activation = layers.Activation(activation='relu')(batchNorm)
    dense2 = layers.Dense(128, activation='relu')(activation)
    y = layers.Dense(1,activation='sigmoid')(dense2)
    model = models.Model(conv_base.inputs, y)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = build_model()
model.summary()

model.fit_generator(train_generator, epochs=100, verbose=1, validation_data=valid_generator)
