from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16

input_shape = [150, 150, 3]


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_model_M():
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(x)
    pool_1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = (layers.Conv2D(64, (3, 3), activation='relu')(pool_1))
    pool_2 = layers.MaxPooling2D((2, 2))(conv2)

    flatten = layers.Flatten()(pool_2)
    dense1 = layers.Dense(512, activation='relu')(flatten)
    y = layers.Dense(1, activation='sigmoid')(dense1)
    model = models.Model(x, y)

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def build_model_T():
    x = layers.Input(shape=(256,256,3))
    conv_base = VGG16(weights='imagenet', include_top=False,input_shape=(256,256,3))(x)
    model = models.Model(conv_base.input, conv_base.layers[-1].output)

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_s = build_model()
model_m = build_model_M()
model_t = build_model_T()
model_s.summary()
print("\n\n\n")
model_m.summary()
