from keras.applications import VGG16
from keras import backend as K
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model('../Binary_classification/cats_and_dogs/cats_and_dogs_small_2.h5')
# model = VGG16(weights='imagenet', include_top=False)
model.summary()


def deprocess_image(X):
    X -= X.mean()
    X /= (X.std() + 1e-5)
    X *= 0.1
    X += 0.5
    X = np.clip(X, 0, 1)
    X *= 255
    X = np.clip(X, 0, 255).astype('uint8')
    return X


def generate_patterns(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128
    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)


def draw_filters(layer_name, size=150):
    images_per_row = 16
    n_cols = 9
    results = np.zeros((size * n_cols, images_per_row * size, 3), dtype='uint8')
    for col in range(n_cols):
        for row in range(images_per_row):
            print((col, row))
            filter_image = generate_patterns(layer_name, 16*col+row, size=size)
            results[col * size:(col + 1) * size, row * size:(row + 1) * size, :] = filter_image
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.title(layer_name)
    plt.show()
    return filter_image, results


layer_name = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']
for i in layer_name:
    print(i)
    filter_image, results = draw_filters(i)
