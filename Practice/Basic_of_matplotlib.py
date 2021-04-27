from keras.preprocessing import image
from keras.models import load_model
from keras import models
import matplotlib.pyplot as plt
import numpy as np

# Image preprocessing
img_path = '../Binary_classification/cats_and_dogs/datasets/cats_and_dogs_small/test/cats/cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
print(img_tensor.shape)

# Draw one of the image
plt.imshow(img_tensor[0])
plt.show()

# Load model (from Binary_class/cats_and_dogs)
model = load_model('../Binary_classification/cats_and_dogs/cats_and_dogs_small_1.h5')
model.summary()

# Define a new model, which is input to model's 0th ~ 7th layers' outputs
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(model.input, layer_outputs)

activations = activation_model.predict(img_tensor)
print(len(activations))
print(activations[0].shape)

# Response of the 19th filter of a first layer
plt.matshow(activations[0][0, :, :, 19], cmap='viridis')

plt.show()


def deprocess_image(X):
    X -= X.mean()
    X /= (X.std() + 1e-5)
    X *= 0.1
    X += 0.5
    X = np.clip(X, 0, 1)
    X *= 255
    X = np.clip(X, 0, 255).astype('uint8')
    return X


def draw_activation(activation, figure_name):
    images_per_row = 16
    n_features = activation.shape[-1]
    size = activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = activation[0, :, :, col * images_per_row + row]
            channel_image = deprocess_image(channel_image)
            display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image
    scale = 1./size
    plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
    plt.title(figure_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()


layer_names = [layer.name for layer in model.layers[:8]]

for figure_name, activation in zip(layer_names, activations):
    draw_activation(activation, figure_name)
