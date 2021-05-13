from keras.models import load_model
from keras import models
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# Image preprocessing
img_path = '../datasets/cats_and_dogs_small/test/cats/cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
print(img_tensor.shape)

# Draw one of the image
# plt.imshow(img_tensor[0])
# plt.show()


model = load_model('cats_and_dogs_small_2.h5')
model.summary()

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(model.input, layer_outputs)

activations = activation_model.predict(img_tensor)
print(len(activations))
print(activations[0].shape)

plt.matshow(activations[6][0, :, :, 3], cmap='viridis')
plt.show()