from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras import backend as K
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Image preprocessing
from Practice.Basic_of_matplotlib import draw_activation

img_path = '../Binary_classification/cats_and_dogs/datasets/cats_and_dogs_small/test/cats/cat.1700.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = preprocess_input(img_tensor)


def gradCAM(model, x):
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])

    max_output = model.output[:, np.argmax(preds)]
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(max_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap, conv_layer_output_value, pooled_grads_value


# prediction
model = VGG16(weights='imagenet')
heatmap, conv_output, pooled_grads = gradCAM(model, img_tensor)

# Visualization
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('chapter5_4_3.jpg', superimposed_img)

# visualization of conv_output and pooled_grads
draw_no = range(256, 256 + 32, 1)
conv_activation = np.expand_dims(conv_output[:, :, draw_no], axis=0)
draw_activation(conv_activation, 'last_conv')
plt.matshow(pooled_grads[draw_no].reshape(-1, 16), cmap='viridis')
