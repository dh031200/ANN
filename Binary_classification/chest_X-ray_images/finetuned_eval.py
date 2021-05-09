from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

root_path = "./chest_xray/"
train_path = root_path + "train/"
valid_path = root_path + "val/"
test_path = root_path + "test/"

train_data_generator = ImageDataGenerator(rescale=1. / 255)
valid_data_generator = ImageDataGenerator(rescale=1. / 255)
test_data_generator = ImageDataGenerator(rescale=1. / 255)
input_shape = [512, 512]
train_generator = train_data_generator.flow_from_directory(train_path, target_size=input_shape, batch_size=10,
                                                           class_mode='binary')
valid_generator = valid_data_generator.flow_from_directory(valid_path, target_size=input_shape, batch_size=10,
                                                           class_mode='binary')
test_generator = test_data_generator.flow_from_directory(test_path, target_size=input_shape, batch_size=10,
                                                         class_mode='binary')

model = models.load_model('./Q3/HW3_Q3_512_ft.h5')

train_loss, train_acc = model.evaluate(train_generator, verbose=1)
test_loss, test_acc = model.evaluate(test_generator, verbose=1)

print('train acc: ', train_acc)
print('train loss: ', train_loss)
print('test acc: ', test_acc)
print('test loss: ', test_loss)
