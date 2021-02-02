import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_dir = 'convolutional_neural_network/dataset'

# Preprocessing the training dataset
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_dir = f'{dataset_dir}/training_set'
training_set = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='binary')

# Prerocessing the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)

test_dir = f'{dataset_dir}/test_set'
test_set = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32, class_mode='binary')

# Building the Convolutional Neural Network

# Initialising CNN
cnn = tf.keras.models.Sequential()

# Convolution and pooling layers
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening layer
cnn.add(tf.keras.layers.Flatten())

# Fully connected and output layers
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=25)


def prediction_result(result):
    if result[0][0] > 0.5:
        return 'dog'
    else:
        return 'cat'


# Make one prediction
test_image = image.load_img(f'{dataset_dir}/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
# training_set.class_indices  # Not sure why this was added

print(f'Cat or Dog (image 1): {prediction_result(result)}')

# Make another prediction
test_image = image.load_img(f'{dataset_dir}/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
# training_set.class_indices  # Not sure why this was added

print(f'Cat or Dog (image 2): {prediction_result(result)}')
