#!/usr/bin/python

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os


train_path = 'train'
test_path = 'validation'

stratocaster_train_path = f'{train_path}/stratocaster'
flyingv_train_path = f'{train_path}/flyingv'
lespaul_train_path = f'{train_path}/lespaul'
telecaster_train_path = f'{train_path}/telecaster'

stratocaster_test_path = f'{test_path}/stratocaster'
flyingv_test_path = f'{test_path}/flyingv'
lespaul_test_path = f'{test_path}/lespaul'
telecaster_test_path = f'{test_path}/telecaster'

# print("2 stratocasters training content:")
# print(os.listdir(stratocaster_train_path)[:2])

# print("2 stratocasters test content:")
# print(os.listdir(stratocaster_test_path)[:2])

# print('total training stratocaster images:', len(os.listdir(stratocaster_train_path ) ))
# print('total training flyingv images:', len(os.listdir(flyingv_train_path ) ))
# print('total training lespaul images:', len(os.listdir(lespaul_train_path ) ))
# print('total training telecaster images:', len(os.listdir(telecaster_train_path ) ))


def getModel(inputShape):
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(inputShape,inputShape,3)), # 98
        tf.keras.layers.MaxPooling2D(2,2), # 49

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # 47
        tf.keras.layers.MaxPooling2D(2,2), # 23 

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'), # 21
        tf.keras.layers.MaxPooling2D(2,2), # 10 

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax)
    ])

    return model

inputShape = 150
train_datagen = ImageDataGenerator(rescale=1/255,
    horizontal_flip=True,
    height_shift_range=0.1,
    width_shift_range=0.1,
    brightness_range=(0.5,1.5),
    zoom_range = [1, 1.5])

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(inputShape, inputShape),  # All images will be resized to (inputShape x inputShape)
        batch_size=128,
        # Since we use sparse_categorical_crossentropy loss, we need binary labels
        class_mode='categorical')

model = getModel(inputShape)

# Print the model summary
# model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy') # or sparse_categorical_crossentropy ?

history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1)