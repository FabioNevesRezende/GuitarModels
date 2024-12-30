#!/usr/bin/python

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt

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

        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(inputShape,inputShape,3)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        # tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax)
    ])

    return model

inputShape = 150
train_datagen = ImageDataGenerator(rescale=1/255,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest')

test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(inputShape, inputShape),  # All images will be resized to (inputShape x inputShape)
        batch_size=5,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical')

test_generator =  test_datagen.flow_from_directory(test_path,
                                                         batch_size=5,
                                                         class_mode  = 'categorical',
                                                         target_size = (inputShape, inputShape))

model = getModel(inputShape)

# Print the model summary
model.summary()

model.compile(optimizer = 'adam',  # optimizer=RMSprop(learning_rate=0.001)
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(
      train_generator,
      steps_per_epoch=4,  
      epochs=150,
      validation_data=test_generator,
      verbose=1)



#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  (epochs, acc)
plt.plot  (epochs, val_acc)
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )

plt.show()