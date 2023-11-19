
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, initializers, metrics, mixed_precision
import matplotlib.pyplot as plt
import math
import numpy as np
import keras_cv
# from keras_cv import utils
from keras_cv.layers import BaseImageAugmentationLayer
from visualization import plot_performance, confusion_matrix, plot_mislabeled
    
class RandSaturationHue(keras_cv.layers.BaseImageAugmentationLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def augment_image(self, image, transformation=None, **kwargs):
    
        image = tf.image.random_saturation(image, 0.75, 1.25)
        image = tf.image.random_hue(image, 0.1)
        return image
    
    def augment_label(self, label, transformation=None, **kwargs):
        # you can use transformation somehow if you want
        return label
    
    def augment_bounding_boxes(self, bounding_boxes, transformation=None, **kwargs):
        # you can also perform no-op augmentations on label types to support them in
        # your pipeline.
        return bounding_boxes


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# CNN training and testing params
INIT_LEARNING_RATE = 0.001
DECAY_RATE = 0.9
# REGULARIZATION
EPOCHS = 2
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
VALIDATION_SPLIT = 0.2
OPTIMIZER = 'adam'
ACTIVATION = 'relu'
DROPOUT = 0.2
DIFFICULITY = 1

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

(o_train_images, train_labels), (o_test_images, test_labels) = datasets.cifar10.load_data()


# Shuffle data to hopefully get balanced class representation
idx_permutes = np.random.permutation(len(o_train_images))
train_images = o_train_images[idx_permutes]
train_labels = train_labels[idx_permutes]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

initializer = initializers.GlorotNormal()

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"), #GOOD learning rate
        layers.RandomBrightness(0.2), #GOOD learning rate
        # layers.RandomRotation(0.05), #SLOW learning rate
        layers.RandomZoom((-0.3, 0.3)), #GOOD learning rate
        layers.RandomContrast(0.2), #GOOD learning rate
        layers.RandomCrop(28,28), #SLOW-AVG learning rate (poor acc performance?), smaller number of params
        layers.RandomTranslation(0.2,0.2), #AVG learning rate
        # RandSaturationHue(),
    ]
)

# PREPROCESSING, DATA AUGMENTATION
inputs = keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
# x = layers.Normalization()(x)

# CNN LAYER 1
x = layers.Conv2D(math.floor(32*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
x = layers.SpatialDropout2D(DROPOUT)(x)
residual = x

# CNN LAYER 2
x = layers.Conv2D(math.floor(32*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)

x = layers.SpatialDropout2D(DROPOUT)(x)

x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 3

x = layers.Conv2D(math.floor(32*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.MaxPooling2D((2, 2))(x)
residual = x

# CNN LAYER 4
x = layers.Conv2D(math.floor(128*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 5
x = layers.Conv2D(math.floor(128*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 6
x = layers.Conv2D(math.floor(128*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 7
x = layers.Conv2D(math.floor(128*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.MaxPooling2D((2, 2))(x)
residual = x

# CNN LAYER 8
x = layers.Conv2D(math.floor(256*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 9
x = layers.Conv2D(math.floor(256*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 10
x = layers.Conv2D(math.floor(256*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 11
x = layers.Conv2D(math.floor(256*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.Concatenate(axis=3)([x, residual])
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.MaxPooling2D((2, 2))(x)
residual = x

x = layers.Flatten()(x)
residual = layers.Flatten()(residual)

x = layers.Dense(math.floor(512*DIFFICULITY), kernel_initializer = initializer, use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
x = layers.Dropout(DROPOUT)(x)
x = layers.Concatenate(axis=1)([x, residual])

outputs = layers.Dense(10, kernel_initializer = initializer, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LEARNING_RATE,
    decay_steps=10000,
    decay_rate=DECAY_RATE)

adam_optimizer = optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=adam_optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["sparse_categorical_accuracy"])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[tensorboard_callback])

plot_performance(history)

test_loss, test_acc = model.evaluate(o_test_images,  test_labels, verbose=2)
print(test_acc)

confusion_matrix(model, o_test_images, test_labels, len(class_names))

plot_mislabeled(model, o_test_images, test_labels, class_names)