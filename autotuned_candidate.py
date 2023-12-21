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
from nn_blocks import residual_block, dense_block
    
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# CNN training and testing params
# REGULARIZATION
EPOCHS = 300
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
VALIDATION_SPLIT = 0.2
OPTIMIZER = 'adam'
ACTIVATION = 'relu'

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
            layers.RandomRotation(0.05), #SLOW learning rate
            layers.RandomZoom((-0.15, 0.15)), #GOOD learning rate
            layers.RandomContrast(0.2), #GOOD learning rate
            # layers.RandomCrop(28,28), #SLOW-AVG learning rate (poor acc performance?), smaller number of params
            layers.RandomTranslation(0.1,0.1), #AVG learning rate
        ]
)

# PREPROCESSING, DATA AUGMENTATION
inputs = keras.Input(shape=(32, 32, 3))
x = inputs
x = data_augmentation(x)
x = layers.Rescaling(1./255)(x)
# x = layers.Normalization()(x)

# CNN BLOCK 1
x = residual_block(x, 64, 3, strides=2, dropout=0)

# CNN BLOCK 2
x = residual_block(x, 128, 4, strides=2, dropout=0.1)

# CNN BLOCK 3
x = residual_block(x, 224, 4, strides=2, dropout=0.1)


x = layers.Flatten()(x)

x = dense_block(x, 320, 2, dropout=0.1)

outputs = layers.Dense(10, kernel_initializer = initializer, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

adam_optimizer = optimizers.Adam()

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