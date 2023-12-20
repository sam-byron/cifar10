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
            layers.RandomRotation(0), #SLOW learning rate
            layers.RandomZoom((-0.3, 0.3)), #GOOD learning rate
            layers.RandomContrast(0.41), #GOOD learning rate
            layers.RandomCrop(32,32), #SLOW-AVG learning rate (poor acc performance?), smaller number of params
            layers.RandomTranslation(0.35,0.35), #AVG learning rate
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

history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=[tensorboard_callback], validation_data=(o_test_images, test_labels))

plot_performance(history)

test_loss, test_acc = model.evaluate(o_test_images,  test_labels, verbose=2)
print(test_acc)

confusion_matrix(model, o_test_images, test_labels, len(class_names))

plot_mislabeled(model, o_test_images, test_labels, class_names)




# 313/313 - 1s - loss: 0.2160 - sparse_categorical_accuracy: 0.9430 - 838ms/epoch - 3ms/step
# 0.9430000185966492
# 313/313 [==============================] - 1s 2ms/step
# CONFUSION MATRIX
# tf.Tensor(
# [[954   4  11   4   0   2   5   0  18   2]
#  [  1 982   1   1   0   1   0   0   4  10]
#  [ 13   1 919  11  13  13  23   5   2   0]
#  [  5   3  13 874   6  61  24   6   6   2]
#  [  2   0  10  18 946   4   9  11   0   0]
#  [  0   1  16  70   6 894   7   6   0   0]
#  [  3   0   5  12   0   0 978   0   1   1]
#  [  4   0   3   7   5  15   1 964   1   0]
#  [ 12   9   2   1   0   0   2   0 967   7]
#  [  2  30   2   2   0   1   3   4   4 952]], shape=(10, 10), dtype=int32)
# MICRO(CLASS AVGs) F1 SCORE 
# <bound method _EagerTensorBase.numpy of <tf.Tensor: shape=(), dtype=float32, numpy=0.94300574>>
# CLASS F1 SCORES
# <bound method _EagerTensorBase.numpy of <tf.Tensor: shape=(10,), dtype=float32, numpy=
# array([0.9559118 , 0.96748763, 0.92734605, 0.874     , 0.95748985,
#        0.89759034, 0.95372623, 0.9659319 , 0.9655517 , 0.96453905],
#       dtype=float32)>>
# 313/313 [==============================] - 1s 2ms/step
