
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, initializers
from keras.layers import concatenate
import matplotlib.pyplot as plt
from tensorflow import math
import numpy as np

# CNN training and testing params
INIT_LEARNING_RATE = 0.001
DECAY_RATE = 0.9
# REGULARIZATION
EPOCHS = 30
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
VALIDATION_SPLIT = 0.2
OPTIMIZER = 'adam'
ACTIVATION = 'relu'
DROPOUT = 0.18
DIFFICULITY = 1

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
        layers.RandomFlip("horizontal"),
        # layers.RandomRotation(0.05),
        layers.RandomZoom((-0.3, 0.3)),
        # layers.RandomContrast(0.2)
        # layers.RandomTranslation((-0.2, 0.2), (-0.2, 0.2))
    ]
)

inputs = keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
# x = layers.Normalization()(x)

x = layers.Conv2D(64*DIFFICULITY, (3, 3), activation=ACTIVATION, kernel_initializer = initializer, padding="same")(x)
x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128*DIFFICULITY, (3, 3), activation=ACTIVATION, kernel_initializer = initializer, padding="same")(x)
x = layers.SpatialDropout2D(DROPOUT)(x)
skip = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(256*DIFFICULITY, (3, 3), activation=ACTIVATION, kernel_initializer = initializer, padding="same")(skip)
x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.MaxPooling2D((2, 2))(x)

merge = concatenate([x, skip], axis=3)
x = layers.Conv2D(1024*DIFFICULITY, (3, 3), activation=ACTIVATION, kernel_initializer = initializer, padding="same")(merge)
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
# x = layers.Dense(128*DIFFICULITY, activation=ACTIVATION, kernel_initializer = initializer)(x)
# x = layers.Dropout(DROPOUT)(x)
outputs = layers.Dense(10, kernel_initializer = initializer)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LEARNING_RATE,
    decay_steps=10000,
    decay_rate=DECAY_RATE)

adam_optimizer = optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=adam_optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[tensorboard_callback])


loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# test_loss, test_acc = model.evaluate(o_test_images,  test_labels, verbose=2)
# print(test_acc)

# # https://www.tensorflow.org/api_docs/python/tf/keras/Model#call
# test_predictions = model(o_test_images)
# # https://www.enthought.com/blog/deep-learning-extracting/
# # see multi-class classification section
# test_predictions = tf.nn.softmax(test_predictions)
# test_predictions = test_predictions.numpy().argmax(axis=1)
# print(test_predictions[0:10])
# test_labels = test_labels.ravel()
# print(test_labels)
# # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
# print(math.confusion_matrix(test_labels, test_predictions, len(class_names)))


# mislabeled = tf.not_equal(test_predictions, test_labels)

# plt.clf()
# # plt.figure(figsize=(10,10))
# for i in range(100):
#     plt.subplot(10,10,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     if mislabeled[i]:
#         plt.imshow(o_test_images[i])
#         # The CIFAR labels happen to be arrays, 
#         # which is why you need the extra index
#         plt.xlabel(class_names[test_predictions[i]])
# plt.show()

