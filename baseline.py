
import tensorflow as tf
import datetime

from tensorflow.keras import datasets, layers, models, optimizers, initializers
import matplotlib.pyplot as plt
from tensorflow import math
import numpy as np

# CNN training and testing params
INIT_LEARNING_RATE = 0.001
DECAY_RATE = 0.9
# REGULARIZATION
EPOCHS = 15
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
VALIDATION_SPLIT = 0.2
OPTIMIZER = 'adam'
ACTIVATION = 'relu'
DROPOUT = 0.3
DIFFICULITY = 2

(o_train_images, train_labels), (o_test_images, test_labels) = datasets.cifar10.load_data()

# Shuffle data to hopefully get balanced class representation
idx_permutes = np.random.permutation(len(o_train_images))
train_images = o_train_images[idx_permutes]
train_labels = train_labels[idx_permutes]

# Standardize pixel values
train_images = (train_images - train_images.mean())/(train_images.std())
test_images = (o_test_images - o_test_images.mean())/(o_test_images.std())

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

initializer = tf.keras.initializers.GlorotNormal()

model = models.Sequential()
model.add(layers.Conv2D(32*DIFFICULITY, (3, 3), activation=ACTIVATION, input_shape=(32, 32, 3), kernel_initializer = initializer))
model.add(layers.Dropout(DROPOUT))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64*DIFFICULITY, (3, 3), activation=ACTIVATION, kernel_initializer = initializer))
model.add(layers.Dropout(DROPOUT))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64*DIFFICULITY, (3, 3), activation=ACTIVATION, kernel_initializer = initializer))
model.add(layers.Dropout(DROPOUT))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64*DIFFICULITY, activation=ACTIVATION, kernel_initializer = initializer))
model.add(layers.Dropout(DROPOUT))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Dense(10, kernel_initializer = initializer))
model.summary()


lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LEARNING_RATE,
    decay_steps=1000,
    decay_rate=DECAY_RATE)

adam_optimizer = optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=adam_optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

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

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

# https://www.tensorflow.org/api_docs/python/tf/keras/Model#call
test_predictions = model(test_images)
# https://www.enthought.com/blog/deep-learning-extracting/
# see multi-class classification section
test_predictions = tf.nn.softmax(test_predictions)
test_predictions = test_predictions.numpy().argmax(axis=1)
print(test_predictions[0:10])
test_labels = test_labels.ravel()
print(test_labels)
# https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
print(math.confusion_matrix(test_labels, test_predictions, len(class_names)))


mislabeled = tf.not_equal(test_predictions, test_labels)
# print(len(mislabeled))

plt.clf()
plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if mislabeled[i]:
        plt.imshow(o_test_images[i])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[test_predictions[i]])
plt.show()