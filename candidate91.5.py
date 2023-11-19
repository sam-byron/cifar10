
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, initializers, metrics, mixed_precision
import matplotlib.pyplot as plt
import math
import numpy as np
import keras_cv
# from keras_cv import utils
from keras_cv.layers import BaseImageAugmentationLayer

class GrayscaleLayer(keras_cv.layers.BaseImageAugmentationLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.factor = factor

    # def get_random_transformation(self, **kwargs):
    #     # kwargs holds {"images": image, "labels": label, etc...}
    #     return self.factor

    def augment_image(self, image, transformation=None, **kwargs):
        sample = np.random.uniform(0,1,1)

        # sample = keras_cv.NormalFactorSampler(mean=0, stddev=1, min_value=0.0, max_value=1.0)
        if sample > 0.5:
            converted = tf.image.rgb_to_grayscale(image)
            return converted
        else:
            return image
    
    def augment_label(self, label, transformation=None, **kwargs):
        # you can use transformation somehow if you want
        return label
    
    def augment_bounding_boxes(self, bounding_boxes, transformation=None, **kwargs):
        # you can also perform no-op augmentations on label types to support them in
        # your pipeline.
        return bounding_boxes
    
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
EPOCHS = 200
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
# o_train_images = np.mean(o_train_images, axis=3)
# o_test_images = np.mean(o_test_images, axis=3)


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
        # GrayscaleLayer(),
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
# x = layers.add([x, residual])
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 3

x = layers.Conv2D(math.floor(32*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
# x = layers.Concatenate(axis=3)([x, residual])
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.MaxPooling2D((2, 2))(x)
# x = layers.add([x, residual])
residual = x

# CNN LAYER 4
x = layers.Conv2D(math.floor(128*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.add([x, residual])
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 5
x = layers.Conv2D(math.floor(128*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
# x = layers.Concatenate(axis=3)([x, residual])
residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.add([x, residual])
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 6
x = layers.Conv2D(math.floor(128*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
# x = layers.Concatenate(axis=3)([x, residual])
# residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.add([x, residual])
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 7
x = layers.Conv2D(math.floor(128*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
# x = layers.Concatenate(axis=3)([x, residual])
# residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.MaxPooling2D((2, 2))(x)
# x = layers.add([x, residual])
residual = x

# CNN LAYER 8
# x = layers.Concatenate(axis=3)([x, residual])
x = layers.Conv2D(math.floor(256*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.add([x, residual])
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 9
x = layers.Conv2D(math.floor(256*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
# x = layers.Concatenate(axis=3)([x, residual])
x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.add([x, residual])
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 10
x = layers.Conv2D(math.floor(256*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
# x = layers.Concatenate(axis=3)([x, residual])
residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.add([x, residual])
x = layers.Concatenate(axis=3)([x, residual])
residual = x

# CNN LAYER 11
x = layers.Conv2D(math.floor(256*DIFFICULITY), 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
x = layers.Concatenate(axis=3)([x, residual])
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
# x = layers.Concatenate(axis=3)([x, residual])
residual = x
x = layers.SpatialDropout2D(DROPOUT)(x)
x = layers.MaxPooling2D((2, 2))(x)
# x = layers.add([x, residual])
residual = x

x = layers.Flatten()(x)
residual = layers.Flatten()(residual)

# x = layers.Dense(math.floor(256*DIFFICULITY), kernel_initializer = initializer, use_bias=False)(x)
# x = layers.BatchNormalization()(x) 
# x = layers.Activation(ACTIVATION)(x)
# # x = layers.Concatenate(axis=1)([x, residual])
# # residual = x
# x = layers.Dropout(DROPOUT)(x)
# # x = layers.add([x, residual])
# residual = x


x = layers.Dense(math.floor(512*DIFFICULITY), kernel_initializer = initializer, use_bias=False)(x)
x = layers.BatchNormalization()(x) 
x = layers.Activation(ACTIVATION)(x)
x = layers.Dropout(DROPOUT)(x)
x = layers.Concatenate(axis=1)([x, residual])
# residual = x

# x = layers.add([x, residual])
# residual = x

# x = layers.Dense(math.floor(256*DIFFICULITY), kernel_initializer = initializer, use_bias=False)(x)
# x = layers.BatchNormalization()(x) 
# x = layers.Activation(ACTIVATION)(x)
# # x = layers.Concatenate(axis=1)([x, residual])
# # residual = x
# x = layers.Dropout(DROPOUT)(x)
# x = layers.add([x, residual])
# residual = x

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
plt.plot(history.history["sparse_categorical_accuracy"], label="sparse_categorical_accuracy")
plt.plot(history.history['val_sparse_categorical_accuracy'], label = 'val_sparse_categorical_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(o_test_images,  test_labels, verbose=2)
print(test_acc)

# https://www.tensorflow.org/api_docs/python/tf/keras/Model#call
test_predictions = model.predict(o_test_images)
# https://www.enthought.com/blog/deep-learning-extracting/
# see multi-class classification section
test_labels = test_labels.ravel()
max_test_predictions = test_predictions.argmax(axis=1)
# https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
print("CONFUSION MATRIX")
print(tf.math.confusion_matrix(test_labels, max_test_predictions, len(class_names)))
ohe_test_labels = tf.keras.utils.to_categorical(test_labels)
print("MICRO(CLASS AVGs) F1 SCORE ")
f1_micro = metrics.F1Score(average="micro")
f1_micro.update_state(ohe_test_labels, test_predictions)
print(f1_micro.result().numpy)
print("CLASS F1 SCORES")
f1 = metrics.F1Score()
f1.update_state(ohe_test_labels, test_predictions)
print(f1.result().numpy)




mislabeled = tf.not_equal(max_test_predictions, test_labels)

plt.clf()
# plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if mislabeled[i]:
        plt.imshow(o_test_images[i])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[max_test_predictions[i]])
plt.show()

