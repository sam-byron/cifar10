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
import keras_tuner as kt
    
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Hyperparamers tuner params
BATCHSIZE = 128
VERBOSE = 1
MAXTRIALS = 100
EXECSPERTRIAL = 1
EPOCHS = 60
SEARCH = True

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def build_model(hp):

    initializer = initializers.GlorotNormal()
    brightness = hp.Float(name="brigthness", min_value=0, max_value=0.8, step=0.1)
    contrast = hp.Float(name="contrast", min_value=0.01, max_value=0.8, step=0.1)
    rotation = hp.Float(name="rotation", min_value=0, max_value=0.5, step=0.05)
    crop = hp.Int(name="crop", min_value=20, max_value=32, step=2)
    zoom = hp.Float(name="zoom", min_value=0, max_value=0.4, step=0.05)
    translation = hp.Float(name="translation", min_value=0, max_value=0.4, step=0.05)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"), #GOOD learning rate
            layers.RandomBrightness(brightness), #GOOD learning rate
            layers.RandomRotation(rotation), #SLOW learning rate
            layers.RandomZoom((-1*zoom, zoom)), #GOOD learning rate
            layers.RandomContrast(contrast), #GOOD learning rate
            layers.RandomCrop(crop,crop), #SLOW-AVG learning rate (poor acc performance?), smaller number of params
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
    # model.summary()

    model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["sparse_categorical_accuracy"])
    
    return model


tuner = kt.BayesianOptimization(
    build_model,
    objective="val_sparse_categorical_accuracy",
    max_trials=MAXTRIALS,
    executions_per_trial=EXECSPERTRIAL,
    directory="autotune_data_augmentation",
    overwrite=False,
    distribution_strategy=tf.distribute.MirroredStrategy(),
)

tuner.search_space_summary()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train_full = x_train[:]
y_train_full = y_train[:]
num_val_samples = 10000
x_train, x_val = x_train[:-num_val_samples], x_train[-num_val_samples:]
y_train, y_val = y_train[:-num_val_samples], y_train[-num_val_samples:]
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
]
if SEARCH:
    tuner.search(
        x_train, y_train,
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=VERBOSE,
    )

# top_n = 4
# best_hps = tuner.get_best_hyperparameters(top_n)
# print(best_hps)

# def get_best_epoch(hp):
#     model = build_model(hp)
#     callbacks=[
#         keras.callbacks.EarlyStopping(
#             monitor="val_loss", mode="min", patience=10)
#     ]
#     history = model.fit(
#         x_train, y_train,
#         validation_data=(x_val, y_val),
#         epochs=100,
#         batch_size=128,
#         callbacks=callbacks)
#     val_loss_per_epoch = history.history["val_loss"]
#     best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
#     print(f"Best epoch: {best_epoch}")
#     return best_epoch

# def get_best_trained_model(hp):
#     best_epoch = get_best_epoch(hp)
#     model = build_model(hp)
#     model.fit(
#         x_train_full, y_train_full,
#         batch_size=128, epochs=int(best_epoch * 1.2))
#     return model

# best_models = []
# for hp in best_hps:
#     model = get_best_trained_model(hp)
#     model.evaluate(x_test, y_test)
#     best_models.append(model)

# best_models = tuner.get_best_models(top_n)