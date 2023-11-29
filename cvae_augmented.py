import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, initializers, metrics, utils, mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from visualization import plot_performance, confusion_matrix, plot_mislabeled, plot_latent_images
import tensorflow_probability as tfp
from keras_cv.layers import BaseImageAugmentationLayer
from cvae import CVAE, generate_save_callback, generate_and_save_images, decode, encode, noisy_decode
from nn_blocks import add_residual_block, transpose_res_block


# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=7240)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


# CONTROL PARAMETERS
TRAIN = True
RESUME = True
SAVE_VAE = False
GENERATE = False
USE_PRETRAIN = False
VAE_AUGMENT = False
CLASSIFY = False
VAE_LOCATION = '/home/sambyron/engineering/ML/tensorflow/cifar10/cvae_augmented/cvae_augmented'


# HYPER-PARAMETERS
EPOCHS = 201
LATENT_DIM = 512
# LEARNING_RATE = 0.0005
LEARNING_RATE = 0.0001
DECAY_RATE = 0.9
BATCH_SIZE = 128
DROPOUT = 0.05
# DROPOUT = 0.2
DIFFICULITY = 1.3
MODIFIER = 0
train_size = 50000
test_size = 10000

        
# TRAINING USING VAE

# LOAD DATASET
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_dataset = (tf.data.Dataset.from_tensor_slices(x_train)
                 .shuffle(train_size).batch(BATCH_SIZE))
test_dataset = (tf.data.Dataset.from_tensor_slices(x_test)
                .shuffle(test_size).batch(BATCH_SIZE))

# x_train = x_train[0:1000]

# x_train = x_train[0:10000]

DATA_SHAPE = x_train.shape[1:]
train_size = x_train.shape[0]
test_size = x_test.shape[0]

# ENCODER

# data_augmentation = keras.Sequential(
#         [
#             # layers.RandomFlip("horizontal"),
#             layers.RandomBrightness(0.2),
#             # layers.RandomRotation(0.05),
#             # layers.RandomZoom((-0.1, 0.1)),
#             layers.RandomContrast(0.2),
#             # layers.RandomCrop(28,28),
#             # layers.RandomTranslation(0.1,0.1)
#         ]
#     )

data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomBrightness(0.2),
            layers.RandomRotation(0.05),
            layers.RandomZoom((-0.3, 0.3)),
            layers.RandomContrast(0.2),
            # layers.RandomCrop(28,28),
            layers.RandomTranslation(0.2,0.2),
        ]
    )

x_train = layers.Rescaling(1./255)(x_train)
# augmented_x_train = data_augmentation(x_train)
x_test = layers.Rescaling(1./255)(x_test)

initializer = initializers.GlorotNormal()

convbase_inputs1 = keras.Input(shape=(32, 32, 3))
x = convbase_inputs1
x = add_residual_block(x, 32, 3, strides=2, dropout=DROPOUT, difficulity=DIFFICULITY)
x = add_residual_block(x, 128, 4, strides=2, dropout=DROPOUT, difficulity=DIFFICULITY)
out = add_residual_block(x, 256, 4, strides=2, dropout=DROPOUT, difficulity=DIFFICULITY)
convbase1 = keras.Model(convbase_inputs1, out, name="convbase1")
convbase1.summary()

convbase_inputs2 = keras.Input(shape=(4, 4, math.floor(256*DIFFICULITY)))
x = convbase_inputs2
# x = add_residual_block(x, 320, 2, strides=1, dropout=DROPOUT)
out = add_residual_block(x, 396, 2, strides=1, dropout=DROPOUT, difficulity=DIFFICULITY)
convbase2 = keras.Model(convbase_inputs2, out, name="convbase2")
convbase2.summary()


encoder_inputs = keras.Input(shape=(4, 4, math.floor(396*DIFFICULITY)))
x = encoder_inputs
x = layers.Flatten()(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean", kernel_initializer=initializer)(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var", kernel_initializer=initializer)(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
encoder.summary()
    
# DECODER
    
latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(4 * 4 * math.floor(396*DIFFICULITY), activation="relu", kernel_initializer=initializer)(latent_inputs)
x = layers.Reshape((4, 4, math.floor(396*DIFFICULITY)))(x)

x = transpose_res_block(x, 396, 1, strides=1, dropout=DROPOUT, difficulity=DIFFICULITY)

# x = transpose_res_block(x, 320, 2, strides=1, dropout=DROPOUT)

x = transpose_res_block(x, 256, 2, strides=2, dropout=DROPOUT, difficulity=DIFFICULITY)

x = transpose_res_block(x, 128, 2, strides=2, dropout=DROPOUT, difficulity=DIFFICULITY)

x = transpose_res_block(x, 64, 3, strides=2, dropout=DROPOUT, difficulity=DIFFICULITY)

decoder_outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same", kernel_initializer=initializer)(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.summary()

# RUN VAE

import numpy as np
import time
import matplotlib.pyplot as plt


cvae = CVAE(convbase1, convbase2, encoder, decoder)
cvae.compile()

# RESUME TRAINING MODEL
if RESUME:
    print("LOADING WEIGHTS AND RESUMING TRAINING")
    cvae.load_weights(VAE_LOCATION)
    # plot_latent_images(cvae, x_train[0:100])


lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=10000,
        decay_rate=DECAY_RATE)
cvae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), run_eagerly=True)

if TRAIN:
    # x_test = layers.Rescaling(1./255)(x_test)
    # cvae.fit(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=[generate_save_callback(x_test)])

    # cvae.fit(augmented_x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=[generate_save_callback(augmented_x_train)])

    for epoch in range(0, EPOCHS):
        start_time = time.time()
        for batch_x_train in train_dataset:
           batch_x_train = data_augmentation(batch_x_train)
           batch_x_train = layers.Rescaling(1./255)(batch_x_train)

           cvae.train_step(batch_x_train)
        

        loss = tf.keras.metrics.Mean()
        for batch_x_test in test_dataset:
            batch_x_test = layers.Rescaling(1./255)(batch_x_test)
            cvae.test_step(batch_x_test)
            loss = cvae.test_total_loss_tracker.result()
        end_time = time.time()
        print('Epoch: {}, Test set loss: {}, time elapse for current epoch: {}'
                .format(epoch, loss, end_time - start_time))
        if epoch % 5 == 0:
            generate_and_save_images(cvae, epoch, batch_x_test, VAE_LOCATION, False)
            if SAVE_VAE:
                print("SAVING WEIGHTS")
                cvae.save_weights(VAE_LOCATION, overwrite=True)


# LOAD MODEL WEIGHTS
if USE_PRETRAIN or VAE_AUGMENT or GENERATE:
    print("LOADING WEIGHTS")
    cvae.load_weights(VAE_LOCATION)
    # plot_latent_images(cvae, x_train[0:100])


if GENERATE:
    plot_latent_images(cvae, x_train[0:100], LATENT_DIM)


#================================================================================
#+=========================================



def augment_ds(images, labels, cvae):
    all_features = []
    all_labels = []
    scaled_imgs = images/255
    generated_imgs = decode(cvae, scaled_imgs)
    # generated_imgs = 0.8*scaled_imgs + 0.2*generated_imgs
    plot_latent_images(cvae, generated_imgs[0:100], LATENT_DIM)
    all_features.append(generated_imgs)
    # all_features.append(images)
    # all_labels.append(labels)
    all_labels.append(labels)

    
    return np.concatenate(all_features), np.concatenate(all_labels)

(o_train_images, train_labels), (o_test_images, test_labels) = datasets.cifar10.load_data()

# Shuffle data to hopefully get balanced class representation
idx_permutes = np.random.permutation(len(o_train_images))
train_images = o_train_images[idx_permutes]
train_labels = train_labels[idx_permutes]

if VAE_AUGMENT:
    print("AUGMENTING WITH CVAE GENERATION")
    generated_train, genereated_labels = augment_ds(train_images, train_labels, cvae)


if CLASSIFY:
    # CNN training and testing params
    INIT_LEARNING_RATE = 0.001
    DECAY_RATE = 0.9
    # REGULARIZATION
    EPOCHS = 1000
    BATCH_SIZE = 128
    VERBOSE = 1
    NB_CLASSES = 10
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = 'adam'
    ACTIVATION = 'relu'

    

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

    initializer = initializers.GlorotNormal()

    # PREPROCESSING, DATA AUGMENTATION
    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs
    x = data_augmentation(x)
    x = layers.Rescaling(1./255)(x)
    # x = layers.Normalization()(x)
    cvae.convbase1.trainable = False
    x = cvae.convbase1(x)
    cvae.convbase2.trainable = False
    x = cvae.convbase2(x)
    cvae.encoder.trainable = False
    z_mean, z_log_var = cvae.encoder(x)
    x = z_mean

    # x = layers.add([z_mean, z_log_var])
    # x = layers.Concatenate(axis=1)([z_mean, z_log_var])
    
    x = layers.Flatten()(x)
    residual = x
    
    DROPOUT = 0.1
    x = layers.Dense(math.floor(2048*DIFFICULITY), kernel_initializer = initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x) 
    x = layers.Activation(ACTIVATION)(x)
    x = layers.Dropout(DROPOUT)(x)
    # if x.shape[-1] != residual.shape[-1]:
    #     residual = layers.Conv2D(x.shape[-1], 1)(residual)
    # x = layers.add([x, residual])
    residual = x

    x = layers.Dense(math.floor(2048*DIFFICULITY), kernel_initializer = initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x) 
    x = layers.Activation(ACTIVATION)(x)
    x = layers.Dropout(DROPOUT)(x)
    # x = layers.add([x, residual])
    residual = x

    outputs = layers.Dense(10, kernel_initializer = initializer, activation="sigmoid")(x)
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

    if VAE_AUGMENT:
        

        history = model.fit(generated_train, genereated_labels, epochs=20, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[tensorboard_callback])

        history = model.fit(train_images, train_labels, epochs=20, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[tensorboard_callback])
    else:
        history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[tensorboard_callback])


    plot_performance(history)

    test_loss, test_acc = model.evaluate(o_test_images,  test_labels, verbose=2)
    print(test_acc)

    confusion_matrix(model, o_test_images, test_labels, len(class_names))
    
    plot_mislabeled(model, o_test_images, test_labels, class_names)