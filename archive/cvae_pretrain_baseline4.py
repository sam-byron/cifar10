import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, initializers, metrics, utils, mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import os
import math

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# CONTROL PARAMETERS
PRETRAIN = False
SAVE_VAE = False
USE_PRETRAIN = True
CLASSIFY = True
VAE_LOCATION = '/home/sambyron/engineering/ML/tensorflow/cifar10/model/'


# HYPER-PARAMETERS
EPOCHS = 51
LATENT_DIM = 128
LEARNING_RATE = 0.0001
DECAY_RATE = 0.9
BATCH_SIZE = 256
DROPOUT = 0.1
DIFFICULITY = 1
MODIFIER = 0

def residual_block(x, filters, num_layers=1, strides=1, activation="relu", dropout=0):
    residual = x
    initializer = initializers.GlorotNormal()

    filters = math.floor(filters*DIFFICULITY)
    for l in range(num_layers):
        if strides > 1 and l == num_layers-1:
            x = layers.Conv2D(filters, 3, strides=strides, kernel_initializer = initializer, padding="same")(x)
            residual = layers.Conv2D(filters, 1, strides=strides)(residual)
        else:
        # Layer l
            x = layers.Conv2D(filters, 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x) 
        x = layers.Activation(activation)(x)
        x = layers.SpatialDropout2D(dropout)(x)

    if filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)

    x = layers.add([x,  residual])
    return x

def transpose_res_block(x, filters, num_layers=1, strides=1, activation="relu", dropout=0):
    residual = x
    initializer = initializers.GlorotNormal()
    filters = math.floor(filters*DIFFICULITY)


    for l in range(num_layers):
        if l == 0:
            x = layers.Conv2DTranspose(filters, 3, strides=strides, kernel_initializer = initializer, padding="same")(x)
        else:
            x = layers.Conv2D(filters, 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x) 
        x = layers.Activation(activation)(x)
        x = layers.SpatialDropout2D(dropout)(x)

    if strides > 1:
        residual = layers.Conv2DTranspose(filters, 1, strides=strides)(residual)
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)

    x = layers.add([x,  residual])
    return x

# PRETRAINING USING VAE

# LOAD DATASET
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_train = x_train / 255
# x_test = x_test / 255

# x_train = x_train[0:1000]

# x_train = x_train[0:10000]

DATA_SHAPE = x_train.shape[1:]
train_size = x_train.shape[0]
test_size = x_test.shape[0]

# ENCODER

data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            # layers.RandomBrightness(0.2),
            layers.RandomRotation(0.05),
            layers.RandomZoom((-0.3, 0.3)),
            layers.RandomContrast(0.2),
            layers.RandomCrop(28,28),
            layers.RandomTranslation(0.2,0.2)
        ]
    )

o_x_train = layers.Rescaling(1./255)(x_train)
x_train = data_augmentation(x_train)
x_train = layers.Rescaling(1./255)(x_train)
# x_train = layers.Normalization()(x_train)
x_test = layers.Rescaling(1./255)(x_test)

initializer = initializers.GlorotNormal()

convbase_inputs1 = keras.Input(shape=DATA_SHAPE)
x = convbase_inputs1
# x = data_augmentation(x)
# x = layers.Rescaling(1./255)(x)
x = residual_block(x, 32+MODIFIER, 1, strides=1, dropout=DROPOUT)
x = residual_block(x, 64+MODIFIER, 1, strides=2, dropout=DROPOUT)
x = residual_block(x, 128+MODIFIER, 1, strides=1, dropout=DROPOUT)
out = residual_block(x, 192+MODIFIER, 1, strides=2, dropout=DROPOUT)
convbase1 = keras.Model(convbase_inputs1, out, name="convbase1")
convbase1.summary()

convbase_inputs2 = keras.Input(shape=(8, 8, 192+MODIFIER))
x = convbase_inputs2
x = residual_block(x, 256+MODIFIER, 1, strides=1, dropout=DROPOUT)
x = residual_block(x, 320+MODIFIER, 1, strides=2, dropout=DROPOUT)
out = residual_block(x, 384+MODIFIER, 1, strides=2, dropout=DROPOUT)
convbase2 = keras.Model(convbase_inputs2, out, name="convbase2")
convbase2.summary()


encoder_inputs = keras.Input(shape=(2, 2, 384+MODIFIER))
x = encoder_inputs
x = layers.Flatten()(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean", kernel_initializer=initializer)(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var", kernel_initializer=initializer)(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
encoder.summary()

class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        epsilon = tf.cast(epsilon, tf.float16)
        return z_mean + tf.exp(tf.cast(0.5, tf.float16) * z_log_var) * epsilon
        # return z_mean + tf.exp(tf.cast(0.5, tf.float16) * z_log_var)
    
# DECODER
    
latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(2 * 2 * (384+MODIFIER), activation="relu", kernel_initializer=initializer)(latent_inputs)
x = layers.Reshape((2, 2, 384+MODIFIER))(x)

x = transpose_res_block(x, 384+MODIFIER, 2, strides=1, dropout=DROPOUT)

x = transpose_res_block(x, 320+MODIFIER, 2, strides=2, dropout=DROPOUT)

x = transpose_res_block(x, 256+MODIFIER, 2, strides=2, dropout=DROPOUT)

x = transpose_res_block(x, 192+MODIFIER, 2, strides=1, dropout=DROPOUT)

x = transpose_res_block(x, 128+MODIFIER, 2, strides=2, dropout=DROPOUT)

x = transpose_res_block(x, 64+MODIFIER, 2, strides=1, dropout=DROPOUT)

x = transpose_res_block(x, 32+MODIFIER, 2, strides=2, dropout=DROPOUT)

decoder_outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same", kernel_initializer=initializer)(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.summary()

convbase_inputs1 = keras.Input(shape=DATA_SHAPE)
x = convbase1(convbase_inputs1)
x = convbase2(x)
z_mean, z_log_var = encoder(x)
sampler = Sampler()
z = sampler(z_mean, z_log_var)
reconstruction = decoder(z)
cvae = keras.Model(convbase_inputs1, reconstruction, name="cvae")
cvae.summary()

# VAE CLASS

class VAE(keras.Model):
    def __init__(self, convbase1, convbase2, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.convbase1 = convbase1
        self.convbase2 = convbase2
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x = self.convbase1(data)
            x = self.convbase2(x)
            z_mean, z_log_var = self.encoder(x)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
                # tf.reduce_sum(
                #     tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=data),
                #     axis=(1, 2)
                # )
            )

            kl_loss = 0.5*(tf.square(z_mean) + tf.exp(z_log_var) - 2*z_log_var - 1)
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
            scaled_loss = self.optimizer.get_scaled_loss(total_loss)
        
        # grads = tape.gradient(total_loss, self.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        scaled_grads = tape.gradient(scaled_loss, self.trainable_weights)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))


        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# RUN VAE

import numpy as np
import matplotlib.pyplot as plt

def generate_and_save_images(cvae, epoch, test_sample):

    test_sample = tf.cast(test_sample, tf.float32)
    x = vae.convbase1(test_sample)
    x = vae.convbase2(x)
    z_mean, z_log_var = vae.encoder.predict(x)
    sampler = Sampler()
    z = sampler(z_mean, z_log_var)
    decoded_imgs = decoder.predict(z)
    decoded_imgs = tf.cast(decoded_imgs, tf.float32)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
    # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_sample[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


class generate_save_callback(keras.callbacks.Callback):
    def __init__(self, test_samples, **kwargs):
        super().__init__(**kwargs)
        self.test_samples = test_samples
   
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            generate_and_save_images(self.model, epoch, x_test)

vae = VAE(convbase1, convbase2, encoder, decoder)
lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=10000,
        decay_rate=DECAY_RATE)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), run_eagerly=True)
if PRETRAIN:
    vae.fit(o_x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=[generate_save_callback(x_test)])

    if SAVE_VAE:
        vae.save_weights(VAE_LOCATION)

#================================================================================
#================================================================================
#================================================================================
#================================================================================

if CLASSIFY:
    # CNN training and testing params
    INIT_LEARNING_RATE = 0.001
    DECAY_RATE = 0.9
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
            layers.RandomFlip("horizontal"),
            # layers.RandomBrightness(0.2),
            layers.RandomRotation(0.05),
            layers.RandomZoom((-0.3, 0.3)),
            layers.RandomContrast(0.2),
            # layers.RandomCrop(28,28),
            layers.RandomTranslation(0.2,0.2)
        ]
    )

    # PREPROCESSING, DATA AUGMENTATION
    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    # x = layers.Normalization()(x)

    cvae = VAE(convbase1, convbase2, encoder, decoder)
    # cvae.convbase1.trainable = False
    cvae.compile()
    cvae.fit(train_images[0:10], epochs=1, batch_size=BATCH_SIZE, verbose=3)
    if USE_PRETRAIN:
        cvae.convbase1.trainable = False
        cvae.convbase2.trainable = False
        if os.path.isfile(VAE_LOCATION):
            cvae.load_weights(VAE_LOCATION)
            # cvae.convbase1.trainable = False

    x = cvae.convbase1(x)
    x = cvae.convbase2(x)
    z_mean, z_log_var = cvae.encoder(x)
    x = layers.add([z_mean, z_log_var])
    x = layers.Dense(2 * 2 * (384+MODIFIER), activation="relu", kernel_initializer=initializer)(x)
    
    # x = layers.SpatialDropout2D(DROPOUT)(x)

    x = layers.Flatten()(x)
    # x = layers.add([x, code])
    
    x = layers.Dense(2048, kernel_initializer = initializer)(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(2048, kernel_initializer = initializer)(x)
    x = layers.Dropout(DROPOUT)(x)
    # x = layers.Dense(512, kernel_initializer = initializer)(x)
    # x = layers.Dropout(DROPOUT)(x)
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

    history = model.fit(train_images, train_labels, epochs=1, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[tensorboard_callback])


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
    ohe_test_labels = utils.to_categorical(test_labels)
    # print("MICRO(CLASS AVGs) F1 SCORE ")
    # f1_micro = metrics.F1Score(average="micro")
    # f1_micro.update_state(ohe_test_labels, test_predictions)
    # print(f1_micro.result().numpy)
    # print("CLASS F1 SCORES")
    # f1 = metrics.F1Score()
    # f1.update_state(ohe_test_labels, test_predictions)
    # print(f1.result().numpy)




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


    # TRAIN ON ENCODINGS

    cvae = VAE(convbase1, convbase2, encoder, decoder)
    # cvae.convbase1.trainable = False
    cvae.compile()
    cvae.fit(train_images[0:10], epochs=1, batch_size=BATCH_SIZE, verbose=3)
    if USE_PRETRAIN:
        cvae.convbase1.trainable = False
        cvae.convbase2.trainable = False
        if os.path.isfile(VAE_LOCATION):
            cvae.load_weights(VAE_LOCATION)
            # cvae.convbase1.trainable = False

    codes = cvae.convbase1.predict(train_images)
    codes = cvae.convbase2.predict(codes)
    z_mean, z_log_var = cvae.encoder.predict(codes)
    codes = layers.add([z_mean, z_log_var])
    # codes = layers.Dense(2 * 2 * (384+MODIFIER), activation="relu", kernel_initializer=initializer)(codes)


    # inputs = keras.Input(shape=(1536,))
    inputs = keras.Input(shape=(LATENT_DIM,))
    x = inputs

    # x = layers.Flatten()(x)
    # x = layers.add([x, code])
    
    x = layers.Dense(2048, kernel_initializer = initializer)(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(2048, kernel_initializer = initializer)(x)
    x = layers.Dropout(DROPOUT)(x)
    # x = layers.Dense(512, kernel_initializer = initializer)(x)
    # x = layers.Dropout(DROPOUT)(x)
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

    history = model.fit(codes, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[tensorboard_callback])
