from tensorflow import keras
from tensorflow.keras import layers, callbacks, initializers
import tensorflow as tf

# LOAD DATASET
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

# x_train = x_train[1:10000]

# CONTROL PARAMETERS
PRETRAIN = False
SAVE_VAE = False
USE_PRETRAIN = True
CLASSIFY = True
VAE_LOCATION = '/home/sambyron/engineering/ML/tensorflow/cifar10/model_pretraining2_tf2.8/vae_weights'


# HYPER-PARAMETERS
EPOCHS = 50
# set the dimensionality of the latent space to a plane for visualization later
LATENT_DIM = 256
LEARNING_RATE = 0.001
BATCH_SIZE = 128
DROPOUT = 0.2
DATA_SHAPE = x_train.shape[1:]
train_size = x_train.shape[0]
test_size = x_test.shape[0]

# ENCODER

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        # layers.RandomBrightness(0.2),
        # layers.RandomRotation(0.05),
        layers.RandomZoom((-0.3, 0.3)),
        # layers.RandomContrast(0.2)
    ]
)

initializer = initializers.GlorotNormal()

encoder_inputs = keras.Input(shape=DATA_SHAPE)
# x = data_augmentation(encoder_inputs)
x = encoder_inputs
x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
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
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# DECODER
    
latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(32 * 32 * 256, activation="relu", kernel_initializer=initializer)(latent_inputs)
x = layers.Reshape((32, 32, 256))(x)
x = layers.Conv2DTranspose(256, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
decoder_outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same", kernel_initializer=initializer)(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.summary()

# VAE CLASS

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
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
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            # kl_loss = -0.5 * (LATENT_DIM + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # kl_loss = 0.5*(tf.square(z_mean) + tf.exp(z_log_var) - z_log_var - LATENT_DIM)
            kl_loss = 0.5*(tf.square(z_mean) + tf.exp(z_log_var) - z_log_var - 1)
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
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

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
if PRETRAIN:
    vae.fit(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    if SAVE_VAE:
        vae.save_weights(VAE_LOCATION)
        # vae.save(VAE_LOCATION, save_format="tf")

    z_mean, z_log_var = vae.encoder.predict(x_test)
    sampler = Sampler()
    z = sampler(z_mean, z_log_var)
    decoded_imgs = decoder.predict(z)

    import matplotlib.pyplot as plt

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
    # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
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
    plt.show()


# CLASSIFICATION
from tensorflow.keras import optimizers
import os

if CLASSIFY:
    initializer = initializers.GlorotNormal()

    cnn_inputs = keras.Input(shape=DATA_SHAPE)
    x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(cnn_inputs)
    x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
    # x = layers.SpatialDropout2D(DROPOUT)(x)
    x = layers.Conv2D(256, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean", kernel_initializer=initializer)(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var", kernel_initializer=initializer)(x)
    sampler = Sampler()
    x = sampler(z_mean, z_log_var)
    x = layers.Dense(128, kernel_initializer = initializer)(x)
    outputs = layers.Dense(10, kernel_initializer = initializer, activation="softmax")(x)
    cnn = keras.Model(inputs=cnn_inputs, outputs=outputs)
    if USE_PRETRAIN:
        cnn.layers[1].trainable = True
        cnn.layers[2].trainable = True
        cnn.layers[3].trainable = True
        cnn.layers[4].trainable = True
        cnn.layers[5].trainable = False
        cnn.layers[6].trainable = False
    cnn.summary()

    adam_optimizer = optimizers.Adam()

    cnn.compile(optimizer=adam_optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # Initialize weights (pretraining)
    if os.path.isfile(VAE_LOCATION):
        # vae = keras.Model()
        # vae.compile()
        # vae.call()
        # vae.build(input_shape = DATA_SHAPE)
        vae.fit(x_train[0:10], epochs=1, batch_size=BATCH_SIZE, verbose=3)
        vae.load_weights(VAE_LOCATION)
        # vae = keras.models.load_model(VAE_LOCATION)

    if USE_PRETRAIN:
        cnn.layers[1].set_weights(vae.encoder.layers[1].get_weights())
        cnn.layers[2].set_weights(vae.encoder.layers[2].get_weights())
        cnn.layers[3].set_weights(vae.encoder.layers[3].get_weights())
        cnn.layers[4].set_weights(vae.encoder.layers[4].get_weights())
        cnn.layers[5].set_weights(vae.encoder.layers[5].get_weights())
        cnn.layers[6].set_weights(vae.encoder.layers[6].get_weights())

    cnn.fit(x_train, y_train, epochs=50, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)