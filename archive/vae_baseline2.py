import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, initializers, metrics, utils
import matplotlib.pyplot as plt
from tensorflow import math
import numpy as np
import os


# PRETRAINING USING VAE

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
VAE_LOCATION = '/home/sambyron/engineering/ML/tensorflow/cifar10/model/'


# HYPER-PARAMETERS
EPOCHS = 25
# set the dimensionality of the latent space to a plane for visualization later
LATENT_DIM = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 128
DROPOUT = 0.1
DATA_SHAPE = x_train.shape[1:]
train_size = x_train.shape[0]
test_size = x_test.shape[0]

# ENCODER

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.2),
        layers.RandomRotation(0.05),
        layers.RandomZoom((-0.3, 0.3)),
        layers.RandomContrast(0.2)
    ]
)

initializer = initializers.GlorotNormal()

encoder_inputs = keras.Input(shape=DATA_SHAPE)
x = encoder_inputs
# x = data_augmentation(x)
# x = layers.Rescaling(1./255)(x)
# x = layers.Normalization(axis=-1)(x)
# x = encoder_inputs
x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
x = layers.Conv2D(512, 3, activation="relu", strides=2, padding="same", kernel_initializer=initializer)(x)
x = layers.Conv2D(512, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
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
x = layers.Dense(8 * 8 * 512, activation="relu", kernel_initializer=initializer)(latent_inputs)
x = layers.Reshape((8, 8, 512))(x)
x = layers.Conv2DTranspose(512, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(512, 3, activation="relu", strides=2, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(256, 3, activation="relu", strides=1, padding="same", kernel_initializer=initializer)(x)
# x = layers.SpatialDropout2D(DROPOUT)(x)
# x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same", kernel_initializer=initializer)(x)
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
            reconstruction = self.decoder(z)
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

#================================================================================
#================================================================================
#================================================================================
#================================================================================

if CLASSIFY:
    # CNN training and testing params
    INIT_LEARNING_RATE = 0.001
    DECAY_RATE = 0.9
    # REGULARIZATION
    EPOCHS = 100
    BATCH_SIZE = 128
    VERBOSE = 1
    NB_CLASSES = 10
    VALIDATION_SPLIT = 0.2
    OPTIMIZER = 'adam'
    ACTIVATION = 'relu'
    DROPOUT = 0.1
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
            layers.RandomBrightness(0.2),
            layers.RandomRotation(0.05),
            layers.RandomZoom((-0.3, 0.3)),
            layers.RandomContrast(0.2)
        ]
    )

    # PREPROCESSING, DATA AUGMENTATION
    inputs = keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    # x = layers.Normalization()(x)

    cvae = VAE(encoder, decoder)
    cvae.trainable = False
    cvae.compile()
    cvae.fit(x_train[0:10], epochs=1, batch_size=BATCH_SIZE, verbose=3)
    if os.path.isfile(VAE_LOCATION):
        if USE_PRETRAIN:
            cvae.load_weights(VAE_LOCATION)
    z_mean, z_log_var = cvae.encoder(x)
    sampler = Sampler()
    sample = sampler(z_mean, z_log_var)
    # sample = layers.Flatten()(sample)
    # code = layers.Concatenate([z_mean, z_log_var])

    x = layers.Flatten()(sample)
    
    # x = layers.add([x, z_mean, z_log_var])
    
    # x = layers.Concatenate()([x, sample])
    # x = layers.concatenate([z_mean, z_log_var])
    x = layers.Dense(512, kernel_initializer = initializer)(x)
    x = layers.Dense(512, kernel_initializer = initializer)(x)
    x = layers.Dense(512, kernel_initializer = initializer)(x)
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
    print(math.confusion_matrix(test_labels, max_test_predictions, len(class_names)))
    ohe_test_labels = utils.to_categorical(test_labels)
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