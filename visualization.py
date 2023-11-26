import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, metrics, utils
from cvae import Sampler
import tensorflow_probability as tfp



def plot_performance(history):
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


def confusion_matrix(model, x, y, num_classes):
     # https://www.tensorflow.org/api_docs/python/tf/keras/Model#call
    test_predictions = model.predict(x)
    # https://www.enthought.com/blog/deep-learning-extracting/
    # see multi-class classification section
    y = y.ravel()
    max_test_predictions = test_predictions.argmax(axis=1)
    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
    print("CONFUSION MATRIX")
    print(tf.math.confusion_matrix(y, max_test_predictions, num_classes))
    ohe_test_labels = utils.to_categorical(y)
    print("MICRO(CLASS AVGs) F1 SCORE ")
    f1_micro = metrics.F1Score(average="micro")
    f1_micro.update_state(ohe_test_labels, test_predictions)
    print(f1_micro.result().numpy)
    print("CLASS F1 SCORES")
    f1 = metrics.F1Score()
    f1.update_state(ohe_test_labels, test_predictions)
    print(f1.result().numpy)

def plot_mislabeled(model, x, y, class_names):
    
    y = y.ravel()
    test_predictions = model.predict(x)
    # https://www.enthought.com/blog/deep-learning-extracting/
    # see multi-class classification section
    max_test_predictions = test_predictions.argmax(axis=1)
    mislabeled = tf.not_equal(max_test_predictions, y)

    plt.clf()
    # plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if mislabeled[i]:
            plt.imshow(x[i])
            # The CIFAR labels happen to be arrays, 
            # which is why you need the extra index
            plt.xlabel(class_names[max_test_predictions[i]])
    plt.show()


def plot_latent_images(cvae, x, latent_dim, n=10, digit_size=32):
    """Plots n x n digit images decoded from the latent space."""
    plt.clf()
    
    norm = tfp.distributions.Normal(0, 1)

    o_x = x
    image_width = digit_size*n
    image_height = image_width

    z_mean = norm.sample((100, latent_dim))
    z_log_var = norm.sample((100, latent_dim))
    z = cvae.sampler(z_mean, z_log_var)
    z_decoded = cvae.decoder.predict(z)

    # x_decoded = cvae.decode(x)
    x = cvae.convbase1.predict(x)
    x = cvae.convbase2.predict(x)
    z_mean, z_log_var = cvae.encoder.predict(x)
    sampler = Sampler()
    s = sampler(z_mean, z_log_var)
    x_decoded = cvae.decoder.predict(s)

    noisy_x = s*z
    noisy_x_decoded = cvae.decoder.predict(noisy_x)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
    # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(o_x[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_decoded[i])
        # plt.imshow(z_decoded[i])

        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()