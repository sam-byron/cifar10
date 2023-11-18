import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, metrics, utils


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