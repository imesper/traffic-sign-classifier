import tensorflow as tf
import pickle
import cv2
import numpy as np

testing_file = 'data/test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']
# Replace each question mark with the appropriate value.
# Use python, pandas or numpy methods rather than hard coding the results

image_witdh = len(X_test[0])
image_height = len(X_test[0][0])
image_channels = len(X_test[0][0][0])

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8, 8))
image_shape = (image_height, image_witdh, image_channels)
X_test_grey = np.zeros((n_test, image_shape[0],  image_shape[1]))
X_test_grey_clahe = np.zeros((n_test, image_shape[0],  image_shape[1]))
X_test_grey_norm = np.zeros((n_test, image_shape[0],  image_shape[1]))

for i in range(n_test):
    X_test_grey[i] = clahe.apply(cv2.cvtColor(
        X_test[i][:32], cv2.COLOR_RGB2GRAY), 0)
    X_test_grey_clahe[i] = clahe.apply(
        cv2.cvtColor(X_test[i], cv2.COLOR_RGB2GRAY))
    X_test_grey_norm[i] = (X_test_grey_clahe[i] / 255).astype(np.float32)

X_test_grey_norm = X_test_grey_norm.reshape(
    n_test, image_shape[0], image_shape[0], 1)
X_test = X_test_grey_norm

image_shape = (image_shape[0], image_shape[1], 1)
EPOCHS = 20
BATCH_SIZE = 128
n_classes = 43


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()

    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset +
                                  BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(op_to_restore, feed_dict={
                            x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:

    saver = tf.train.import_meta_graph('trafficSignModel01.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    [print(n.name) for n in tf.get_default_graph().get_operations()]
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("Placeholder:0")
    y = graph.get_tensor_by_name("Placeholder_1:0")
    one_hot_y = graph.get_tensor_by_name("one_hot:0")
    op_to_restore = graph.get_tensor_by_name("Mean_1:0")

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
