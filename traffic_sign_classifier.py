# Load pickled data
import pickle
import pandas as pd
import numpy as np
# TODO: Fill this in based on where you saved the training and testing data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import os

if os.path.isfile('trainAugmented.p'):
    training_file = 'trainAugmented.p'
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    X_train, y_train = train['features'], train['labels']
else:
    training_file = 'data/train.p'
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    X_train, y_train = train['features'], train['labels']

validation_file = 'data/valid.p'
testing_file = 'data/test.p'


with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

print('files loaded')
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Replace each question mark with the appropriate value.
# Use python, pandas or numpy methods rather than hard coding the results

image_witdh = len(X_train[0])
image_height = len(X_train[0][0])
image_channels = len(X_train[0][0][0])

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = (image_height, image_witdh, image_channels)

import collections
import operator


def normInverseDictProbability(data):
    value_sum = sum(data.values())
    max_value = max(data.values()) * 2
    norm_data = {}
    for i in range(len(data)):
        norm_data[i] = (max_value - data[i]) / max_value
    return norm_data


def normalizeDictData(data):
    value_sum = sum(data.values())
    print("Sum: ", value_sum, data)
    norm_data = {}
    for i in range(len(data)):
        print(data[i] / value_sum)
        norm_data[i] = data[i] / value_sum

    return norm_data


def plotHistogram(label_data):
    known = []
    classes_counts = {}
    classes = {}
    n_classes = 0
    for i, label in enumerate(label_data):
        if label not in known:
            n_classes += 1
            known.append(label)
            classes_counts[label] = 1
            classes[label] = [i]
        else:
            classes_counts[label] += 1
            classes[label].append(i)

    classes_counts = collections.OrderedDict(sorted(classes_counts.items()))

    plt.figure()
    plt.bar(range(len(classes_counts)), list(
        classes_counts.values()), align='center')
    plt.xticks(range(len(classes_counts)), list(classes_counts.keys()))
    plt.title("Data")
    plt.show()
    return classes, classes_counts, n_classes


import cv2
import os
import random


def generateAffineTransformedImage(image):
    pt1 = random.randrange(0, 10)
    pt2 = random.randrange(20, 30)
    pt3 = random.randrange(0, 10)
    pt1_2 = random.randrange(0, 10)
    pt2_2 = random.randrange(0, 10)
    pt3_2 = random.randrange(20, 30)

    var = 3
    pt1_var = random.randrange(-var, var)
    pt2_var = random.randrange(-var, var)
    pt3_var = random.randrange(-var, var)

    pt1_2_var = random.randrange(-var, var)
    pt2_2_var = random.randrange(-var, var)
    pt3_2_var = random.randrange(-var, var)

    pts1 = np.float32([[pt1, pt1_2], [pt2, pt2_2], [pt3, pt3_2]])
    pts2 = np.float32([[pt1 + pt1_var, pt1_2 + pt1_2_var], [pt2 +
                                                            pt2_var, pt2_2 + pt2_2_var], [pt3 + pt3_var, pt3_2+pt3_2_var]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M, (image_shape[0], image_shape[1]))

    return dst


def generatePerspectiveTransformedImage(image):

    pt1 = random.randrange(0, 32)
    pt2 = random.randrange(0, 32)
    pt3 = random.randrange(0, 32)
    pt4 = random.randrange(0, 32)
    pt1_2 = random.randrange(0, 32)
    pt2_2 = random.randrange(0, 32)
    pt3_2 = random.randrange(0, 32)
    pt4_2 = random.randrange(0, 32)

    var = 3
    pt1_var = random.randrange(-var, var)
    pt2_var = random.randrange(-var, var)
    pt3_var = random.randrange(-var, var)
    pt4_var = random.randrange(-var, var)
    pt1_2_var = random.randrange(-var, var)
    pt2_2_var = random.randrange(-var, var)
    pt3_2_var = random.randrange(-var, var)
    pt4_2_var = random.randrange(-var, var)
    pts1 = np.float32([[pt1, pt1_2], [pt2, pt2_2], [pt3, pt3_2], [pt4, pt4_2]])
    pts2 = np.float32([[pt1 + pt1_var, pt1_2 + pt1_2_var], [pt2 +
                                                            pt2_var, pt2_2 + pt2_2_var], [pt3 + pt3_var, pt3_2+pt3_2_var], [pt4 + pt4_var, pt4_2 + pt4_2_var]])
    print(pts1, pts2)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (image_shape[0], image_shape[1]))

    return dst


def generateRotatedImage(image):
    degree = random.randrange(-20, 20)
    M = cv2.getRotationMatrix2D(
        (image_shape[0]/2, image_shape[1]/2), degree, 1)
    dst = cv2.warpAffine(image, M, (image_shape[0], image_shape[1]), )
    return dst


# Calculating the probability of each class to augment data
train_list, train_classes, n_classes = plotHistogram(y_train)
valid_list, valid_classes, valid_n_classes = plotHistogram(y_valid)
plotHistogram(y_test)

if not os.path.isfile('trainAugmented.p'):

    inverse_train_prob = normInverseDictProbability(train_classes)
    inverse_valid_prob = normInverseDictProbability(valid_classes)

    # inverse_train_prob = inverseDictProbability(norm_train_classes)
    # inverse_valid_prob = inverseDictProbability(norm_valid_classes)

    print("Prob", inverse_train_prob.values())
    new_pop_rot = random.choices(
        list(inverse_train_prob.keys()), inverse_train_prob.values(), k=15000)
    new_pop_aff = random.choices(
        list(inverse_train_prob.keys()), inverse_train_prob.values(), k=15000)
    # print("New Pop" , new_pop)

    for i in new_pop_rot:
        index = random.choice(train_list[i])
        image = X_train[index]
        new_image = generateRotatedImage(image)
        X_train = np.append(X_train, [new_image], axis=0)
        y_train = np.append(y_train, [i])

    for i in new_pop_aff:
        index = random.choice(train_list[i])
        image = X_train[index]
        new_image = generateAffineTransformedImage(image)
        X_train = np.append(X_train, [new_image], axis=0)
        y_train = np.append(y_train, [i])

    train_data = {'features': X_train, 'labels': y_train}
    pickle.dump(train_data, open("trainAugmented.p", "wb"))

    plotHistogram(y_train)


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# Visualizations will be shown in the notebook.

index = random.randint(0, n_train)
image = X_train[index].squeeze()

# plt.figure()
# cv2.imshow('Sample', image)

from sklearn.utils import shuffle
from skimage import exposure
import numpy as np

X_train_grey = np.zeros((n_train, image_shape[0],  image_shape[1]))
X_train_grey_clahe = np.zeros((n_train, image_shape[0],  image_shape[1]))
X_train_grey_norm = np.zeros((n_train, image_shape[0],  image_shape[1]))
clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8, 8))
for i in range(n_train):
    X_train_grey[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY, 0)
    X_train_grey_clahe[i] = clahe.apply(
        cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY))
    X_train_grey_norm[i] = (X_train_grey_clahe[i] / 255).astype(np.float32)

print('Images Test')
index = random.randint(0, n_train)
dst = generateRotatedImage(X_train[index])

plt.figure(figsize=(2, 2))
plt.imshow(dst)
plt.title('Rotate')
# for i in range(10):
plt.figure(figsize=(2, 2))
plt.imshow(X_train[index])
plt.title('Normal')
plt.figure(figsize=(2, 2))
plt.imshow(cv2.cvtColor(X_train[index][:32], cv2.COLOR_RGB2GRAY), cmap='gray')
plt.title('Gray')
plt.figure(figsize=(2, 2))
plt.imshow(X_train_grey_norm[index], cmap='gray')
plt.title('Gray Norm')
plt.figure(figsize=(2, 2))
plt.imshow(X_train_grey_clahe[index], cmap='gray')
plt.title('Gray Clahe')
plt.figure(figsize=(2, 2))


X_train_grey_norm = X_train_grey_norm.reshape(
    n_train, image_shape[0], image_shape[0], 1)
X_train = X_train_grey_norm

X_valid_grey = np.zeros((n_validation, image_shape[0],  image_shape[1]))
X_valid_grey_clahe = np.zeros((n_validation, image_shape[0],  image_shape[1]))
X_valid_grey_norm = np.zeros((n_validation, image_shape[0],  image_shape[1]))
for i in range(n_validation):
    X_valid_grey[i] = clahe.apply(cv2.cvtColor(
        X_valid[i][:32], cv2.COLOR_RGB2GRAY), 0)
    X_valid_grey_clahe[i] = clahe.apply(
        cv2.cvtColor(X_valid[i], cv2.COLOR_RGB2GRAY))
    X_valid_grey_norm[i] = (X_valid_grey_clahe[i] / 255).astype(np.float32)

X_valid_grey_norm = X_valid_grey_norm.reshape(
    n_validation, image_shape[0], image_shape[0], 1)
X_valid = X_valid_grey_norm

image_shape = (image_shape[0], image_shape[1], 1)

# print(X_train[0])
# X_train = ((X_train) / 255).astype(np.float32)

# X_valid = ((X_valid) / 255).astype(np.float32)
# plt.figure(figsize=(2,2))
# plt.imshow(X_train[0][:32] )
# plt.show()
X_train, y_train = shuffle(X_train, y_train)
# exit(1)
import tensorflow as tf

from tensorflow.contrib.layers import flatten


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, image_shape[2], 32), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[
                         1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    # conv1 = tf.nn.dropout(conv1, 0.5)
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 32, 64), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[
                         1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    # conv2 = tf.nn.dropout(conv2, 0.5)
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='VALID')

# SOLUTION: Layer 3: Convolutional. Output = 10x10x16.
    conv3_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 64, 128), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(128))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[
                         1, 1, 1, 1], padding='SAME') + conv3_b

    # SOLUTION: Activation.
    conv3 = tf.nn.relu(conv3)
    # conv3 = tf.nn.dropout(conv3, 0.5)
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    f1 = flatten(conv1)
    f2 = flatten(conv2)
    f3 = flatten(conv3)

    fc0 = tf.concat([f1, f2, f3], 1)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(
        shape=(9024, 3000), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(3000))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    fc1_1_W = tf.Variable(tf.truncated_normal(
        shape=(3000, 400), mean=mu, stddev=sigma))
    fc1_1_b = tf.Variable(tf.zeros(400))
    fc1 = tf.matmul(fc1, fc1_1_W) + fc1_1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)
    fc2_W = tf.Variable(tf.truncated_normal(
        shape=(400, 120), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(120))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc3_W = tf.Variable(tf.truncated_normal(
        shape=(120, 84), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(84))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b

    # SOLUTION: Activation.
    fc3 = tf.nn.relu(fc3)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc4_W = tf.Variable(tf.truncated_normal(
        shape=(84, n_classes), mean=mu, stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc3, fc4_W) + fc4_b

    return logits


x = tf.placeholder(
    tf.float32, (None, image_shape[0], image_shape[1], image_shape[2]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

EPOCHS = 20
BATCH_SIZE = 128

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset +
                                  BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={
                            x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = n_train

    print("Training...")

    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './trafficSignModel02')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
