import tensorflow as tf
import pickle
import cv2
import os
import numpy as np
import matplotlib.image as mpimg
import argparse
import matplotlib.pyplot as plt
import csv

signs = {}
with open('./signnames.csv') as signnames:
    reader = csv.reader(signnames, skipinitialspace=True)
    for row in reader:
        signs[int(row[0])] = row[1]

for filename in os.listdir('Web'):

    image = mpimg.imread('Web/'+filename)

    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)

    # print("Image shape: ",  image.shape)
    # plt.figure(figsize=(1, 1))
    # plt.imshow(image)
    clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8, 8))

    image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # plt.figure(figsize=(1, 1))
    # plt.imshow(image_grey, cmap='gray')
    # plt.show()

    image_grey_clahe = clahe.apply(image_grey)
    # image_grey_norm = image_grey_clahe
    # image_grey_norm = cv2.normalize(
    # image_grey_norm, image_grey_norm, 0, 1, cv2.NORM_MINMAX)
    image_grey_norm = (image_grey_clahe / 255).astype(np.float32)
    print(image_grey_norm.shape)
    image_grey_norm = np.reshape(
        image_grey_norm, (1, image_grey_norm.shape[0], image_grey_norm.shape[1], 1))
    # X_test = X_test_grey_norm

    # image_shape = (image_shape[0], image_shape[1], 1)
    # EPOCHS = 20
    # BATCH_SIZE = 128
    # n_classes = 43

    with tf.Session() as sess:

        saver = tf.train.import_meta_graph('trafficSignModel01.meta')
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        # [print(n.name) for n in tf.get_default_graph().get_operations()]
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("Placeholder:0")
        y = graph.get_tensor_by_name("Placeholder_1:0")
        one_hot_y = graph.get_tensor_by_name("one_hot:0")
        # op_to_restore = graph.get_tensor_by_name(
        # "softmax_cross_entropy_with_logits:0")
        op_to_restore = graph.get_tensor_by_name(
            "add_7:0")
        res = sess.run(op_to_restore, feed_dict={
            x: image_grey_norm, y: [27]})

        #
        # print(res[0].index(max(res[0])))
        values, indices = sess.run(tf.nn.top_k(tf.constant(res), k=5))
        print(signs[indices[0][0]])
        # print(top5)

        # for i, classification in enumerate(res[0]):
        # print(i, classification)
        # print(res)
        # print(np.unravel_index(np.argmax(res[0], axis=None), res[0].shape))
