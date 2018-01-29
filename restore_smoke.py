import tensorflow as tf
import tensorflowvisu
import math
import mnist2 as mnist_data
import cv2
import matplotlib.pyplot as plt
import numpy as np

### Description du programme
# Ce programme a pour but de charger le reseau de neurones fumee
# afin de l'utiliser sur des images reelles donnees en entree
# Il renvoit une probabilite entre 0 et 1 de detecter de la fumee
### Fin de la description



def compute(img):
    IMAGE_SIZE = 64

    def loadWithImg(img):
        img = cv2.resize(img, (64, 64))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Declaration des tenseurs

    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    Y_ = tf.placeholder(tf.float32, [None, 2])  # reponse attendue
    lr = tf.placeholder(tf.float32)  # Taux d'apprentissage
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)  # probabilite de droupout

    # --- NEURAL NETWORK STRUCTURE ---

    K = 4
    L = 8

    # noeuds fully connected
    R = 32

    # strides pour les convolutions
    strideK = 1
    strideL = 2

    # --- STRUCTURE DU RESEAU DE NEURONES ---

    # convolutional layers neurons
    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, K], stddev=0.1))  # 3x3 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))

    convOutputSize = IMAGE_SIZE // (strideK * strideL)

    # convolutionnal layer
    Wfc1 = tf.Variable(tf.truncated_normal([convOutputSize * convOutputSize * L, R], stddev=0.1))
    Bfc1 = tf.Variable(tf.constant(0.1, tf.float32, [R]))

    Wlast = tf.Variable(tf.truncated_normal([R, 2], stddev=0.1))
    Blast = tf.Variable(tf.constant(0.1, tf.float32, [2]))

    Y1l = tf.nn.conv2d(X, W1, strides=[1, strideK, strideK, 1], padding='SAME')
    Y1 = tf.nn.relu(Y1l)

    Y2l = tf.nn.conv2d(Y1, W2, strides=[1, strideL, strideL, 1], padding='SAME')
    Y2 = tf.nn.relu(Y2l)
    YY = tf.reshape(Y2, shape=[-1, convOutputSize * convOutputSize * L])

    # for fully connected layers
    Yfc1l = tf.matmul(YY, Wfc1)
    Yfc1r = tf.nn.leaky_relu(Yfc1l, alpha=0.3)
    Yfc1 = tf.nn.dropout(Yfc1r, pkeep)

    Ylogits = tf.matmul(Yfc1, Wlast) + Blast

    Y = tf.nn.softmax(Ylogits)  # sortie du reseau de neurones

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./tmp/smoke3")
        img = loadWithImg(img)
        res = sess.run(Y, {X:[img], pkeep:1})

    sess.close()
    tf.reset_default_graph()
    return res[0,1]
