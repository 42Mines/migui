import math
import random
import tensorflow as tf
import numpy as np
import mnist2 as mnist_data
import tensorflowvisu

tf.set_random_seed(0)


def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
    return noiseshape

# learning rates params
max_learning_rate = 0.001
min_learning_rate = 0.00001
decay_speed = 2000

BATCH_SIZE = 200
TEST_BATCH_SIZE = 1000
IMAGE_SIZE = 64
<<<<<<< HEAD
NB_ITERATIONS = 1000
TRAIN_UPDATE_FREQ = 10
TEST_UPDATE_FREQ = 30
PATH = "C:/Users/Victor/Desktop/Mines_ParisTech/MIG/BDD/BDD_FUMEE/"
=======
NB_ITERATIONS = 50
TRAIN_UPDATE_FREQ = 5
TEST_UPDATE_FREQ = 40
PATH = "/Users/adrienbocquet/Desktop/BDD/BDD_FUMEE/"
>>>>>>> c0300233c0cd0952c7d0d8f8045a007311ec4eb3

mnist = mnist_data.read_data_sets(PATH, one_hot=True, reshape=False, mode="smoke")
IMAGE_SIZE = 64

lr = tf.placeholder(tf.float32) # Taux d'apprentissage
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
pkeep = tf.placeholder(tf.float32) #probabilite de dropout
PKEEP = 0.95 #probabilite de droupout
#Declaration des tenseurs

<<<<<<< HEAD
# --- PARAMETRES DU RESEAU DE NEURONES ---

PKEEP = 0.95

# learning rates params
max_learning_rate = 0.0006
min_learning_rate = 0.00001
decay_speed = 2000
=======
###Structure du réseau de neurones
X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 2])
# variable learning rate
>>>>>>> c0300233c0cd0952c7d0d8f8045a007311ec4eb3

# convolutional layers output depth
K = 5
L = 6
M = 6
# N = 4

# fully connected layers output
R = 50
# S = 50

# strides for convolutional layers
strideK = 1
strideL = 2
strideM = 2
# strideN = 2

# 16 * 49* 3
# 8 * 25 * 16
# 32 * 32 * 8 * 300 =
# 300 * 2 = 600
# --- NEURAL NETWORK STRUCTURE ---

# convolutional layers neurons
W1 = tf.Variable(tf.truncated_normal([7, 7, 3, K], stddev=0.1))  # 3x3 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([3, 3, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([3, 3, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
# W4 = tf.Variable(tf.truncated_normal([3, 3, M, N], stddev=0.1))
# B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))

# fully connected layers neurons
convOutputSize = IMAGE_SIZE // (strideK * strideL * strideM)  # size of the image produced by the last
# convolutionnal layer
Wfc1 = tf.Variable(tf.truncated_normal([convOutputSize * convOutputSize * M, R], stddev=0.1))
Bfc1 = tf.Variable(tf.constant(0.1, tf.float32, [R]))

# Wfc2 = tf.Variable(tf.truncated_normal([R, S], stddev=0.1))
# Bfc2 = tf.Variable(tf.constant(0.1, tf.float32, [S]))

Wlast = tf.Variable(tf.truncated_normal([R, 2], stddev=0.1))
Blast = tf.Variable(tf.constant(0.1, tf.float32, [2]))

# For convolutional layers
Y1l = tf.nn.conv2d(X, W1, strides=[1, strideK, strideK, 1], padding='SAME')
Y1 = tf.nn.relu(Y1l)
# Y1 = tf.nn.dropout(Y1r, pkeep, compatible_convolutional_noise_shape(Y1r))

Y2l = tf.nn.conv2d(Y1, W2, strides=[1, strideL, strideL, 1], padding='SAME')
Y2r = tf.nn.relu(Y2l)
Y2 = tf.nn.dropout(Y2r, pkeep, compatible_convolutional_noise_shape(Y2r))

Y3l = tf.nn.conv2d(Y2, W3, strides=[1, strideM, strideM, 1], padding='SAME')
# Y3bn = no_batchnorm(Y3l, tst, iter, B3, convolutional=True)
Y3 = tf.nn.relu(Y3l)
# Y3 = tf.nn.dropout(Y3r, pkeep, compatible_convolutional_noise_shape(Y3r))

# Y4l = tf.nn.conv2d(Y3, W4, strides=[1, strideN, strideN, 1], padding='SAME')
# Y4bn, update_ema4 = no_batchnorm(Y4l, tst, iter, B4, convolutional=True)
# Y4r = tf.nn.leaky_relu(Y4bn)
# Y4 = tf.nn.dropout(Y4r, pkeep, compatible_convolutional_noise_shape(Y4r))

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, convOutputSize * convOutputSize * M])

# for fully connected layers
Yfc1l = tf.matmul(YY, Wfc1)
Yfc1r = tf.nn.leaky_relu(Yfc1l, alpha=0.3)
Yfc1 = tf.nn.dropout(Yfc1r, pkeep)

# Yfc2l = tf.matmul(Yfc1, Wfc2)
# Yfc2bn, update_ema6 = no_batchnorm(Yfc2l, tst, iter, Bfc2)
# Yfc2r = tf.nn.relu(Yfc2bn)
# Yfc2 = tf.nn.dropout(Yfc2r, pkeep)

Ylogits = tf.matmul(Yfc1, Wlast) + Blast
Y = tf.nn.softmax(Ylogits)

### Calcul de parametres de controle

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * BATCH_SIZE
0
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tmp1 = tf.constant(1, dtype=tf.int64, shape=[TEST_BATCH_SIZE])
# tenseur contenant les indices des labels avec une image positive
indices_positifs = tf.where(tf.equal(tf.argmax(Y_, 1), tmp1))
# tenseur contenant uniquement les labels d'images positives
labels_positifs = tf.gather(Y_, indices_positifs)
labels_positifs = tf.reshape(labels_positifs, [tf.size(indices_positifs), 2])
# tenseur contenant uniquement les predictions face à des images positives
predictions_positifs = tf.gather(Y, indices_positifs)
predictions_positifs = tf.reshape(predictions_positifs, [tf.size(indices_positifs), 2])
# precision sur les images positives
correct_prediction_pos = tf.equal(tf.argmax(labels_positifs, 1), tf.argmax(predictions_positifs, 1))
accuracy_pos = tf.reduce_mean(tf.cast(correct_prediction_pos, tf.float32))

tmp2 = tf.constant(0, dtype=tf.int64, shape=[TEST_BATCH_SIZE])
# tenseur contenant les indices des labels avec une image negative
indices_negatifs = tf.where(tf.equal(tf.argmax(Y_, 1), tmp2))
# tenseur contenant uniquement les labels d'images negatives
labels_negatifs = tf.gather(Y_, indices_negatifs)
labels_negatifs = tf.reshape(labels_negatifs, [tf.size(indices_negatifs), 2])
# tenseur contenant uniquement les predictions face à des images negatives
predictions_negatifs = tf.gather(Y, indices_negatifs)
predictions_negatifs = tf.reshape(predictions_negatifs, [tf.size(indices_negatifs), 2])
# precision sur les images negatives
correct_prediction_neg = tf.equal(tf.argmax(labels_negatifs, 1), tf.argmax(predictions_negatifs, 1))
accuracy_neg = tf.reduce_mean(tf.cast(correct_prediction_neg, tf.float32))

# matplotlib visualisation
allweights = tf.concat(
    [tf.reshape(W1, [-1]), tf.reshape(Wfc1, [-1])], 0)
allbiases = tf.concat(
    [tf.reshape(B1, [-1]), tf.reshape(Bfc1, [-1])], 0)
dense_activations = tf.reduce_max(Yfc1r, [0])
datavis = tensorflowvisu.MnistDataVis(histogram4colornum=2, histogram5colornum=2)

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy) # Choix de l'optimiseur

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

def training_step(i, update_test_data, update_train_data):
    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)

    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

    if update_train_data:
        acc, c, res, w, b = sess.run([accuracy, cross_entropy, Y, allweights, allbiases],{X: batch_x, Y_: batch_y, tst: False, pkeep: PKEEP})
        print(str(i) + ": accuracy:" + str(acc) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        #datavis.append_training_curves_data(i, acc, c)
        #datavis.append_data_histograms(i, w, b)

    if update_test_data:
        test_index = random.sample(range(len(mnist.validation.images)), TEST_BATCH_SIZE)
        acc, acc_pos, acc_neg, c,y = sess.run([accuracy, accuracy_pos, accuracy_neg, cross_entropy,Y],{X: [mnist.validation.images[i] for i in test_index],Y_: [mnist.validation.labels[i] for i in test_index], tst: True, pkeep: PKEEP})
        print(str(i) + ": ********* epoch " + str(i * 100 // mnist.train.images.shape[0] + 1) + "accuracy:" + str(acc)+ " test loss: " + str(c))
        #datavis.append_test_curves_data(i, acc, acc_pos, acc_neg, c)
        print(y)
        print("acc pos, neg, et globale : " + str(acc_pos) + ", " + str(acc_neg) + ", " + str(acc))

    # the backpropagation training step
    sess.run(train_step, {X: batch_x, Y_: batch_y, lr: learning_rate, tst: False, pkeep: PKEEP})


datavis.animate(training_step, iterations=NB_ITERATIONS, train_data_update_freq=TRAIN_UPDATE_FREQ, test_data_update_freq=TEST_UPDATE_FREQ, more_tests_at_start=True)


print("--- Training finished ---")
#print("average test accuracy: " + str(datavis.get_average_test_accuracy()))
#Sauvegarde du reseau de neurones
<<<<<<< HEAD
saver.save(sess,'./tmp/smoke3')
=======
saver.save(sess,'./tmp/smoke2')
>>>>>>> c0300233c0cd0952c7d0d8f8045a007311ec4eb3
