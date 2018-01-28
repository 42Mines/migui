import math
import random
import tensorflow as tf
import numpy as np
import mnist2 as mnist_data
import tensorflowvisu

tf.set_random_seed(0)

BATCH_SIZE = 200
TEST_BATCH_SIZE = 1000
IMAGE_SIZE = 64
NB_ITERATIONS = 3000
TRAIN_UPDATE_FREQ = 5
TEST_UPDATE_FREQ = 50
PATH = "C:/Users/Victor/Desktop/Mines_ParisTech/MIG/Codes/TestNN/BDD_FUMEE/"

mnist = mnist_data.read_data_sets(PATH, one_hot=True, reshape=False, mode="smoke")

X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
# reponses attendues
Y_ = tf.placeholder(tf.float32, [None, 2])
# learning rate
lr = tf.placeholder(tf.float32)
# test flag
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
pkeep = tf.placeholder(tf.float32) #probabilite de dropout

# --- PARAMETRES DU RESEAU DE NEURONES ---

PKEEP = 0.95

# learning rates params
max_learning_rate = 0.001
min_learning_rate = 0.00001
decay_speed = 2000

#nombre de couches de convolution
K = 4
L = 8

#noeuds fully connected
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
Yfc1r = tf.nn.leaky_relu(Yfc1l,alpha = 0.3)
Yfc1 = tf.nn.dropout(Yfc1r, pkeep)


Ylogits = tf.matmul(Yfc1, Wlast) + Blast

Y = tf.nn.softmax(Ylogits) #sortie du reseau de neurones

### Calcul de parametres de controle

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * BATCH_SIZE

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
        datavis.append_training_curves_data(i, acc, c)
        datavis.append_data_histograms(i, w, b)

    if update_test_data:
        test_index = random.sample(range(len(mnist.validation.images)), TEST_BATCH_SIZE)
        acc, acc_pos, acc_neg, c,y = sess.run([accuracy, accuracy_pos, accuracy_neg, cross_entropy,Y],{X: [mnist.validation.images[i] for i in test_index],Y_: [mnist.validation.labels[i] for i in test_index], tst: True, pkeep: PKEEP})
        print(str(i) + ": ********* epoch " + str(i * 100 // mnist.train.images.shape[0] + 1) + "accuracy:" + str(acc)+ " test loss: " + str(c))
        datavis.append_test_curves_data(i, acc, acc_pos, acc_neg, c)
        print(y)
        print("acc pos, neg, et globale : " + str(acc_pos) + ", " + str(acc_neg) + ", " + str(acc))

    # the backpropagation training step
    sess.run(train_step, {X: batch_x, Y_: batch_y, lr: learning_rate, tst: False, pkeep: PKEEP})


datavis.animate(training_step, iterations=NB_ITERATIONS, train_data_update_freq=TRAIN_UPDATE_FREQ, test_data_update_freq=TEST_UPDATE_FREQ,
                more_tests_at_start=True)


print("--- Training finished ---")
print("max test accuracy: " + str(datavis.get_max_test_accuracy()))
print("average test accuracy: " + str(datavis.get_average_test_accuracy()))
#Sauvegarde du reseau de neurones
saver.save(sess,'./tmp/smoke')