import tensorflow as tf
import tensorflowvisu
import math
import mnist2 as mnist_data
import cv2
import matplotlib.pyplot as plt
import numpy as np

### FONCTIONS DE DEBUGGAGE
IMAGE_SIZE = 64


def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
    return noiseshape

def loadWithImg(img):
    img = cv2.resize(img, (64, 64))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


### FIN FONCTIONS DEBUGGAGE

def predict(img):
    img = cv2.resize(img, (64, 64))
    with tf.Session() as sess:
        saver.restore(sess, "./tmp/smoke2")
        return sess.run(Y, {X: [img], pkeep: 1})[1]


def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
    return noiseshape


###Structure du réseau de neurones
X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 2])
# variable learning rate
lr = tf.placeholder(tf.float32)
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# dropout probability
pkeep = tf.placeholder(tf.float32)

# learning rates params
max_learning_rate = 0.001
min_learning_rate = 0.0005
decay_speed = 1000

# convolutional layers output depth
K = 5
L = 6
M = 6
# N = 4

<<<<<<< HEAD
K = 4
L = 8

#noeuds fully connected
R = 32

# strides pour les convolutions
=======
# fully connected layers output
R = 50
# S = 50

# strides for convolutional layers
>>>>>>> c0300233c0cd0952c7d0d8f8045a007311ec4eb3
strideK = 1
strideL = 2
strideM = 2
# strideN = 2

<<<<<<< HEAD
# --- STRUCTURE DU RESEAU DE NEURONES ---

# convolutional layers neurons
W1 = tf.Variable(tf.truncated_normal([5, 5, 3, K], stddev=0.1))  # 3x3 patch, 1 input channel, K output channels
=======
# 16 * 49* 3
# 8 * 25 * 16
# 32 * 32 * 8 * 300 =
# 300 * 2 = 600
# --- NEURAL NETWORK STRUCTURE ---

# convolutional layers neurons
W1 = tf.Variable(tf.truncated_normal([7, 7, 3, K], stddev=0.1))  # 3x3 patch, 1 input channel, K output channels
>>>>>>> c0300233c0cd0952c7d0d8f8045a007311ec4eb3
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([3, 3, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
<<<<<<< HEAD

convOutputSize = IMAGE_SIZE // (strideK * strideL)  

# convolutionnal layer
Wfc1 = tf.Variable(tf.truncated_normal([convOutputSize * convOutputSize * L, R], stddev=0.1))
=======
W3 = tf.Variable(tf.truncated_normal([3, 3, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
# W4 = tf.Variable(tf.truncated_normal([3, 3, M, N], stddev=0.1))
# B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))

# fully connected layers neurons
convOutputSize = IMAGE_SIZE // (strideK * strideL * strideM)  # size of the image produced by the last
# convolutionnal layer
Wfc1 = tf.Variable(tf.truncated_normal([convOutputSize * convOutputSize * M, R], stddev=0.1))
>>>>>>> c0300233c0cd0952c7d0d8f8045a007311ec4eb3
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
<<<<<<< HEAD
Y2 = tf.nn.relu(Y2l)
YY = tf.reshape(Y2, shape=[-1, convOutputSize * convOutputSize * L])

=======
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

>>>>>>> c0300233c0cd0952c7d0d8f8045a007311ec4eb3
# for fully connected layers
Yfc1l = tf.matmul(YY, Wfc1)
Yfc1r = tf.nn.leaky_relu(Yfc1l, alpha=0.3)
Yfc1 = tf.nn.dropout(Yfc1r, pkeep)
<<<<<<< HEAD


Ylogits = tf.matmul(Yfc1, Wlast) + Blast

Y = tf.nn.softmax(Ylogits) #sortie du reseau de neurones

=======

# Yfc2l = tf.matmul(Yfc1, Wfc2)
# Yfc2bn, update_ema6 = no_batchnorm(Yfc2l, tst, iter, Bfc2)
# Yfc2r = tf.nn.relu(Yfc2bn)
# Yfc2 = tf.nn.dropout(Yfc2r, pkeep)
>>>>>>> c0300233c0cd0952c7d0d8f8045a007311ec4eb3

Ylogits = tf.matmul(Yfc1, Wlast) + Blast
Y = tf.nn.softmax(Ylogits)
saver = tf.train.Saver()

def compute(img, debug=False):
    with tf.Session() as sess:

        def plotNNFilter(units, src):
            filters = units.shape[3]
            plt.figure(1, figsize=(20, 20))
            n_columns = 6
            n_rows = math.ceil(filters / n_columns) + 1
            plt.subplot(n_rows, n_columns, 1)
            plt.imshow(src)
            for i in range(filters):
                plt.subplot(n_rows, n_columns, i + 2)
                plt.title('Filter ' + str(i))
                print(type(units))
<<<<<<< HEAD
                plt.imshow(units[0,:,:,i], interpolation="nearest",cmap='gray')

        def getActivations(layer,stimuli):
            units = sess.run(layer,feed_dict={X:[stimuli],pkeep:1.0})
            plotNNFilter(units,stimuli)
        ### CHARGEMENT DU RESEAU DE NEURONES
        saver.restore(sess, "./tmp/smoke3")
        img = loadWithImg(img)
        res = sess.run(Y, {X:[img], pkeep:1})
        ### MODE DEBUG :###
        # Affichage graphique de la probabilite donnee par le reseau de neurones
        # a l'aide de matplotlib.pyplot 
=======
                plt.imshow(units[0, :, :, i], interpolation="nearest", cmap='gray')

        def getActivations(layer, stimuli):
            units = sess.run(layer, feed_dict={X: [stimuli], pkeep: 1.0})
            plotNNFilter(units, stimuli)

        saver.restore(sess, "./tmp/smoke2")
        img = np.multiply(loadWithImg(img), 1.0 / 255.0)
        #plt.imshow(img)
        #plt.show()
        res = sess.run(Y, {X: [img], pkeep: 1})
>>>>>>> c0300233c0cd0952c7d0d8f8045a007311ec4eb3
        if debug:
            plt.rcdefaults()
            fig, ax = plt.subplots()
            ax3 = plt.subplot2grid((3, 3), (1, 0))
            ax2 = plt.subplot2grid((3, 3), (1, 1), colspan=2)
            # Example data
            people = ('Feu', 'Rien')
            y_pos = np.arange(2)
            performance = res[0][::-1]

            ax2.barh(y_pos, performance, align='center',
                     color='red', ecolor='black')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(people)
            ax2.invert_yaxis()  # labels read top-to-bottom
            ax2.set_xlabel('Performance')
            ax2.set_title('Résultat du réseau de neurones')
            ax3.set_title('Image')
            ax3.imshow(img)
            plt.show()
            """
            plt.imshow(np.reshape(img,[64,64,3]))
            plt.show()
            getActivations(Y1,img)
            plt.show()
            getActivations(Y2,img)
            plt.show()
<<<<<<< HEAD
            
    return res[0,1]

print(compute(cv2.imread("./a.jpg")))
=======
            """
    return res[0, 1]
>>>>>>> c0300233c0cd0952c7d0d8f8045a007311ec4eb3
