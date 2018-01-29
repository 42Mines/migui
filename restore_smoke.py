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

IMAGE_SIZE = 64

def loadWithImg(img):
    img = cv2.resize(img, (64,64))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#Declaration des tenseurs

X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
Y_ = tf.placeholder(tf.float32, [None, 2]) #reponse attendue
lr = tf.placeholder(tf.float32) # Taux d'apprentissage
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
pkeep = tf.placeholder(tf.float32) #probabilite de droupout

# --- NEURAL NETWORK STRUCTURE ---

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


saver = tf.train.Saver()

def compute(img, debug=False):
    with tf.Session() as sess:  
        # Fonction servant à afficher les images d'activation
        def plotNNFilter(units,src):
            filters = units.shape[3]
            plt.figure(1, figsize=(20,20))
            n_columns = 6
            n_rows = math.ceil(filters / n_columns) + 1
            plt.subplot(n_rows, n_columns, 1)
            plt.imshow(src)
            for i in range(filters):
                plt.subplot(n_rows, n_columns, i+2)
                plt.title('Filter ' + str(i))
                print(type(units))
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
        if debug:
            plt.rcdefaults()
            fig, ax = plt.subplots()
            ax3= plt.subplot2grid((3, 3), (1, 0))
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
            ax2.set_title('Resultat du reseau de neurones')
            ax3.set_title('Image')
            ax3.imshow(img)
            plt.show()
            
            plt.imshow(np.reshape(img,[64,64,3]))
            plt.show()
            getActivations(Y1,img)
            plt.show()
            getActivations(Y2,img)
            plt.show()
            
    return res[0,1]

print(compute(cv2.imread("./a.jpg")))