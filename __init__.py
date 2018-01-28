import restore_fire
import detec_zone_flammes
import cv2
import numpy as np
import matplotlib.pyplot as plt

PRIORITE_COME = 0.2
PRIORITE_CNN = 0.8
EPSILON = 10**-1

def first_fire_pass(img):
    """
        Repère dans une image source les sous-rectangles susceptibles de contenir du feu.
        Cette fonction correspond aux methodes traditionnelles (ie OpenCV)

        :param img: une image de taille quelconque
        :return: [(int, int, int, int), bool] une liste de tuple contenant la liste des coordonnees des feux sur l'image
                sous la forme (xa, ya, xb, yb) avec a le coin haut gauche, b le coin bas droite et si du feu a ete detecte
    """
    
    return detec_zone_flammes.extraire_feu_opt(img)

def confirm_fire(img, x1, y1, x2, y2):
    """
        Associe à l'image une probabilite que l'image contienne effectivement un feu de foret.
        Utilise les reseaux de neurones

       :param img: une image de taille quelconque, qui montre essentiellement la zone suspecte
       :param (x1, y1, x2, y2): les coordonnees des coins opposes (haut gaut, bas droite) du carre
       :return: double la probabilite qu'on ait bien du feu
    """

    img_ = cv2.medianBlur(img,5)[x1:x2,y1:y2]
    probas2 = restore_fire.compute(img)
    return probas2


def proba_fire(img):
    """
    Prend en paramètre une image et renvoie un couple(boundingbox, probabilite qu'il y ait un feu sur l'image)
    après verification par les 2 algos

    """
    res = first_fire_pass(img)
    proba1 = res[1]
    x1,y1,x2,y2 = res[0]
    proba2 = confirm_fire(img, x1, y1,x2,y2)
    if proba2==0:
        proba2 = EPSILON #pour eviter d'avoir 0
    proba = (proba1**PRIORITE_COME*proba2**PRIORITE_CNN)**(PRIORITE_COME*PRIORITE_CNN)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax3= plt.subplot2grid((3, 3), (1, 0))
    ax2 = plt.subplot2grid((3, 3), (1, 1), colspan=2)
    people = ('Feu', 'Rien')
    y_pos = np.arange(2)
    performance = [proba, 1-proba]      
    ax2.barh(y_pos, performance, align='center',
            color='red', ecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(people)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Performance')
    ax2.set_title('Resultat du reseau de neurones')
    ax3.set_title('Image')
    ax3.imshow(img[:,:,::-1])
    plt.show()
    return (x1,y1,x2,y2,proba)

#Exemple d'utilisation
#proba_fire(cv2.imread("./test/1 (7).jpg"))

