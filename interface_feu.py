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

    img = cv2.medianBlur(img, 5)[y1:y2, x1:x2]
    probas2 = restore_fire.compute(img)
    return probas2
