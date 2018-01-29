import restore_smoke
import restore_fire
import detec_zones_fumee
import cv2
import numpy as np
import matplotlib.pyplot as plt



def first_smoke_pass(img, max_zones_detected = 4):
    """
        Repère dans une image source les sous-rectangles susceptibles de contenir de la fumée.
        Cette fonction correspond aux methodes traditionnelles (ie OpenCV)

        :param img: une image de taille quelconque
        :return: [(int, int, int, int)] une liste de tuple contenant la liste des coordonnees des fumées sur l'image
                sous la forme (xa, ya, xb, yb) avec a le coin haut gauche
    """
    
    return detec_zones_fumee.selectZones(img,max_zones_detected)

def confirm_smoke(img, x1, y1, x2, y2):
    """
        Associe à l'image une probabilite que l'image contienne effectivement un feu de foret.
        Utilise les reseaux de neurones

       :param img: une image de taille quelconque, qui montre essentiellement la zone suspecte
       :param (x1, y1, x2, y2): les coordonnees des coins opposes (haut gaut, bas droite) du carre
       :return: double la probabilite qu'on ait bien du feu
    """

    img = cv2.medianBlur(img,5)[x1:x2,y1:y2]
    probas2 = restore_smoke.compute(img)
    return probas2


def proba_fumee(fn):
    """
    Prend en paramètre une image et renvoie un couple(boundingbox, probabilite qu'il y ait de la fumée sur l'image)
    après verification par les 2 algos

    """
    img = cv2.imread(fn)
    INPUT_AREA  = 64*64
    MAX_INPUT_AREA = 4 * INPUT_AREA

    """
    Fonction auxiliaire pour redécouper une zone trop grande fournie en entrée au réseau de neurones
    """

    def decoupe(x1,y1,x2,y2):
        decoupe_x = (x2-x1)//64
        decoupe_y = (y2-y1)//64
        tmp = []
        for i in range(0,decoupe_x):
            for j in range(0,decoupe_y):
                if i==decoupe_x-1:
                    xM = x2
                else:
                    xM = x1+(i+1)*64
                if i==decoupe_y-1:
                    yM = y2
                else:
                    yM = y1+(j+1)*64
                tmp.append((x1+i*64,y1+j*64,xM,yM))
        return tmp

    res = first_smoke_pass(fn)

   
    ### Adaptation des zones fournies en entrée
    for i in range(0,len(res)):
        x1, y1,x2,y2 = res[i]
        area = abs((y2-y1)*(x2-x1))
        if area > MAX_INPUT_AREA:
            tmp = decoupe(x1, y1, x2, y2)
            del res[i]
            res += tmp


    xMin,yMin, xMax, yMax = 0,0,0,0
    max_proba = 0

    for elem in res:
        proba_tmp = confirm_smoke(img, elem[0],elem[1], elem[2],elem[3])
        if proba_tmp == max_proba and proba_tmp != 0:
            xMin = min(xMin, elem[0])
            yMin = min(yMin, elem[1])
            xMax = max(xMax, elem[2])
            yMax = max(yMax, elem[3])
        if proba_tmp > max_proba:
            max_proba = proba_tmp
            xMin = elem[0]
            yMin = elem[1]
            xMax = elem[2]
            yMax = elem[3]

    ### Affichage du résultat
    return (xMin,yMin,xMax,yMax,max_proba)


