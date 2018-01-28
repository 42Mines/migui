import cv2
from pylab import *
from matplotlib import *
import random as rd
from scipy import ndimage
import numpy as np
    
def extraire_feu_opt(filename,erosion=True,normalize=False,fraction_detectee=0.8, debug=False):
    img=cv2.cvtColor(filename,cv2.COLOR_BGR2HSV)
    #normalisation vers les valeurs moyennes S et V de la base de donnees (valeurs experimentales)
    if(normalize):
        H_mean,S_mean,V_mean = [  62.14123946,   99.81583164,  117.88332891]
        img[:,:,1] = np.vectorize(normalize_sigmoid)(img[:,:,1])
        img[:,:,2] = np.vectorize(normalize_sigmoid)(img[:,:,2])
    #determination des pixels de feu probables
    tab=np.zeros(np.shape(img)[:2])
    fiabilite=0
    #liste de valeurs empiriques mesurées sur des sets d'images pointees
    liEmpirique=[[3,97,128,134],[22,79,133,147],[34,80,136,155],[36,95,138,169],[47,89,142,181]]
    for bound in liEmpirique:
        fiabilite+=1/len(liEmpirique)
        tab[((img[:,:,0]+50)%180>bound[0])*((img[:,:,0]+50)%180<bound[1])*(img[:,:,1]>bound[2])*(img[:,:,2]>bound[3])]=fiabilite
    tot_original=np.sum(tab)  
    if debug:  
            cv2.imshow('detec-feu',1-tab)
    tab_feu=tab.copy()
    #tentative d'erosion de l'image pour "nettoyer" les pixels isoles
    if(erosion):
        tab_feu = ndimage.morphology.grey_erosion(tab,footprint=np.array([[ 0.,  0.,  1.,  0.,  0.],
                                                                   [ 0.,  1.,  1.,  1.,  0.],
                                                                   [ 1.,  1.,  1.,  1.,  1.],
                                                                   [ 0.,  1.,  1.,  1.,  0.],
                                                                   [ 0.,  0.,  1.,  0.,  0.]]))
        if(np.sum(tab_feu)<0.6*tot_original):
            tab_feu=ndimage.morphology.grey_erosion(tab,footprint=np.ones((3,3)))
        if(np.sum(tab_feu)<0.6*tot_original):
            tab_feu=ndimage.morphology.grey_erosion(tab,footprint=np.ones((2,2)))
        if(np.sum(tab_feu)<0.33*tot_original):
            tab_feu=tab.copy()
    if debug:
            cv2.imshow('detec-feu_erodee',1-tab_feu)
    
    #calcul du centre du feu (isobarycentre des fiabilites)
    L,l=tab_feu.shape[0],tab_feu.shape[1]
    tot=np.sum(tab_feu)
    if debug:
        print("tot=",tot)
    if(tot>0):
        center = ndimage.center_of_mass(tab_feu)[::-1]
        center=(int(center[0]+0.5),int(center[1]+0.5))
    else:
        center=(int(l/2+0.5),int(L/2+0.5))
    if debug:
        print("center=",center)
        img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
        cv2.circle(img,center,3,(255,0,0),-1)
    center=np.array(center,dtype=int)
    
    #calcul de la boite englobante
    size=min(32,min(center[0],min(l-center[0],min(center[1],L-center[1]))))
    xmin,xmax,ymin,ymax=0,0,0,0
    while(tot==0 or np.sum(tab_feu[xmin:xmax,ymin:ymax])<fraction_detectee*tot):
        xmin,xmax,ymin,ymax=center[0]-size,center[0]+size,center[1]-size,center[1]+size
        if(xmin<=0 or xmax>=l or ymin<=0 or ymax>=L):
            break
        size+=1
    #evaluation de la proba d'avoir un feu
    proba=0.01
    if(np.count_nonzero(tab[xmin:xmax,ymin:ymax])>0):
        proba = min(1,4/3*np.sum(tab[xmin:xmax,ymin:ymax])/np.count_nonzero(tab[xmin:xmax,ymin:ymax]))
    if debug:
        print("proba=",proba)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0))
        cv2.imshow("center",img)
        cv2.imshow("feu_isole",img[ymin:ymax+1,xmin:xmax+1])
        while(1):
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows() 
    return (xmin,ymin,xmax,ymax),proba


def normalize_sigmoid(y):
    return (-np.log(257/(y+1)-1)+np.log(256))*255/(-np.log(1/256)+np.log(256))

