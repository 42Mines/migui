import matplotlib.patches as patches
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def selectZones(filename, number_zones):

    """"
    Prend en paramètre un chemin d'accès vers une image et un nombre de zones à isoler
    Retourne un tableau de taille number_zones d'images isolées de potentielle fumée

    """

    src = cv2.imread(filename, cv2.IMREAD_COLOR)[:, :, ::-1]
    lg = src.shape[0]
    Lg = src.shape[1]
    
    miniL = min(src.shape[0], src.shape[1])
    src = cv2.medianBlur(src, 5)
    src = cv2.resize(src,(min(miniL,200),min(miniL,200)))
    deltalg = lg/src.shape[0]
    deltaLG = Lg/src.shape[1]
    t = time.time()
    def findPossibleSmoke(src):

        smoke_pixels = np.zeros((src.shape[0], src.shape[1]))
        R = src[:, :, 0]
        G = src[:, :, 1]
        B = src[:, :, 2]
        I = (1/3)*(R+G+B)+1e-10
        M =  np.minimum(R, G, B)
        c = 1 - M/I < 0.1
        smoke_pixels[c] = 1

        return smoke_pixels
    def findPossibleSmoke2(src):

        smoke_pixels = np.zeros((src.shape[0], src.shape[1]))
        for i in range(0, src.shape[0]):
            for j in range(0, src.shape[1]):
                r = int(src[i, j, 0])
                g = int(src[i, j, 1])
                b = int(src[i, j, 2])
                m = min(r,g,b)
                I = 1/3*(r+g+b)

                if I!=0 and 1 - m/I < 0.1:
                    smoke_pixels[i,j] = 1

        return smoke_pixels
    def connex_components(smoke_pixels):
        l = smoke_pixels.shape[0]
        L = smoke_pixels.shape[1]
        visited = np.zeros((l, L), dtype=np.int)
        curCC = 1
        deplacements = [(-1, 0), (0, 1), (0, -1), (1, 0)]
        compactCC = []
        for i in range(0, l):
            for j in range(0, L):
                if visited[i, j] == 0:
                    pile = [(i, j)]
                    nXMax = i
                    nXMin = i
                    nYMax = j
                    nYMin = j
                    total = 1
                    while len(pile) > 0:
                        cur = pile.pop()
                        for dep in deplacements:
                            dx = dep[0]
                            dy = dep[1]
                            nX = cur[0] + dx
                            nY = cur[1] + dy
                            if nX >= 0 and nY >= 0 and nX < l and nY < L and visited[nX, nY] == 0 and smoke_pixels[nX, nY]:
                                pile.append((nX, nY))
                                visited[nX, nY] = curCC
                                nXMax = max(nXMax, nX)
                                nXMin = min(nXMin, nX)
                                nYMax = max(nYMax, nY)
                                nYMin = min(nYMin, nY)
                                total += 1
                    if total > 2000:
                        compactCC.append([curCC, total, nXMin, nYMin, nXMax, nYMax])
                    curCC += 1
        compactCC.sort(key=lambda l:l[1], reverse=True)
        return compactCC, visited
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.imshow(src)
    """
    smoke_pixels = findPossibleSmoke2(src)
    """
    ax2 = fig.add_subplot(222)
    ax2.imshow(smoke_pixels,cmap='gray')
    """
    nsmoke_pixels = cv2.morphologyEx(smoke_pixels, cv2.MORPH_OPEN, np.ones((3,3)))
    """
    ax3=fig.add_subplot(223)
    ax3.imshow(nsmoke_pixels,cmap='gray')
    """
    compactCC, visited = connex_components(nsmoke_pixels)
    #imgs = []
    answer = []
    print(len(compactCC))
    for i in range(0, min(number_zones, len(compactCC))):
        nXMax = compactCC[i][4]
        nXMin = compactCC[i][2]
        nYMax = compactCC[i][5]
        nYMin = compactCC[i][3]
        answer.append((nXMin, nYMin, nXMax,nYMax))
        #imgs.append(src[nXMin:nXMax, nYMin:nYMax, :])
    """
    keptCC = compactCC[0:min(len(compactCC), number_zones)]
    availNamesCC = []
    for i in range(0, len(keptCC)):
        availNamesCC.append(keptCC[i][0])

    for i in range(0, src.shape[0]):
        for j in range(src.shape[1]):
            if visited[i, j] in availNamesCC:
                nsmoke_pixels[i, j] = 1
            else:
                nsmoke_pixels[i, j] = 0

    u = (time.time() - t)
    """
    return answer,deltalg,deltaLG

#selectZones("C:/Users/Victor/Desktop/Mines_ParisTech/MIG/img/fumee_bdd/1420085.jpg",4)
#selectZones("C:/Users/Victor/Desktop/Mines_ParisTech/MIG/img/fumee_bdd/1450134.jpg",4)
#   selectZones("C:/Users/Victor/tensorflow-mnist-tutorial/algo/test/1 (1).jpg",4)