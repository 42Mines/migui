import interface_feu
import interface_fumee
import cv2

from os import listdir
from os.path import isfile

folder = "./uploads/"

for file in listdir(folder):
    if isfile(folder + file) and (file.find("jpg") >= 0 or file.find("jpeg") >= 0) and file.find("fire") == -1 and file.find("smoke") == -1:
        if isfile(folder + "fire_" + file):
            continue

        # feu
        (x1, y1, x2, y2), proba = interface_feu.first_fire_pass(cv2.imread(folder + file))

        img = cv2.imread(folder + file)

        if proba > 0.1:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        cv2.imwrite(folder + "fire_" + file, img)

        metadata = open(folder + "metadata_fire_" + file.replace(".jpeg", "").replace(".jpg", "") + ".txt", "w")
        metadata.write(str(interface_feu.confirm_fire(img, x1, y1, x2, y2)))
        metadata.close()

        # fumée
        zones_fumee_potentielle = interface_fumee.first_smoke_pass(folder + file)
        img = cv2.imread(folder + file)
        metadata = open(folder + "metadata_smoke_" + file.replace(".jpeg", "").replace(".jpg", "") + ".txt", "w")

        for i in range(len(zones_fumee_potentielle)):
            y1, x1, y2, x2 = zones_fumee_potentielle[i]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            metadata.write(str(interface_fumee.confirm_smoke(cv2.imread(folder + file), x1, y1, x2, y2)) + "\n")

        cv2.imwrite(folder + "smoke_" + file, img)
        metadata.close()
