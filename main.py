import interface
import cv2

from os import listdir
from os.path import isfile

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def process():
    folder = "./uploads/"
    for file in listdir(folder):
        if isfile(folder + file) and (file.find("jpg") >= 0 or file.find("jpeg") >= 0) and file.find("fire") == -1:
            print(folder + file)
            (x1, y1, x2, y2), _ = interface.first_fire_pass(cv2.imread(folder + file))

            img = cv2.imread(folder + file)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.imwrite(folder + "fire_" + file, img)

            metadata = open(folder + "metadata_" + file.replace(".jpeg", "").replace(".jpg", "") + ".txt", "w")
            metadata.write(str(interface.confirm_fire(img, x1, y1, x2, y2)))


class EventHandler(FileSystemEventHandler):
    """Logs all the events captured."""

    def on_created(self, event):
        super(EventHandler, self).on_created(event)

        try:
            path = event.src_path
            if (path.find("jpg") >= 0 or path.find("jpeg") >= 0) and path.find("fire") == -1:
                folder = "./uploads/"
                for file in listdir(folder):
                    if isfile(folder + file) and (file.find("jpg") >= 0 or file.find("jpeg") >= 0) and file.find(
                            "fire") == -1:
                        print(folder + file)
                        (x1, y1, x2, y2), _ = interface.first_fire_pass(cv2.imread(folder + file))

                        img = cv2.imread(folder + file)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.imwrite(folder + "fire_" + file, img)

                        metadata = open(folder + "metadata_" + file.replace(".jpeg", "").replace(".jpg", "") + ".txt","w")
                        print(interface.confirm_fire(img, x1, y1, x2, y2))
                        metadata.write(str(interface.confirm_fire(img, x1, y1, x2, y2)))
                        metadata.write("coucou")
        except Exception as a:
            print(a)


event_handler = EventHandler()
observer = Observer()
observer.schedule(event_handler, "./uploads")
observer.start()

import time

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
