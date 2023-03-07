import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import argparse
import time


# инициализирование обнаружение объектов
od = ObjectDetection()

# сборка аргументов парсера
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
args = vars(parser.parse_args())


# загрузка видео
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')


# получаем высоту и ширину кадра    
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = "videoTZ"
# определяем кодек и создаем видео которое будет на выходе 
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))


count = 0
# список точек
center_points = []

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # обнаружение объектов на кадре
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int(y + h)
        # добавление координат центральной точки в список
        center_points.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    for pt in center_points:
        cv2.circle(frame, pt, 1, (0, 0, 255), -3)
        

    cv2.imshow("Frame", frame)
    out.write(frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
