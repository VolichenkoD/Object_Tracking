# YOLOv4 обнаружение объектов и показ пройденного пути с помощью OpenCV

Модель создает ограничивающие рамки для каждого объекта на видео. Далее покадрово рисует и сохраняет точки, которые показывают пройденный путь.

Для начала нужно скачать yolov4.weights и добавить файл в папку dnn_model. этот файл много весит, поэтому я не смог закинуть его сюда.

Репозиторий включает в себя:
* [Папку](dnn_model) с настройками в которую нужно добавить yolov4.weights, в ней так же находиться [текстовый файл с классами](classes.txt) и [настройка  модели](yolov4.cfg)
* [Папку](input) в которой лежит видео
* Скрипт для обнаружения [object_detection.py](object_detection.py)
* Скрипт для деления видео на фреймы, обнаружения в них обьектов с точкой, которая указывает где был обьеьект в данное время [object_tracking.py](object_tracking.py)
* [Папку](outputs) в в которой лежит видео с результатом работы
