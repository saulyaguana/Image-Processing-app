## Top Apps:

This is a list of the top applications using OpenCV and some pre-trained models from the `dnn` module.

1. Motion detection.
2. Motion detection with contours.
3. Color Segmentation.
4. RGB detection.
5. Canny edge detection.
6. RGB detection under color segmentation
7. Canny Edge Detection with Bilateral Filter
8. Detecting Faces with pre trained models from Caffe
9. Landmark face detection with pre trainied models from Caffe
10. Object detection with SSD mobilenet architecture with COCO dataset
11. HCI application moving just your head (designed for offline web games)
12. Control the mouse with your index finger


The way this project is build is using an _object composition_ structure where the __main.py__ file is the interface to interact with this program.

The **images_ops.py** file is where all the logic is placed, all operations and validations.

The __calculator.py__ file is the bridge between __main.py__ and **images_ops.py**.