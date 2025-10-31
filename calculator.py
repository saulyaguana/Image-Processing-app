from images_ops import *

class Calculator:
    def __init__(self):
        self.img_object = ImageOps()
        
    def split_image(self):
        print("*** Split Image Channels")
        account_id = self.valid_id()
        account = self.img_object
        
        path = input("Enter the path of the image: ")
        account.split_image(path)
        
    def contrast_image(self):
        print("*** Contrast Image ***")
        account = self.img_object
        
        path = input("Enter the path of the image: ")
        contrast = input("Enter the contrast value: ")
        
        account.contrast_image(path, contrast)
        
    def binary_image(self):
        print("*** Binary Image ***")
        account = self.img_object
        
        path = input("Enter the path of the image: ")
        threshold = input("Enter the threshold value: ")
        max_value = input("Enter the max value: ")
        
        account.binary_image(path, threshold, max_value)
        
    def color_histogram(self):
        print("*** Color Histogram ***")
        account = self.img_object
        path = input("Enter the path of the image: ")
        
        account.color_histogram(path)
        
    def motion_detection(self):
        print("*** Motion Detection ***")
        account = self.img_object
        
        history = int(input("Write the history value: "))
        path = int(input("Write the device you want to connect: "))
        
        account.motion_detection(history, path)
        
    def rgb_video(self):
        print("*** RGB Video ***")
        account = self.img_object
        
        path = int(input("Write the device you want to connect: "))
        
        account.rgb_video(path)
        
    def motion_video_contours(self):
        print("*** Motion Video With Contours ***")
        account = self.img_object
        
        history = int(input("The history: "))
        path = int(input("The device you want use: "))
        
        account.motion_video_contours(history, path)
        
    def segmented_frames(self):
        print("*** Segmented Video ***")
        account = self.img_object
        
        path = int(input("The device you want use: "))
        
        account.segmented_frames(path)
        
    def canny_edge(self):
        print("*** Canny Edge Detection ***")
        account = self.img_object
        
        path = int(input("The device you want use: "))
        threshold1 = int(input("Enter the first threshold: "))
        threshold2 = int(input("Enter the second threshold: "))
        account.canny_edge(path, threshold1, threshold2)
        
    def rgb_segmented(self):
        print("*** RGB Segmented Video ***")
        account = self.img_object
        
        path = int(input("The device you want use: "))
        account.rgb_segmented(path)
        
    def bilateral_canny(self):
        print("*** Bilateral Canny Edge Detection ***")
        account = self.img_object
        
        path = int(input("The device you want use: "))
        account.bilateral_canny(path)
        
    def binary_global_threshold(self):
        print("*** Binary Global Threshold ***")
        
        path = int(input("Enter the path of the device: "))
        obj = self.img_object
        
        obj.binary_global_threshold(path)
        
    def detect_faces(self):
        print("*** Detect Faces ***")
        
        path = int(input("Enter the path of the device: "))
        
        obj = self.img_object
        obj.detect_faces(path)
        
    def blur_faces(self):
        print("*** Blur Faces ***")
        
        # There will be more versions for blurring faces
        path = int(input("Enter the path of the device: "))
        
        obj = self.img_object
        obj.blur_faces(path)
        
    def landmark_faces(self):
        print("*** Landmark Faces ***")
        
        path = int(input("Enter the path of the device: "))
        
        obj = self.img_object
        obj.landmark_faces(path)
        
    def detect_coco_dataset(self):
        print("*** Detect COCO Dataset Objects ***")
        
        path = int(input("Enter the path of the device: "))
        
        obj = self.img_object
        obj.detect_coco_dataset(path)
        
    def web_game(self):
        print("*** Web Game Control with Face Movements ***")
        
        path = int(input("Enter the path of the device: "))
        
        obj = self.img_object
        obj.web_game(path)