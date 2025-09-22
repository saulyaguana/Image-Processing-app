from images_ops import *

class Calculator:
    def __init__(self):
        self.accounts_dict = {}
        self.next_account_id = 0
    
    def open_account(self, name, password):
        account = ImageOps(name, password)
        current_account_id = self.next_account_id
        self.accounts_dict[current_account_id] = account
        self.next_account_id += 1
        return current_account_id
    
    def create_account(self):
        print("*** create account ***")
        name = input("Enter your name: ")
        password = input("Enter the password for this account: ")
        
        number_id = self.open_account(name, password)
        
        print(f"Your number account is: {number_id}")
        print()
        
    def valid_id(self):
        number_id = input("Enter your account number: ")
        try:
            number_id = int(number_id)
        except ValueError:
            raise OpError("Please enter a valid account number.")
        if number_id not in self.accounts_dict:
            raise OpError("Account number not found.")
        return number_id
        
    def split_image(self):
        print("*** Split Image Channels")
        account_id = self.valid_id()
        account = self.accounts_dict[account_id]
        
        path = input("Enter the path of the image: ")
        account.split_image(path)
        
    def contrast_image(self):
        print("*** Contrast Image ***")
        account_id = self.valid_id()
        account = self.accounts_dict[account_id]
        
        path = input("Enter the path of the image: ")
        contrast = input("Enter the contrast value: ")
        
        account.contrast_image(path, contrast)
        
    def binary_image(self):
        print("*** Binary Image ***")
        
        number_id = self.valid_id()
        account = self.accounts_dict[number_id]
        
        path = input("Enter the path of the image: ")
        threshold = input("Enter the threshold value: ")
        max_value = input("Enter the max value: ")
        
        account.binary_image(path, threshold, max_value)
        
    def color_histogram(self):
        print("*** Color Histogram ***")
        
        number_id = self.valid_id()
        account = self.accounts_dict[number_id]
        path = input("Enter the path of the image: ")
        
        account.color_histogram(path)
        
    def motion_detection(self):
        print("*** Motion Detection ***")
        number_id = self.valid_id()
        account = self.accounts_dict[number_id]
        
        history = int(input("Write the history value: "))
        path = int(input("Write the device you want to connect: "))
        
        account.motion_detection(history, path)
        
    def rgb_video(self):
        print("*** RGB Video ***")
        number_id = self.valid_id()
        
        account = self.accounts_dict[number_id]
        
        path = int(input("Write the device you want to connect: "))
        
        account.rgb_video(path)
        
    def motion_video_contours(self):
        print("*** Motion Video With Contours ***")
        number_id = self.valid_id()
        account = self.accounts_dict[number_id]
        
        history = int(input("The history: "))
        path = int(input("The device you want use: "))
        
        account.motion_video_contours(history, path)
        
    def segmented_frames(self):
        print("*** Segmented Video ***")
        number_id = self.valid_id()
        account = self.accounts_dict[number_id]
        
        path = int(input("The device you want use: "))
        
        account.segmented_frames(path)
        
    def canny_edge(self):
        print("*** Canny Edge Detection ***")
        number_id = self.valid_id()
        account = self.accounts_dict[number_id]
        
        path = int(input("The device you want use: "))
        threshold1 = int(input("Enter the first threshold: "))
        threshold2 = int(input("Enter the second threshold: "))
        account.canny_edge(path, threshold1, threshold2)
        
    def rgb_segmented(self):
        print("*** RGB Segmented Video ***")
        number_id = self.valid_id()
        account = self.accounts_dict[number_id]
        
        path = int(input("The device you want use: "))
        account.rgb_segmented(path)
        
    def bilateral_canny(self):
        print("*** Bilateral Canny Edge Detection ***")
        number_id = self.valid_id()
        account = self.accounts_dict[number_id]
        
        path = int(input("The device you want use: "))
        account.bilateral_canny(path)
        
    def binary_global_threshold(self):
        print("*** Binary Global Threshold ***")
        
        path = int(input("Enter the path of the device: "))
        
        obj = self.accounts_dict[0]
        obj.binary_global_threshold(path)