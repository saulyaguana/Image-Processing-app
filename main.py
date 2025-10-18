from calculator import *

calculator = Calculator()



while True:
    print("Hello to the OpenCV calculator!")
    print()

    print("""
      
To create an account, press a

To split an image in RGB, press s

To contrast an image, press c

To binary an image, press b

To histogram a color image, press h

To capture motion, press m

To watch the RGB of your video, press v

To capture motion with contours, press n

To segment the video, press seg

To detect edges (with Canny), press canny

To detect RGB in a segmented video, press seg_rgb

To detect edges with bilateral Canny, press bi_canny

To binary a frame with global thresholding, press thresh_frame

To detect faces, type detect_faces

To blur faces, type blur_faces

To landmark faces, press land_faces

To exit, type break""")

    print()
    
    # action = input("Enter your choice: ").lower().replace(" ", "")[0]
    action = input("Enter your choice: ")
    
    try:
        if action == "a":
            calculator.create_account()
        elif action == "s":
            calculator.split_image()
        elif action == "c":
            calculator.contrast_image()
        elif action == "b":
            calculator.binary_image()
        elif action == "h":
            calculator.color_histogram()    
        elif action == "m":
            calculator.motion_detection()
        elif action == "v":
            calculator.rgb_video()
        elif action == "n":
            calculator.motion_video_contours()
        elif action == "seg":
            calculator.segmented_frames()
        elif action == "canny":
            calculator.canny_edge()
        elif action == "seg_rgb":
            calculator.rgb_segmented()
        elif action == "bi_canny":
            calculator.bilateral_canny()
        elif action == "thresh_frame":
            calculator.binary_global_threshold()
        elif action == "detect_faces":
            calculator.detect_faces()
        elif action == "blur_faces":
            calculator.blur_faces()
        elif action == "land_faces":
            calculator.landmark_faces()
        elif action == "break":
            break
        else:
            print("Invalid choice. Please try again.")
    except OpError as e:
        print(e)
        
print("Done")