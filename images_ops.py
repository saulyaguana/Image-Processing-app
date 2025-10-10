import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["image.cmap"] = "gray"

class OpError(Exception):
    pass

class ImageOps:
    def __init__(self):
        pass
        
    def validate_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise OpError("Please check the correct path again.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def validate_video(self, path):
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            raise OpError("Please check the correct path again.")
        return video
    
    def validate_pre_trained_model(self, path_proto, path_model):
        if not os.path.exists(path_proto):
            raise OpError("Prototxt file not found, check the path again")
        if not os.path.exist(path_model):
            raise OpError("Model not found, check the path again")
        
        try:
            net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
        except cv2.error as e:
            raise OpError(f"Failed to load model. OpenCV error - {e}")
        except Exception as e:
            raise OpError("Unexpected error while loading model")
        
        if net.empty():
            raise OpError("Loaded model, but it is empty")
        
        return net
        
    
    # def validate_password(self, password):
    #     if self.password != password:
    #         raise OpError("Please enter the correct password.")
            
        
    def split_image(self, path):
        image = self.validate_image(path)
        r, g, b = cv2.split(image)
        merged = cv2.merge((r, g, b))
        
        fig, ax = plt.subplots(4, figsize=(10, 10))
        ax[0].imshow(r)
        ax[0].set_title('Red Channel')
        
        ax[1].imshow(g)
        ax[1].set_title('Green Channel')
        
        ax[2].imshow(b)
        ax[2].set_title('Blue Channel')
        
        ax[3].imshow(merged)
        ax[3].set_title('Merged Image')
        
        plt.show()
        
    def contrast_image(self, path, contrast=1.0):
        try:
            contrast = float(contrast)
        except ValueError:
            raise OpError("Please enter a valid contrast value.")
        image = self.validate_image(path)
        matrix = np.ones(image.shape, dtype="float64")
        
        image_contrast = np.uint8(np.clip(cv2.multiply(np.float64(image), matrix, scale=contrast), 0, 255))
        
        fig, ax = plt.subplots(2, figsize=(10, 10))
        
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original Image")
        
        ax[1].imshow(cv2.cvtColor(image_contrast, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Contrast Image")
        
        plt.show()
        
    def binary_image(self, path, threshold, max_value=255):
        
        try:
            threshold = int(threshold)
            max_value = int(max_value)
        except ValueError:
            raise OpError("Please enter a valid threshold value.")
        
        if max_value > 255 or max_value < 0:
            raise OpError("Please enter a valid max value.")
        
        image = self.validate_image(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, binary_image = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
        
        fig, ax = plt.subplots(3, figsize=(10, 10))
        ax[0].imshow(image)
        ax[0].set_title("GrayScale Image")
        
        ax[1].imshow(binary_image)
        ax[1].set_title("Binary Image")
        
        ax[2].imshow(cv2.bitwise_and(cv2.imread(path, cv2.IMREAD_COLOR), cv2.imread(path, cv2.IMREAD_COLOR), mask=binary_image)[:, :, ::-1])
        
        plt.show()
        
    def binary_global_threshold(self, path, max_value=255):
        video  = self.validate_video(path)
        win_name = "Binary Global Threshold"
        cv2.namedWindow(win_name)
        
        while True:
            has_frame, frame = video.read()
            
            if not has_frame:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary_frame = cv2.threshold(gray_frame, 120, max_value, cv2.THRESH_BINARY)
            
            cv2.imshow(win_name, binary_frame)
            
            key = cv2.waitKey(1)
            
            if key == ord("q") or key == ord("Q") or key == 27:
                break
            
        video.release()
        cv2.destroyAllWindows()
        
        
    def color_histogram(self, path):
        image = self.validate_image(path)
        
        red = cv2.calcHist([image], [0], None, [256], [0, 255])  # R
        green = cv2.calcHist([image], [1], None, [256], [0, 255])  # G
        blue = cv2.calcHist([image], [2], None, [256], [0, 255])  # B
        
        fig, ax = plt.subplots(2, figsize=(10, 10))
        
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].set_axis_off()
        
        ax[1].plot(red, color="red")
        ax[1].plot(green, color="green")
        ax[1].plot(blue, color="blue")
        
        plt.show()
    
    
    def motion_detection(self, history, path=0, kernel_size=(5, 5)):
        knn = cv2.createBackgroundSubtractorKNN(history=history)
        video = self.validate_video(path)
        color_frame = "Color Frame"
        binary_frame = "Binary Frame"
        cv2.namedWindow(color_frame)
        cv2.namedWindow(binary_frame)
        
        while True:
            ok, frame = video.read()
            
            if not ok:
                break
            
            mask = knn.apply(frame)
            mask_eorded = cv2.erode(mask, np.ones(kernel_size, np.uint8))
            non_zero = cv2.findNonZero(mask_eorded)
            x, y, xw, yh = cv2.boundingRect(non_zero)
            
            if mask_eorded is not None:
                cv2.rectangle(frame, (x, y), (x + xw, y + yh), (0, 0, 255), 4)
                
            cv2.imshow(color_frame, frame)
            cv2.imshow(binary_frame, mask_eorded)
            
            key = cv2.waitKey(1)
            
            if ord("Q") == key or ord("q") == 27 or key == 27:
                break
            
        video.release
        cv2.destroyAllWindows()
        
    def motion_video_contours(self, history, path=0, kernel_size=(5, 5)):
        knn = cv2.createBackgroundSubtractorKNN(history=history)
        
        video = self.validate_video(path)
        
        
        frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_contorus = 3
        min_contour_are_thresh = 0.01
        frame_area = frame_w * frame_h
        
        color_window = "Color Window"
        binary_window = "Binary Window"
        
        cv2.namedWindow(color_window)
        cv2.namedWindow(binary_window)
        
        while True:
            ok, frame = video.read()
            
            if not ok:
                break
            
            foreground_mask = knn.apply(frame)
            eroded_foreground = cv2.erode(foreground_mask, np.ones(kernel_size, np.uint8))
            eroded_color = cv2.cvtColor(eroded_foreground, cv2.COLOR_GRAY2BGR)
            
            contours, hiererchy = cv2.findContours(eroded_foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
                contour_area_max = cv2.contourArea(contours_sorted[0])
                contour_frac = contour_area_max / frame_area
                cv2.drawContours(eroded_color, contours, -1, (0, 255, 0), 2)
                if contour_frac > min_contour_are_thresh:
                    for idx in range(min(max_contorus, len(contours_sorted))):
                        x, y, xw, yh = cv2.boundingRect(contours_sorted[idx])
                        if idx == 0:
                            x1 = x
                            y1 = y
                            x2 = x + xw
                            y2 = y + yh
                        else:
                            x1 = min(x1, x)
                            y1 = min(y1, y)
                            x2 = max(x1, x + xw)
                            y2 = max(y2, y + yh)
                            
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    #os.system("ls -la")
                    
                cv2.imshow(color_window, frame)
                cv2.imshow(binary_window, eroded_color)
            
                key = cv2.waitKey(1)
                
                if key == ord("Q") or key == ord("q") or key == 27:
                    break
                
        video.release()
        cv2.destroyAllWindows()         

    def rgb_video(self, path):
        video = self.validate_video(path)
        win_name = "RGB Video"
        cv2.namedWindow(win_name)
        
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("RGB Histogram")
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 5000])
        
        # Empty line to start with
        r_line, = ax.plot(np.zeros(256), color="red", label="Red")
        g_line, = ax.plot(np.zeros(256), color="green", label="Green")
        b_line, = ax.plot(np.zeros(256),color="blue", label="Blue")
        ax.legend()
        
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            r = cv2.calcHist([frame_rgb], [0], None, [256], [0, 255])
            g = cv2.calcHist([frame_rgb], [1], None, [256], [0, 255])
            b = cv2.calcHist([frame_rgb], [2], None, [256], [0, 255])
        
            # Update the line data
            r_line.set_ydata(r.flatten())
            g_line.set_ydata(g.flatten())
            b_line.set_ydata(b.flatten())
            
            fig.canvas.draw_idle()
            plt.pause(0.001)
            
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        video.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close()
        
    def rgb_segmented(self, path, lb=[15, 80, 90], up=[35, 255, 255]):
        
        lb = np.array(lb, np.uint8)
        up = np.array(up, np.uint8)
        
        video = self.validate_video(path)
        
        original_window = "Original Window"
        segmented_window = "Segmented Window"
        
        cv2.namedWindow(original_window)
        cv2.namedWindow(segmented_window)
        
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("RGB Histogram (segmented only)")
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 5000])
        
        # Lines to start with
        r_line, = ax.plot(np.zeros(256), color="red", label="Red")
        g_line, = ax.plot(np.zeros(256), color="green", label="Green")
        b_line, = ax.plot(np.zeros(256), color="blue", label="Blue")
        ax.legend()
        
        while True:
            ok, frame = video.read()
            if not ok:
                break
            
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, lb, up)
            
            segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            b = cv2.calcHist([segmented_frame], [0], mask, [256], [0, 255])
            g = cv2.calcHist([segmented_frame], [1], mask, [256], [0, 255])
            r = cv2.calcHist([segmented_frame], [2], mask, [256], [0, 255])
            
            # Update line data
            b_line.set_ydata(b.flatten())
            g_line.set_ydata(g.flatten())
            r_line.set_ydata(r.flatten())
            
            fig.canvas.draw_idle()
            plt.pause(0.001)
            
            cv2.imshow(original_window, frame)
            cv2.imshow(segmented_window, segmented_frame)
            
            key = cv2.waitKey(1)
            
            if key == ord("q") or key == ord("Q") or key == 27:
                break
        video.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close()
            
        
        
    # Greens_lower_bound = [45, 30, 150]
    # Greens_upper_bound = [90, 255, 255]
    def segmented_frames(self, path, lower_bound=[45, 30, 150], upper_bound=[90, 255, 255]):
        video = self.validate_video(path)
        win_name = "Segmented Window"
        win_name2 = "Original Window"
        cv2.namedWindow(win_name)
        cv2.namedWindow(win_name2)
        
        lb = np.array(lower_bound, np.uint8)
        up = np.array(upper_bound, np.uint8)
        
        while True:
            ok, frame = video.read()
            
            if not ok:
                break
            
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            hsv_detected = cv2.inRange(frame_hsv, lb, up)
            
            segmented = cv2.bitwise_and(frame, frame, mask=hsv_detected)
            
            cv2.imshow(win_name, segmented)
            cv2.imshow(win_name2, frame)
            
            key = cv2.waitKey(1)
            
            if key == ord("q") or key == ord("Q") or key == 27:
                break
            
        video.release()
        cv2.destroyAllWindows()
            
    def canny_edge(self, path, threshold1, threshold2):
        try:
            threshold1= int(threshold1)
            threshold2 = int(threshold2)
        except ValueError:
            raise OpError("Please enter a valid threshold value.")
        video = self.validate_video(path)
        win_name = "Canny Edge Detection"
        cv2.namedWindow(win_name)
        
        while True:
            ok, frame = video.read()
            
            if not ok:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)  
            edges = cv2.Canny(blurred_frame, threshold1, threshold2)
            
            cv2.imshow(win_name, edges)
            key = cv2.waitKey(1)
            
            if key == ord("q") or key == ord("Q") or key == 27:
                break
        video.release()
        cv2.destroyAllWindows()
        
    def bilateral_canny(self, path, d=9, sigma_color=75, sigma_space=75, threshold1=100, threshold2=200):
        try:
            d = int(d)
            sigma_color = int(sigma_color)
            sigma_space = int(sigma_space)
            threshold1 = int(threshold1)
            threshold2 = int(threshold2)
        except ValueError:
            raise OpError("Please enter valid values.")
        video = self.validate_video(path)
        
        bilateral_canny_window = "Bilateral Canny Edge Detection"
        original = "Original Video"
        
        cv2.namedWindow(bilateral_canny_window)
        cv2.namedWindow(original)
        
        while True:
            ok, frame = video.read()
            
            if not ok:
                break
            
            bilateral_frame = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
            gray_bilateral_frame = cv2.cvtColor(bilateral_frame, cv2.COLOR_BGR2GRAY)
            canny_bilateral_frame = cv2.Canny(gray_bilateral_frame, threshold1, threshold2)
            
            cv2.imshow(bilateral_canny_window, canny_bilateral_frame)
            cv2.imshow(original, bilateral_frame)
            
            key = cv2.waitKey(1)
            
            if key == ord("q") or key == ord("Q") or key == 27:
                break
            
        video.release()
        cv2.destroyAllWindows()
        
    def detect_faces(
        self,
        path_proto,
        path_model,
        device,
        scale=1.0,
        mean=[104, 117, 123],
        width=300,
        height=300,
    ):
        net = self.validate_pre_trained_model(path_proto, path_model)
        
        win_name = "Face detector"
        cv2.namedWindow(win_name)
        
        video = self.validate_video(device)
        
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            has_frame, frame = video.read()
            
            if not has_frame:
                break
            
            # Convert the image to blob
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=scale,
                size=(width, height),
                mean=mean,
                swapRB=False,
                crop=False,
            )
            
            # Pass the blob to the DNN model
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = "Confidence: %.4f" % confidence
                    label_size , base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_size[1]),
                        (x1 + label_size[0], y1 + base_line),
                        (255, 255, 255),
                        cv2.FILLED,
                    )
                    
                    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
                    
            cv2.imshow(win_name, frame)
            
            key = cv2.waitKey(1)
            
            if key == 27:
                break
        video.release()
        cv2.destroyAllWindows()