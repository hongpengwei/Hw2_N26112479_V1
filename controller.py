from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow
import tkinter as tk
from tkinter import filedialog 
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import *
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.decomposition import PCA
import glob


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
		# in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
         # TODO
         self.ui.pushButton.clicked.connect(self.load_video)
         self.ui.pushButton_2.clicked.connect(self.load_image)
         self.ui.pushButton_3.clicked.connect(self.load_folder)
         self.ui.pushButton_4.clicked.connect(self.background_subtraction)
         self.ui.pushButton_5.clicked.connect(self.preprocessing)
         self.ui.pushButton_6.clicked.connect(self.video_tracking)
         self.ui.pushButton_7.clicked.connect(self.perspective_transform)
         self.ui.pushButton_8.clicked.connect(self.image_reconstruction)
         self.ui.pushButton_9.clicked.connect(self.compute_the_error)
    
    def load_video(self):
        root = tk.Tk()
        root.withdraw()
        self.video = filedialog.askopenfilename()
        print(self.video)
        

    def load_image(self):
        root = tk.Tk()
        root.withdraw()
        self.image = filedialog.askopenfilename()
        print(self.image)

    def load_folder(self):
        root = tk.Tk()
        root.withdraw()
        self.folder = filedialog.askdirectory()
        print(self.folder)
    def background_subtraction(self):
        cap = cv2.VideoCapture(self.video)
        framecount=0
        frames=[]
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        ###################################################算mean
        while(cap.isOpened()):
            ret, frame = cap.read()
            if(framecount < 25):
            # Capture frame-by-frame         
                if ret == True:
                    framecount += 1
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray)
            if(framecount==25):        
                mean = np.mean(frames, axis= 0)
                standard = np.std(frames, axis=0)
                standard[standard < 5] = 5      
            if(framecount>=25):
                if ret == True:
                    framecount += 1
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mask = np.zeros_like(gray)
                    mask[np.abs(gray - mean) > standard*5] = 255
                    foreground = cv2.bitwise_and(frame, frame, mask= mask)
                    mask_out = np.zeros_like(frame)
                    mask_out[:,:,0] = mask
                    mask_out[:,:,1] = mask
                    mask_out[:,:,2] = mask
                    #ret, black = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
                    out = cv2.hconcat([frame, mask_out, foreground])
                    cv2.imshow('Result' , out)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
        
        cap.release()
        cv2.destroyAllWindows()

    def preprocessing(self):
        video = VideoFileClip(self.video)
        video.save_frame("frame.jpg", t = 0)
        img = cv2.imread('frame.jpg')

        # create the small border around the image, just bottom
        img=cv2.copyMakeBorder(img, top=0, bottom=1, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] ) 

        # create the params and deactivate the 3 filters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 60

        params.filterByColor = True    
        params.blobColor = 0

        params.filterByCircularity = True
        params.minCircularity = 0.8

        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        params.maxInertiaRatio = 1

        params.filterByConvexity = True
        params.minConvexity = 0.95

        # detect the blobs
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)

        # display them
        for i in range(len(cv2.KeyPoint_convert(keypoints))):
            img_with_keypoints = cv2.rectangle(img, (int(cv2.KeyPoint_convert(keypoints)[i][0]) - 6, int(cv2.KeyPoint_convert(keypoints)[i][1]) - 6), (int(cv2.KeyPoint_convert(keypoints)[i][0]) + 6, int(cv2.KeyPoint_convert(keypoints)[i][1]) + 6), (0, 0, 255), 1, cv2.LINE_AA)
            img_with_keypoints = cv2.line(img, (int(cv2.KeyPoint_convert(keypoints)[i][0]) - 6, int(cv2.KeyPoint_convert(keypoints)[i][1])), (int(cv2.KeyPoint_convert(keypoints)[i][0]) + 6, int(cv2.KeyPoint_convert(keypoints)[i][1])), (0, 0, 255), 1)
            img_with_keypoints = cv2.line(img, (int(cv2.KeyPoint_convert(keypoints)[i][0]), int(cv2.KeyPoint_convert(keypoints)[i][1]) - 6), (int(cv2.KeyPoint_convert(keypoints)[i][0]), int(cv2.KeyPoint_convert(keypoints)[i][1] + 6)), (0, 0, 255), 1)
        cv2.imshow("circle_detect", img_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def video_tracking(self):
        video = VideoFileClip(self.video)
        video.save_frame("frame.jpg", t = 0)
        img = cv2.imread('frame.jpg')

        # create the small border around the image, just bottom
        img=cv2.copyMakeBorder(img, top=0, bottom=1, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] ) 

        # create the params and deactivate the 3 filters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 60

        params.filterByColor = True    
        params.blobColor = 0

        params.filterByCircularity = True
        params.minCircularity = 0.8

        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        params.maxInertiaRatio = 1

        params.filterByConvexity = True
        params.minConvexity = 0.95

        # detect the blobs
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)
        KeyPoint_convert = []
        for i in range(len(keypoints)):
            KeyPoint_convert.append([[(float(int(cv2.KeyPoint_convert(keypoints)[i][0]))), (float(int(cv2.KeyPoint_convert(keypoints)[i][1])))]])
        # print(KeyPoint_convert)

        # Read the video 
        cap = cv2.VideoCapture(self.video)

        # Parameters for Lucas Kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Create random colors
        color = (0, 255, 255)

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        p0 = np.array(KeyPoint_convert)
        p0 = p0.astype(np.float32)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while True:
            # Read new frame
            ret, frame = cap.read()
            if ret:
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate Optical Flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None, **lk_params
                )
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)

                # Display the demo
                img = cv2.add(frame, mask)
                cv2.imshow("frame", img)
                k = cv2.waitKey(10) & 0xFF
                if k == 27:
                    cv2.imshow('frame', img)
                    cv2.waitKey()
                    break

                # Update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            else:
                break
    def perspective_transform(self):
        img = cv2.imread(self.image)
        height = img.shape[0]
        width = img.shape[1]
        cap = cv2.VideoCapture(self.video)    
        # Loop until the end of the video
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            original_frame = np.copy(frame)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            arucoDict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
            arucoParam=cv2.aruco.DetectorParameters_create()
            Corners,ids,_=cv2.aruco.detectMarkers(frame,arucoDict,parameters=arucoParam) 
            id1 = np.squeeze(np.where(ids == 1))
            id2 = np.squeeze(np.where(ids == 2))
            id3 = np.squeeze(np.where(ids == 3))
            id4 = np.squeeze(np.where(ids == 4))   
            # 定义对应的点
            if len(ids)==4:

                # points1 = np.float32([i]).reshape(-1,1,2)
                # Get the top-left corner of marker1
                pt1 = np.squeeze(Corners[id1[0]])[0]
                # Get the top-right corner of marker2
                pt2 = np.squeeze(Corners[id2[0]])[1]
                # Get the bottom-right corner of marker3
                pt3 = np.squeeze(Corners[id3[0]])[2]
                # Get the bottom-left corner of marker4
                pt4 = np.squeeze(Corners[id4[0]])[3]

                points1 = [[pt1[0], pt1[1]]]
                points1 = points1 + [[pt2[0], pt2[1]]]
                points1 = points1 + [[pt3[0], pt3[1]]]
                points1 = points1 + [[pt4[0], pt4[1]]]

                points2 = [[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]
                
            # 计算得到转换矩阵
                M ,status= cv2.findHomography(np.float32(points2), np.float32(points1),cv2.RANSAC,5.0)
                h, w = frame.shape[:2]
                # Apply perspective transformation to an image
                result = cv2.warpPerspective(img, M, (w,h))

            
            
            # #Display the resulting frame
                mask2 = np.zeros(frame.shape, dtype=np.uint8)
                roi_corners2 = np.int32(points1)
                channel_count2 = frame.shape[2]  
                ignore_mask_color2 = (255,)*channel_count2
                cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)
                mask2 = ~mask2
                masked_image2 = frame&mask2

                #Using Bitwise or to merge the two images
                result = result|masked_image2
                
                # Display result
                m = cv2.hconcat((original_frame, result))
                cv2.namedWindow('frame', 0)
                cv2.resizeWindow('frame', 1920, 480)
                cv2.imshow('frame', m)
                key = cv2.waitKey(1)
                if key == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()
    def image_reconstruction(self):
            img_org = []
            img_pca = []
            folder=self.folder+"/*.jpg"
            images = glob.glob(folder)
            for fname in images:
                #print(fname)
                img = cv2.imread(fname)
                blue, green, red = cv2.split(img)
                pca = PCA()
                img_compressed = (np.dstack((red, green, blue))).astype(np.uint8)
                img_org.append(img_compressed)

                pca = PCA(65)
                red_transformed = pca.fit_transform(red)
                red_inverted = pca.inverse_transform(red_transformed)

                green_transformed = pca.fit_transform(green)
                green_inverted = pca.inverse_transform(green_transformed)

                blue_transformed = pca.fit_transform(blue)
                blue_inverted = pca.inverse_transform(blue_transformed)

                img_compressed = (np.dstack((red_inverted, green_inverted, blue_inverted))).astype(np.uint8)
                img_pca.append(img_compressed)

            plt.figure(figsize=(30, 30))
            for i in range(60):
                plt.subplot(4, 15, i + 1)
                if int(i / 15) == 0:
                    plt.imshow(img_org[i], cmap=plt.cm.binary)
                elif int(i / 15) == 1:
                    plt.imshow(img_pca[i - 15], cmap=plt.cm.binary)
                elif int(i / 15) == 2:
                    plt.imshow(img_org[i - 15], cmap=plt.cm.binary)
                else:
                    plt.imshow(img_pca[i - 30], cmap=plt.cm.binary)
                plt.xticks([])
                plt.yticks([])
            plt.show()
    def compute_the_error(self):
        result = []
        folder=self.folder+"/*.jpg"
        images = glob.glob(folder)
        for fname in images:
            img = cv2.imread(fname)
            blue, green, red = cv2.split(img)

            pca = PCA(65)
            red_transformed = pca.fit_transform(red)
            red_inverted = pca.inverse_transform(red_transformed)

            green_transformed = pca.fit_transform(green)
            green_inverted = pca.inverse_transform(green_transformed)

            blue_transformed = pca.fit_transform(blue)
            blue_inverted = pca.inverse_transform(blue_transformed)

            sum = 0
            for i in range(100):
                for j in range(100):
                    if red_inverted[i, j] > 255:
                        red_inverted[i, j] = 255
                    elif red_inverted[i, j] < 0:
                        red_inverted[i, j] = 0
                    sum += abs(red[i, j] - red_inverted[i, j])
            for i in range(100):
                for j in range(100):
                    if green_inverted[i, j] > 255:
                        green_inverted[i, j] = 255
                    elif green_inverted[i, j] < 0:
                        green_inverted[i, j] = 0
                    sum += abs(green[i, j] - green_inverted[i, j])
            for i in range(100):
                for j in range(100):
                    if blue_inverted[i, j] > 255:
                        blue_inverted[i, j] = 255
                    elif blue_inverted[i, j] < 0:
                        blue_inverted[i, j] = 0
                    sum += abs(blue[i, j] - blue_inverted[i, j])
            result.append(sum)
        print(result)
        print ("max :",max(result),"min :",min(result))


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())