import cv2
import numpy as np
import datetime
from matplotlib import pyplot as plt

##  1.READING AND WRITING IMAGES

# img=cv2.imread("messi5.jpg",0)

# print(img)

# cv2.imshow('image',img)
# cv2.waitKey()

# cv2.destroyWindow()

# cv2.imwrite('messi5_copy.png',img)


## 2.READING AND WRITING VIDEOS FROM CAMERA

# cap=cv2.VideoCapture(0) 
# # we can use 1 for second camera, 2 for 3rd and so on.
# #cv2.VideoCapture('<file_name>') to capture from a specific file 

# fourcc=cv2.VideoWriter_fourcc(*'XVID')
# out=cv2.VideoWriter('output.avi',fourcc,20.0, (640,480)) # to save capture video cv2.VideoWriter('<file_name>', <fourcc_code>,<frame_rate>,<size>)

# while(True): # we can also write while(cap.isOpened()) this will give true if file exist 
#     ret, frame=cap.read()

#     if ret==True:
#         out.write(frame) # to save video

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #TO CONVERT TO GRAY SCALE
#         cv2.imshow('frame', gray)
#         key=cv2.waitKey(1)
#         if key == 13:
#             break
#     else:
#         break

# cap.release() 
# cv2.destroyAllWindows()


## 3.DRAW GEOMETRIC SHAPES AND TEXT ON IMAGES

# img = cv2.imread('messi5.jpg',1)

# img = np.zeros([512, 512, 3], np.uint8)

# img = cv2.line(img, (0,0), (255, 255), (255, 0, 0), 3 ) #color is in bgr format

# img = cv2.rectangle(img, (0,0), (255,255), (255, 0, 0), 4) #type -1 in place of thickness to fill it.

# img = cv2.circle(img, (255,255), 40, (255, 0, 0), 6)

# font = cv2.FONT_HERSHEY_SIMPLEX
# img = cv2.putText(img , 'OpenCV', (257,255),font , 2, (255, 0, 0), 5)

# cv2.imshow('frame',img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


## 4. SETTING CAMERA PARAMETERS

# cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # cap.set(3, 1208)
# # cap.set(4, 720) # can also be written like this

# while(cap.isOpened()):
#     ret, frame=cap.read()

#     if ret==True:

#         cv2.imshow('frame', frame)
#         key=cv2.waitKey(1)
#         if key == 13:
#             break
#     else:
#         break

# cap.release() 
# cv2.destroyAllWindows


## 5. SHOW DATE AND TIME ON VIDEOS

# cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # cap.set(3, 1208)
# # cap.set(4, 720) # can also be written like this

# while(cap.isOpened()):
#     ret, frame=cap.read()

#     if ret==True:

#         font = cv2.FONT_HERSHEY_COMPLEX
#         datet = str(datetime.datetime.now())
#         # text = 'Width:' + str(cap.get(3)) + ' Height:' + str(cap.get(4))
#         frame=cv2.putText(frame, datet, (10,50), font, 1, (255, 0, 0))
#         cv2.imshow('frame', frame)
#         key=cv2.waitKey(40)
#         if key == 13:
#             break
#     else:
#         break

# cap.release() 
# cv2.destroyAllWindows


## 6. MOUSE EVENTS

# # events = [i for i in dir(cv2) if 'EVENT' in i]
# # print(events) #yo print all events

# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x,' ',y)
#         font = cv2.FONT_HERSHEY_COMPLEX
#         strXY= str(x) + ',' + str(y)
#         cv2.putText(img, strXY, (x,y), font, 0.5, (255,255,0), 2)
#         cv2.imshow('image', img)
#     if event== cv2.EVENT_RBUTTONDOWN:
#         blue=img[y, x, 0]
#         green=img[y, x, 1]
#         red=img[y, x, 2]
#         font = cv2.FONT_HERSHEY_COMPLEX
#         strBGR= str(blue) + ',' + str(green) + ',' + str(red)
#         cv2.putText(img, strBGR, (x,y), font, 0.5, (0,255,0), 1)
#         cv2.imshow('image', img)

# img = cv2.imread('messi5.jpg', 1)
# cv2.imshow('image', img)

# cv2.setMouseCallback('image', click_event)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


## 7. MORE MOUSE EVENTS

# def click_event(event, x, y, flags, param): #mark a point on the location of click and join with previous line
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(img, (x,y), 3, (0,0,255), -1)
#         points.append((x,y))
#         if len(points) >=2:
#             cv2.line(img, points[-1], points[-2], (255, 0, 0), 3)
#         cv2.imshow('image', img)
#     if event == cv2.EVENT_RBUTTONDOWN: #to show the color of point clicked on a different video.
#         blue=img[y, x, 0]
#         green=img[y, x, 1]
#         red=img[y, x, 2]
#         cv2.circle(img, (x,y), 3, (0, 0, 255), -1)

#         mycolorImage = np.zeros((512,512,3), np.uint8)
#         mycolorImage[:]=[blue, green, red]
#         cv2.imshow('colorWindow', mycolorImage)

# img = cv2.imread('messi5.jpg', 1)
# cv2.imshow('image', img)

# points = []

# cv2.setMouseCallback('image', click_event)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


## 8. CV.SPLIT, CV.MERGE, CV.RESIZE, CV.ADD, CV.ADDWEIGHT, ROI(region of interest)

# img=cv2.imread('messi5.jpg',1)
# img2 = cv2.imread('LinuxLogo.jpg')

# print(img.shape) # returns numer of rows, columns and channels
# print(img.size) # return number of pixels 
# print(img.dtype) # return dataType of image

# b,g,r = cv2.split(img) #use to split image into different channel
# img = cv2.merge((b,g,r)) #use to merge different channel into one

# ball = img[280:340, 330:390] #this is to take a part of some image and insert at any other location
# img[273:333, 100:160] = ball

# img = cv2.resize(img, (512,512))
# img2 = cv2.resize(img2, (512,512))
# # new_img = cv2.add(img, img2)

# new_img = cv2.addWeighted(img, 0.5, img2, 0.5, 0)

# cv2.imshow('image',new_img)
# cv2.waitKey()

# cv2.destroyAllWindows()


## 9. BITWISE OPERATION

# img1 = np.zeros((960, 1080, 3), np.uint8)
# img1 = cv2.rectangle(img1, (300,0), (700,300), (255,255,255), -1)
# img2 = cv2.imread('image_1.jpg')

# bitAnd = cv2.bitwise_and(img2, img1) #black is 0 and white is 1

# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('bitAnd', bitAnd)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


## 10A. TRACKBAR

# def nothing(x):
#     print(x)

# img = np.zeros((300,512,3), np.uint8)
# cv2.namedWindow('image')

# cv2.createTrackbar('B', 'image', 0, 255, nothing)
# cv2.createTrackbar('G', 'image', 0, 255, nothing)
# cv2.createTrackbar('R', 'image', 0, 255, nothing)

# switch = '0 : OFF\n 1 : ON'
# cv2.createTrackbar(switch, 'image', 0, 1, nothing)

# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k== 27:
#         break

#     b = cv2.getTrackbarPos('B', 'image')
#     g = cv2.getTrackbarPos('G', 'image')
#     r = cv2.getTrackbarPos('R', 'image')
#     s = cv2.getTrackbarPos(switch, 'image')

#     if s == 0:
#         img[:] = 0
#     else:
#         img[:] = [b,g,r]
# cv2.destroyAllWindows()


## 10B. TRACKBAR EXAPMLE 2

# def nothing(x):
#     print(x)

# cv2.namedWindow('image')

# cv2.createTrackbar('CP', 'image', 10, 400, nothing)

# switch = 'color/gray'
# cv2.createTrackbar(switch, 'image', 0, 1, nothing)

# while(1):
#     img = cv2.imread('messi5.jpg', 1)
#     pos = cv2.getTrackbarPos('CP', 'image')

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(img, str(pos), (50,150), font, 4, (0,0,255))

#     k = cv2.waitKey(1)

#     if k== 27:
#         break

#     s = cv2.getTrackbarPos(switch, 'image')

#     if s == 0:
#         pass
#     else:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     cv2.imshow('image', img)
    
# cv2.destroyAllWindows()


## 11A. OBJECT DETECTION(image)


# def nothing(x):
#     pass

# cv2.namedWindow('Tracking')
# cv2.createTrackbar('LH', 'Tracking', 0, 255, nothing)
# cv2.createTrackbar('LS', 'Tracking', 0, 255, nothing)
# cv2.createTrackbar('LV', 'Tracking', 0, 255, nothing)

# cv2.createTrackbar('UH', 'Tracking', 255, 255, nothing)
# cv2.createTrackbar('US', 'Tracking', 255, 255, nothing)
# cv2.createTrackbar('UV', 'Tracking', 255, 255, nothing)

# while(1):
#     frame = cv2.imread('smarties.png')

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #to convert to hsv

#     l_h = cv2.getTrackbarPos('LH', 'Tracking')
#     l_s = cv2.getTrackbarPos('LS', 'Tracking')
#     l_v = cv2.getTrackbarPos('LV', 'Tracking')

#     u_h = cv2.getTrackbarPos('UH', 'Tracking')
#     u_s = cv2.getTrackbarPos('US', 'Tracking')
#     u_v = cv2.getTrackbarPos('UV', 'Tracking')

#     l_b = np.array([l_h, l_s, l_v]) #lower bound of blue
#     u_b = np.array([u_h, u_s, u_v])

#     mask = cv2.inRange(hsv, l_b, u_b)

#     res = cv2.bitwise_and(frame, frame, mask=mask)

#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res', res)

#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:
#         break

# cv2.destroyAllWindows()


## 11B. OBJECT DETECTION(video)

# def nothing(x):
#     pass

# cap = cv2.VideoCapture(0)

# cv2.namedWindow('Tracking')
# cv2.createTrackbar('LH', 'Tracking', 0, 255, nothing)
# cv2.createTrackbar('LS', 'Tracking', 0, 255, nothing)
# cv2.createTrackbar('LV', 'Tracking', 0, 255, nothing)

# cv2.createTrackbar('UH', 'Tracking', 255, 255, nothing)
# cv2.createTrackbar('US', 'Tracking', 255, 255, nothing)
# cv2.createTrackbar('UV', 'Tracking', 255, 255, nothing)

# while(1):
#     _ , frame = cap.read() 

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #to convert to hsv

#     l_h = cv2.getTrackbarPos('LH', 'Tracking')
#     l_s = cv2.getTrackbarPos('LS', 'Tracking')
#     l_v = cv2.getTrackbarPos('LV', 'Tracking')

#     u_h = cv2.getTrackbarPos('UH', 'Tracking')
#     u_s = cv2.getTrackbarPos('US', 'Tracking')
#     u_v = cv2.getTrackbarPos('UV', 'Tracking')

#     l_b = np.array([l_h, l_s, l_v]) #lower bound of blue
#     u_b = np.array([u_h, u_s, u_v])

#     mask = cv2.inRange(hsv, l_b, u_b)

#     res = cv2.bitwise_and(frame, frame, mask=mask)

#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res', res)

#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


## 12. IMAGE THRESHOLDING

# img = cv2.imread('gradient.png',0)

# _ , th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)# white if greater than threshold value, and black if less.

# _ , th2 =cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)# inverse of binary

# _ , th3 = cv2.threshold(img, 50, 255, cv2.THRESH_TRUNC)# if value is in range then it will all be same as base value

# _ , th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)# if value is not in range then black

# cv2.imshow('Image', img)
# cv2.imshow('th1', th1)
# cv2.imshow('th2', th2)
# cv2.imshow('th3', th3)
# cv2.imshow('th4', th4)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

## 13. ADAPTIVE THRESHOLDING

# img = cv2.imread('sudoku.png',0)

# _, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# titles = ['Original image', 'binary', 'adaptive_mean', 'adaptive_gaussian']
# images = [img, th1, th2, th3]

# for i in range(4):
#     plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])

# plt.show()

## 14A.MATPLOTLIB EITH OPENCV

# img = cv2.imread('messi5.jpg', -1)

# cv2.imshow('image',img)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.imshow(img)
# plt.xticks([]),plt.yticks([])#to hide text on x and y axis
# plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()


## 14B.USING MATPLOTLIB TO DISPLAY MULTIPLE IMAGE IN ONE WINDOW

# img = cv2.imread('gradient.png',0)

# _ , th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)# white if greater than threshold value, and black if less.

# _ , th2 =cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)# inverse of binary

# _ , th3 = cv2.threshold(img, 50, 255, cv2.THRESH_TRUNC)# if value is in range then it will all be same as base value

# _ , th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)# if value is not in range then black

# titles = ['Original image', 'binary', 'binary_inv', 'trunc', 'tozero']
# images = [img, th1, th2, th2, th4]

# for i in range(5):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])

# plt.show()


## 15. MORPHOLOGICAL TRANFORMATION

# img = cv2.imread('smarties.png', 0)
# _, mask = cv2.threshold(img, 220, 225, cv2.THRESH_BINARY_INV)

# kernal = np.ones((5,5), np.uint8)

# dilation = cv2.dilate(mask, kernal, iterations=2) #to remove black dots in mask

# erosion = cv2.erode(mask, kernal, iterations=1)

# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal) #first eroded then dialated

# closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal) #first dialated and then eroded

# mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal) #difference between dilation and erosion

# th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernal) #difference between image and opening

# titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'mg', 'th']
# images = [img, mask, dilation, erosion, opening, closing, mg, th]

# for i in range(8):
#     plt.subplot(2,4, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])

# plt.show()


## 16.SMOOTHING OR BLURRING

# img = cv2.imread('opencv-logo.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# kernel = np.ones((5, 5), np.float32)/25

# dst = cv2.filter2D(img, -1, kernel) #HOMOGENEOUS #each output pixel is mean of its kernel neighbour

# blur = cv2.blur(img, (5,5)) #averaging

# gbl = cv2.GaussianBlur(img, (5,5), 0) #gaussian blur #better in removing high frequency noise # wieghted mean with centre having more weight

# median = cv2.medianBlur(img, 5) #used to remove salt and pepper noise

# bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75) #preserves edges #(<source>, <kernel_diameter>, <sigma_space>, <sigma_color>)

# titles = ['image', '2D convolution', 'blur', 'Gaussian blur', 'Median blur', 'bilateral filter']
# images = [img, dst, blur, gbl, median, bilateralFilter]

# for i in range(6):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])

# plt.show()


## 17. IMAGE GRADIENTS AND EDGE DETECTION

# img = cv2.imread('sudoku.png', 0) #also check with sudoku.png and messi5.jpg

# lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3) #has negative float numbers # ksize is optional
# lap = np.uint8(np.absolute(lap)) #converts back to absolute unsigned 8bit charachter

# sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0) #for dx 1=sobelX and 0=sobelY
# sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1) #ksize can also be provided
# sobelX = np.uint8(np.absolute(sobelX))
# sobelY = np.uint8(np.absolute(sobelY))

# sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# titles = ['image', 'laplacian', 'sobelX', 'sobelY', 'sobel combined']
# images = [img, lap, sobelX, sobelY, sobelCombined]

# for i in range(5):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])

# plt.show()


## 18.CANNY EDGE DETECTION

# def nothing(x):
#     pass

# img = cv2.imread('messi5.jpg', 0)
# cv2.namedWindow('trackBar')

# cv2.createTrackbar('Threshold 1', 'trackBar', 0, 255, nothing)
# cv2.createTrackbar('Threshold 2', 'trackBar', 255, 255, nothing)

# while(1):
#     th_1=cv2.getTrackbarPos('Threshold 1', 'trackBar')
#     th_2=cv2.getTrackbarPos('Threshold 2', 'trackBar')

#     canny = cv2.Canny(img, th_1, th_2)

    # cv2.imshow('image', img)
    # cv2.imshow('canny', canny)

    # key_in = cv2.waitKey(1)

    # if key_in == 27:
    #     break
    

# cv2.destroyAllWindows()


## 19A. IMAGE PYRAMIDS

# img=cv2.imread('messi5.jpg')

# lr1 = cv2.pyrDown(img)
# lr2 = cv2.pyrDown(lr1)

# hr1 = cv2.pyrUp(lr2)
# hr2 = cv2.pyrUp(hr1)

# cv2.imshow('image',img)
# cv2.imshow('Lower resolution',lr1)
# cv2.imshow('Even Lower resolution',lr2)
# cv2.imshow('Higher resolution of Even Lower resolution',hr1)
# cv2.imshow('Even higher Higher resolution of Even Lower resolution',hr2)

# cv2.waitKey()

# cv2.destroyAllWindows()


## 20.FIND AND DRAW CONTOUR

# img=cv2.imread('opencv-logo.png')
# imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret,  thresh = cv2.threshold(imgray, 50, 255, 0)
# contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # contour is a list of all images and heirarchy is optional vector

# print('number of contours= '+ str(len(contours)))

# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# cv2.imshow('image',img)
# cv2.imshow('image gray',imgray)

# cv2.waitKey(0)

# cv2.destroyAllWindows()


## 21. MOTION DETECTION AND TRACKING USING CONTOURS

# cap=cv2.VideoCapture('vtest.avi')

# ret, frame1=cap.read()
# ret, frame2=cap.read()

# while(cap.isOpened()):
#     diff = cv2.absdiff(frame1, frame2)
#     gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)

#     _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
#     dilated = cv2.dilate(thresh, None, iterations=3)
#     contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
        
#         if cv2.contourArea(contour) < 1000: 
#             continue #to remove contour with less area
#         cv2.rectangle(frame1, (x,y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame1, "Status: {}".format('Movement'), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

#     # cv2.drawContours(frame1, contours, -1, (0,255,0), 2)

#     cv2.imshow('feed', frame1)
#     cv2.imshow('difference', diff)
#     cv2.imshow('thresh', thresh)
#     cv2.imshow('dilated', dilated)
#     frame1 = frame2
#     ret, frame2 = cap.read()

#     if cv2.waitKey(40) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


## 22.DETECTING SIMPLE SHAPES

# img = cv2.imread('shapes.png')
# imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# _, thresh =cv2.threshold(imgray, 240, 255, cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True) #approximate polygon with certain precision #arc length calculate contours' parameter
#     cv2.drawContours(img, [approx], 0, (0,255,0), 3)
#     x = approx.ravel() [0]
#     y = approx.ravel() [1] # coordinates of shapes #approx[1][0][0] se pehla vetrex ka x coordiante measure kar sakte hai aur approx[i][0] se ith vertex ka tupple aa jayega
#     if len(approx) == 3:
#         cv2.putText(img, 'Triangle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)

#     elif len(approx) == 4:
#         x, y, w, h = cv2.boundingRect(approx)
#         aspectRatio = float(w) /h
#         if aspectRatio >=0.95 and aspectRatio<=1.05:
#             cv2.putText(img, 'Square', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
#         else:
#             cv2.putText(img, 'Rectangle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
        

#     elif len(approx) == 5:
#         cv2.putText(img, 'Pentagon', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)

#     elif len(approx) == 6:
#         cv2.putText(img, 'Hexagon', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)

#     else:
#         cv2.putText(img, 'Polygon', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)

# cv2.imshow('shapes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


## 23A. HISTOGRAM

# img = cv2.imread('messi5.jpg')

# b, g, r =cv2.split(img)

# cv2.imshow('img', img)
# cv2.imshow('b', b)
# cv2.imshow('g', g)
# cv2.imshow('r', r)

# plt.hist(b.ravel(), 256, (0,256))
# plt.hist(g.ravel(), 256, (0,256))
# plt.hist(r.ravel(), 256, (0,256))
# plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()


## 23B. HISTOGRAM

# img = cv2.imread('messi5.jpg')

# b, g, r =cv2.split(img)

# cv2.imshow('img', img)
# cv2.imshow('b', b)
# cv2.imshow('g', g)
# cv2.imshow('r', r)

# hist = cv2.calcHist([img], [0,1,2], None, [256], (0,256))
# plt.plot(hist)

# cv2.waitKey(0)
# cv2.destroyAllWindows()