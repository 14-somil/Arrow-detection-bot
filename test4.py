import numpy as np
import cv2

def angle_between(p1,p2,p3):
    a=np.array(p1)
    b=np.array(p2)
    c=np.array(p3)

    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return (np.degrees(angle))

cap = cv2.VideoCapture(0)

while (1):
    _, frame = cap.read()
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh =cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        check = False
        approx = cv2.approxPolyDP(contour, 0.04*cv2.arcLength(contour, True), True) 
        # print(len(approx))
        if cv2.contourArea(contour)>1000:
            if len(approx)==7:
                angle=[]
                head=[]
                tail=[]

                for i in range (7): #finding acute angles of all the vertex
                    vertex = angle_between(approx[i-1][0], approx[i][0], approx[0][0] if i==6 else approx[i+1][0])
                    angle.append(vertex)
                # print (angle)

                # 4 consecutive right angles dhund raha hun aur unn 4 ke adjacent dono angle acute honge
                for i in range(7):
                    if 0<= (angle[i-1] if (i-1)>=0 else angle[(i-1)+7]):
                        if 80<=angle[i]<=100:
                            if 80<= (angle[i+1] if (i+1)<7 else angle[(i+1)-7]) <=100:
                                if 80<= (angle[i+2] if (i+2)<7 else angle[(i+2)-7]) <=100:
                                    if 80<= (angle[i+3] if (i+3)<7 else angle[(i+3)-7]) <=100:
                                        if 0<= (angle[i+4] if (i+4)<7 else angle[(i+4)-7]) <=90:
                                            check = True
                                            head = approx[i-2][0] if (i-2)<7 else approx[(i-2)+7][0]

                                            tail[0] = approx[(i+1) if (i+1)]

                                            # tail = approx[i+1][0] if (i+1)<7 else approx[(i+1)-7][0]
                                            print('check')
                                            break
                                        
                if check == True:
                    slope = (head[1] - tail[1]) / (head[0] - tail[0])
                    if (-10 <= slope <= 10):
                        if()
                    cv2.putText(frame, 'Arrow', approx[0][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

            else:
                continue
        
    cv2.imshow('frame', frame)
    cv2.imshow('gray', imgray)
    cv2.imshow('thresh', thresh)
    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()