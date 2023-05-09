import cv2
import numpy as np

def angle_between(p1,p2,p3):
    a=np.array(p1)
    b=np.array(p2)
    c=np.array(p3)

    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return (np.degrees(angle))

img = cv2.imread('arrow_3.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh =cv2.threshold(imgray, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    check = False
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True) 
    print(len(approx))
    if len(approx)==7:
        # for i in range(7):
        #     print(approx[i][0])
        #     cv2.putText(img, str(i), approx[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1)
        angle=[]

        print(approx[0][0])

        for i in range (7): #finding acute angles of all the vertex
            vertex = angle_between(approx[i-1][0], approx[i][0], approx[0][0] if i==6 else approx[i+1][0])
            angle.append(vertex)
        print (angle)

        # 4 consecutive right angles dhund raha hun
        for i in range(7):
            if 0<= (angle[i-1] if (i-1)>=0 else angle[(i-1)+7]):
                if 80<=angle[i]<=100:
                    if 80<= (angle[i+1] if (i+1)<7 else angle[(i+1)-7]) <=100:
                        if 80<= (angle[i+2] if (i+2)<7 else angle[(i+2)-7]) <=100:
                            if 80<= (angle[i+3] if (i+3)<7 else angle[(i+3)-7]) <=100:
                                if 0<= (angle[i+4] if (i+4)<7 else angle[(i+4)-7]) <=90:
                                    check = True
                                    print('check')
                                    break
        
        if check == True:
            cv2.putText(img, 'Arrow', approx[0][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        
    else:
        continue
cv2.imshow('arrows', img)
cv2.waitKey(0)
cv2.destroyAllWindows()