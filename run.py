import imutils
from keras.models import load_model
from keras.preprocessing  import  image
import tensorflow as tf


from PIL import Image
import math
import json
import cv2
import numpy as np
import pyautogui
import tensorflow
import time
model=load_model("hand_pose.h5")
cap = cv2.VideoCapture(0)


def minimum(a,n ):
    # inbuilt function to find the position of minimum
    minpos = a.index(min(a))

    # inbuilt function to find the position of maximum
    maxpos = a.index(max(a))
    #print(a[maxpos])

    # printing the position
    #print("The maximum is at position", maxpos + 1)

    if a[maxpos]>0.9 and maxpos + 1 == 1:
        return ("close_hand")
    elif a[maxpos]>0.9 and maxpos + 1 == 2:
        return ("one_finger")
    elif a[maxpos]>0.9 and maxpos + 1 == 3:
        return ("open_hand")
    elif a[maxpos]>0.8 and maxpos + 1 == 4:
        return ("two_finger")
def saving(thresh ):
    with open('data collection/count.txt', 'r') as filehandle:
        basicList = json.load(filehandle)


    file = open('data collection/count.txt', "r+")
    file.truncate(0)
    file.close()

    with open('data collection/count.txt', 'w') as filehandle:
        json.dump(basicList-1, filehandle)
        dp=basicList
        nam = ("dump/tst{}.jpg".format(dp))

    return nam


    cv2.imwrite(nam,thresh)
    print(type(thresh))
i=1020
while True:
    points = []
    point1 = []
    point2 = []

    success, img = cap.read()

    cv2.rectangle(img,(200,200),(450,450),(0,255,0),0)
    crop_img=img[200:450,200:450]

    blur=cv2.GaussianBlur(crop_img
                          ,(3,3),0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    '''lower- [ 0 46 51] upper- [ 54 184 255]'''
    l= np.array([0, 46 ,51])
    u= np.array([54, 184 ,255])
    mask1 = cv2.inRange(hsv, l, u)


    kernel=np.zeros((5,5),np.uint8)

    dil=cv2.dilate(mask1,kernel,iterations=1)
    ero=cv2.erode(dil,kernel,iterations=1)

    fil=cv2.GaussianBlur(mask1,(3,3),0)
    ret,thresh=cv2.threshold(fil,127,255,0)
    cv2.imshow("thresh", thresh)

    pr = cv2.resize(crop_img, (224, 224))
    im = Image.fromarray(pr)

    im_array = np.array(im)

    im_array = np.expand_dims(im_array, axis=0)

    prd = model.predict(im_array)

    #im_array = np.expand_dims(thresh, axis=0)
    #prd = model.predict(im_array)
    #print(prd)
    #ltt=[]
    lst=[]
    '''for i in range(4):
        if prd[0][i]>0.8:
            lst.append(i)
            #ltt.append(i)'''

    lst = [prd[0][0], prd[0][1], prd[0][2], prd[0][3]]
    tx=minimum(lst,len(lst))
    #print(len(lst))
    cv2.putText(img,tx, (150,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cnts4 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts4 = imutils.grab_contours(cnts4)
    if tx=="one_finger":


        for c in cnts4:
            area1 = cv2.contourArea(c)
            if area1 > 5000:
                #cv2.drawContours(img, [c], -1, (0, 255, 0), 3)

                M = cv2.moments(c)

                cx = int(M['m10'] / M['m00']) + 550
                cy = int(M['m01'] / M['m00']) + 550

                #extLeft = tuple(c[c[:, :, 0].argmin()][0])
                #extRight = tuple(c[c[:, :, 0].argmax()][0])
                #extTop = tuple(c[c[:, :, 1].argmin()][0])
                #extBot = tuple(c[c[:, :, 1].argmax()][0])

                #cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
                #cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
                #cv2.circle(img, extRight, 8, (0, 255, 0), -1)
                #cv2.circle(img, extTop, 8, (255, 0, 0), -1)
                #cv2.circle(img, extBot, 8, (255, 255, 0), -1)



                extTop = tuple(c[c[:, :, 1].argmin()][0])
                #cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
                cv2.circle(img, extTop, 8, (255, 0, 0), -1)
                if extTop[0]<100:
                    #v=int(90-extTop[0])
                    pyautogui.hotkey('volumeup')

                if extTop[0]>100:
                    #v = int(extTop[0]-90)
                    pyautogui.hotkey('volumedown')

                print(extTop)

                '''
                ,presses=15*v
                if tx=="two_finger":
                    pyautogui.click(clk)'''
                #pyautogui.moveRel(extTop)
                #pyautogui.scroll(exttop[0])


    if tx=="two_finger":


        for c in cnts4:
            area1 = cv2.contourArea(c)
            if area1 > 5000:
                #cv2.drawContours(img, [c], -1, (0, 255, 0), 3)

                M = cv2.moments(c)

                cx = int(M['m10'] / M['m00']) + 550
                cy = int(M['m01'] / M['m00']) + 550

                #extLeft = tuple(c[c[:, :, 0].argmin()][0])
                #extRight = tuple(c[c[:, :, 0].argmax()][0])
                #extTop = tuple(c[c[:, :, 1].argmin()][0])
                #extBot = tuple(c[c[:, :, 1].argmax()][0])

                #cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
                #cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
                #cv2.circle(img, extRight, 8, (0, 255, 0), -1)
                #cv2.circle(img, extTop, 8, (255, 0, 0), -1)
                #cv2.circle(img, extBot, 8, (255, 255, 0), -1)



                extTop = tuple(c[c[:, :, 1].argmin()][0])
                #cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
                cv2.circle(img, extTop, 8, (255, 0, 0), -1)
                '''if extTop[0]<55:
                    pyautogui.hotkey('fn','right',presses=3)

                if extTop[0]>55:
                    pyautogui.hotkey('fn','left',presses=3)'''
                #print(extTop)

    if tx=="open_hand":


        for c in cnts4:
            area1 = cv2.contourArea(c)
            if area1 > 5000:
                #cv2.drawContours(img, [c], -1, (0, 255, 0), 3)

                M = cv2.moments(c)

                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)

                '''#extLeft = tuple(c[c[:, :, 0].argmin()][0])
                #extRight = tuple(c[c[:, :, 0].argmax()][0])
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                #extBot = tuple(c[c[:, :, 1].argmax()][0])

                cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
                #cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
                #cv2.circle(img, extRight, 8, (0, 255, 0), -1)
                cv2.circle(img, extTop, 8, (255, 0, 0), -1)
                #cv2.circle(img, extBot, 8, (255, 255, 0), -1)'''
                if cy<140 :
                    #v=int(90-extTop[0])
                    pyautogui.hotkey("window", "up")


                if cy>140 :
                    #v = int(extTop[0]-140)
                    pyautogui.hotkey("window", "down")

                print(cy)

    if tx=="close_hand":


        for c in cnts4:
            area1 = cv2.contourArea(c)
            if area1 > 5000:
                #cv2.drawContours(img, [c], -1, (0, 255, 0), 3)

                M = cv2.moments(c)

                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(img, (cx, cy), 8, (255, 0, 255), -1)
                '''#extLeft = tuple(c[c[:, :, 0].argmin()][0])
                #extRight = tuple(c[c[:, :, 0].argmax()][0])
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                #extBot = tuple(c[c[:, :, 1].argmax()][0])

                cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
                #cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
                #cv2.circle(img, extRight, 8, (0, 255, 0), -1)
                cv2.circle(img, extTop, 8, (255, 0, 0), -1)
                #cv2.circle(img, extBot, 8, (255, 255, 0), -1)'''

    cv2.imshow("tracking", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

