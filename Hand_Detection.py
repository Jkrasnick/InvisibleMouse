#HAND

import math

import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pyautogui
from google.protobuf.json_format import MessageToDict

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils

    mp_hands = mp.solutions.hands

    pTime = 0
    # capture from webcam
    cap = cv2.VideoCapture(0)
    counter = 0

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

    currScroll = 0;
    while (cap.isOpened()):
        counter = counter + 1
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (2000, 1000))
        if success == False:
            break
        image_height, image_width, _ = img.shape
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        lmList = []
        rl = []
        currentMouseX = 100
        currentMouseY = 100
        click = False
        clicked = False
        if results.multi_hand_landmarks:
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image=img, landmark_list=hand_landmarks,
                                          connections=mp_hands.HAND_CONNECTIONS)
                lmList.append(hand_landmarks)

            for i in results.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']
                rl.append(label)
        count = 0
        for handy in lmList:
            if rl[count] == 'Left':
                if ((math.sqrt((int(handy.landmark[4].y*image_width - handy.landmark[8].y*image_width)**2 + int(handy.landmark[4].x*image_height - handy.landmark[8].x*image_height)**2)) < 75)and not(clicked)):
                    click = True
                    print("CLICK")
                else:
                    click = False
                    clicked = True
                if (count % 5 == 0):
                    if currScroll == 0:
                        currScroll = handy.landmark[20].y * image_height
                    else:
                        currScroll = currScroll - handy.landmark[20].y * image_height
            else:
                currScroll = 0
            if rl[count] == 'Right':
                cv2.circle(img, (int(handy.landmark[8].x * image_width), int(handy.landmark[8].y * image_height)), 10,
                           (0, 0, 255), 3)
                currentMouseX = int(handy.landmark[8].x * image_width)
                currentMouseY = int(handy.landmark[8].y * image_height)



            count += 1

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # out.write(img
        if (counter % 2 == 0):
            if (click):
                pyautogui.click(currentMouseX, currentMouseY, 1, 0, 'left')
                click = False
            if (currScroll != 0):
                pyautogui.scroll(currScroll/100)
                print(currScroll/100)
            print(currScroll/100)
            pyautogui.moveTo(currentMouseX, currentMouseY, duration=0.1)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
