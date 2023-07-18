import cv2
import mediapipe as mp
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

mphand=mp.solutions.hands
Hand=mphand.Hands(max_num_hands=2)
mpDraw=mp.solutions.drawing_utils
cap=cv2.VideoCapture(0)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
pTime = 0
vol = 0
volBar = 400
volPer = 0
minVol = volRange[0]
maxVol = volRange[1]
print(minVol,maxVol)
while True:
    _,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=Hand.process(imgRGB)

    lm_list=[]
    if results.multi_hand_landmarks:
        for handle in results.multi_hand_landmarks:
            for id,lm in enumerate(handle.landmark):
                h,w,c=img.shape
                cx,cy=int((lm.x)*w),int((lm.y)*h)
                lm_list.append([id,cx,cy])
            mpDraw.draw_landmarks(img,handle,mphand.HAND_CONNECTIONS,mpDraw.DrawingSpec(color=(0,0,255)),mpDraw.DrawingSpec(color=(0,255,0)))
    if len(lm_list) != 0:
        print(lm_list[4],lm_list[8])
        x1,y1=lm_list[4][1],lm_list[4][2]
        x2,y2=lm_list[8][1],lm_list[8][2]
        cx1,cy1=(x1+x2)//2,(y1+y2)//2
        cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img, (cx1, cy1), 10, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [50, 270], [minVol, maxVol])
        volBar = np.interp(length, [50, 270], [400, 150])
        volPer = np.interp(length, [50, 270], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        if length < 50:
            cv2.circle(img, (cx1, cy1), 10, (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"{int(volPer)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    cv2.imshow("image",img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break