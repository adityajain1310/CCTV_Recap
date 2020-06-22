from collections import deque
import cv2
import argparse
import numpy as np
import imageio

parser = argparse.ArgumentParser(description='CCtv Recap')

parser.add_argument('--input', help='Path to video or to sequence of image', default='highwayTraffic.mp4')
parser.add_argument('--algo', help='Background Subtraction Method (KNN, MOG2)', default='MOG2')

args = parser.parse_args()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
#video_writer = imageio.get_writer('sam.avi', 'mp4', mode='I', fps=15)

frameIds = capture.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
frames = []
shp = []
for fid in frameIds:
    capture.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = capture.read()
    if ret == False:
        continue
    shp = frame.shape
    frames.append(frame)

background = np.median(frames, axis=0).astype(dtype=np.uint8)
graybgrd = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
graybgrd = cv2.GaussianBlur(graybgrd, (5, 5), 0)


'''re, thresh = cv2.threshold(fgmask, 80, 255, cv2.THRESH_BINARY)
a, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for i in range(frame.shape[0]):
    for j in range(frame.shape[1]):
        if(fgmask[i][j] == 255):
            cv2.putText(frame, '1', (i, j), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)'''
i = 0
while True:
    ret, frame = capture.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #fgmask = backSub.apply(frame)
    grayfrm = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grayfrm = cv2.GaussianBlur(grayfrm, (5, 5), 0)

    fgmask = cv2.absdiff(graybgrd, grayfrm)
    _, thres = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
    fgmask[fgmask == 127] = 0
    re, thresh = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    s1 = 200
    s2 = 20000
    xcnts = []
    for cnt in contours:
        if s1 < cv2.contourArea(cnt) < s2:

            #print(x, " ", y)

                #cv2.imshow("img", frame)
            #cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 3)
            #cv2.putText(frame, '.', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
            x, y, w, h = cv2.boundingRect(cnt)
            y = y - 10
            h = h + 10
            cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 200, 200), -1)
    i+=1
    cv2.imshow('image2',thres)
    #video_writer.append_data(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        #video_writer.release()
        cv2.destroyAllWindows()


