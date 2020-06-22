
import cv2
import argparse
import numpy as np
import imageio

parser = argparse.ArgumentParser(description='CCtv Recap')

parser.add_argument('--input', help='Path to video or to sequence of image', default='highwayTraffic.mp4')
parser.add_argument('--algo', help='Background Subtraction Method (KNN, MOG2)', default='MOG2')

args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()


capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))

recapVideo = args.input.split('.')[0] + 'Recap.avi'
if not capture.isOpened:
    print('Unable to open: ', args.input)

frameIds = capture.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
fps = capture.get(cv2.CAP_PROP_FPS)
nFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
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
grayBackground = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
grayBackground = cv2.GaussianBlur(grayBackground, (5, 5), 0)

frameArray = []
jump = 0

size = (shp[0], shp[1])
#writer = cv2.VideoWriter('2.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, size)
video_writer = imageio.get_writer(recapVideo, 'mp4', mode='I', fps=15)

nObjects = 5
nGap = int(nFrames/nObjects)
print(nGap)
for i in range(nGap + 1):
    outputFrame = np.zeros([shp[0], shp[1], shp[2]])
    mainFrame = np.zeros([shp[0], shp[1], shp[2]])
    #mainFrame = mainFrame.astype(dtype=np.uint8)
    mainFrame = background.copy()
    allMovngObj = np.zeros([shp[0], shp[1], shp[2]])
    allMovngObj = allMovngObj.astype(dtype=np.uint8)

    minutes = []
    seconds = []
    cor_x = []
    cor_y = []
    for obj in range(nObjects):
        movngObj = np.zeros([shp[0], shp[1], shp[2]])
        movngObj = movngObj.astype(dtype=np.uint8)

        if i % nGap == 0 and i != 0:
            jump += nGap * nObjects

        frameNo = ((obj * nGap) + i + jump)

        capture.set(cv2.CAP_PROP_POS_FRAMES, frameNo)

        ret, frame = capture.read()
        if ret == False:
            continue
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        grayFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)

        #fgmask = backSub.apply(frame)
        fgmask = cv2.absdiff(grayFrame, grayBackground)
        _, mask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
        mask[mask == 127] = 0
        th, thresh = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        dur = int(frameNo/fps)
        min = int(dur/60)
        sec = dur % 60
        x = 0
        y = 0
        s1 = 200
        s2 = 5000
        for cnt in contours:
            if s1 < cv2.contourArea(cnt) < s2:
                x, y, w, h = cv2.boundingRect(cnt)
                y = y - 10
                h = h + 10
                cv2.rectangle(mainFrame, (x, y - 13), (x + 35, y), (200, 200, 200), -1)
                timePeriod = str(min) + ':' + str(sec)
                cv2.putText(mainFrame, timePeriod, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
                for j in range(y, y+h):
                    for k in range(x, x+w):
                        if (mask[j][k] > 0):
                            movngObj[j][k][0] = frame[j][k][0]
                            movngObj[j][k][1] = frame[j][k][1]
                            movngObj[j][k][2] = frame[j][k][2]
                            mainFrame[j][k][0] = 0
                            mainFrame[j][k][1] = 0
                            mainFrame[j][k][2] = 0
        minutes.append(min)
        seconds.append(sec)
        cor_x.append(x)
        cor_y.append(y)
        allMovngObj = cv2.add(allMovngObj, movngObj)
    outputFrame = cv2.add(mainFrame, allMovngObj)
    cv2.imshow('res', outputFrame)
    key = cv2.waitKey(2)
    if key == 'q':
        break
    #writer.write(outputFrame)
    outputFrame = cv2.cvtColor(outputFrame, cv2.COLOR_RGB2BGR)
    video_writer.append_data(np.asarray(outputFrame))
    print(i)
capture.release()
#writer.release()
cv2.destroyAllWindows()
