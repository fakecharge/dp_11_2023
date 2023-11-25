import json
from datetime import datetime

import pandas

motion_list = [None, None]
time = []

df = pandas.DataFrame(columns=["Start", "End"])

import numpy as np
import cv2

cap = cv2.VideoCapture('KRA-2-7-2023-08-23-evening.mp4')
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

with open("KRA-2-7-2023-08-23-evening.json", 'r') as f:
    data = json.load(f)
areas = data['areas']
zones = data['zones']
directions = data['directions']

video = cv2.VideoCapture('KRA-2-7-2023-08-23-evening.mp4')

if cap.isOpened():
    w_im = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    h_im = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)


areas = [[[x * w_im, y * h_im] for x, y in areas[i]] for i in range(len(areas))]
zones = [[[x * w_im, y * h_im] for x, y in zones[i]] for i in range(len(zones))]

image_mask = np.zeros((int(h_im), int(w_im), 3), dtype="uint8")

for i in range(len(areas)):
    points = np.hstack(areas[i]).astype(np.int32).reshape((-1, 1, 2))
    image_mask = cv2.fillPoly(image_mask, [points], color=(1, 1, 1))

frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

medianFrame *= image_mask
static_back = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
static_back = cv2.GaussianBlur(static_back, (21, 21), 0)
# static_back *= image_mask

k = 0
while True:
    # break
    k += 1
    print(k)
    # display.publish_display_data(f"count: {k}")
    check, frame = video.read()

    if not check:
        break

    frame *= image_mask

    motion = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # gray *= image_mask

    if static_back is None:
        static_back = gray
        continue

    diff_frame = cv2.absdiff(static_back, gray)

    thresh_frame = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)[1]

    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        motion = 1

    motion_list.append(motion)
    motion_list = motion_list[-2:]

    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now())

        # Appending End time of motion
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now())

    # _, frame = cv2.imencode('.jpeg', frame)
    # _, thresh_frame = cv2.imencode('.jpeg', thresh_frame)

    # display_handle.clear()
    cv2.imshow("rgb", frame)
    # cv2.imshow("ИЛЬЯ ЧМОШНИК", thresh_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
