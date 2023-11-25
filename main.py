import os
import csv
import statistics

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import json

SHOW = False

cls_ids = [2, 5, 7]
cls_ids_dict = {'car': 2, 'bus': 5, 'van': 7}


def crop(point_bb):
    if point_bb[0] < 0:
        point_bb[0] = 0
    if point_bb[0] >= w_im:
        point_bb[0] = w_im - 1
    if point_bb[1] < 0:
        point_bb[1] = 0
    if point_bb[1] >= h_im:
        point_bb[1] = h_im - 1
    point_bb = [int(point_bb[0]), int(point_bb[1])]
    return point_bb


def cross_line(line, point_cur, point_prev):
    x1, y1 = line[0]
    x2, y2 = line[1]
    x3, y3 = point_cur
    x4, y4 = point_prev
    D3 = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)
    D4 = (x4 - x1) * (y2 - y1) - (y4 - y1) * (x2 - x1)
    return D3 * D4 < 0

video_path = "Video2/" 

from os import listdir
from os.path import isfile, join

fn_ext = ".mp4"

for fn in listdir(video_path): 
    if isfile(join(video_path, fn)) and fn.endswith(fn_ext):
        name_file = fn[:-4]
        #name_file = "KRA-2-7-2023-08-22-morning"

        f = open("markup/jsons/" + name_file + ".json")
        data = json.load(f)
        f.close()
        areas = data['areas']
        zones = data['zones']
        directions = data['directions']

        track_history = defaultdict(lambda: [])
        id_frame_in = defaultdict(lambda: [])
        id_frame_out = defaultdict(lambda: [])
        id_vel = defaultdict(lambda: [])
        id_class = defaultdict(lambda: [])
        class_count = defaultdict(lambda: [])
        result_out = defaultdict(lambda: [])
        id_in_area = defaultdict(lambda: [])


        cap = cv2.VideoCapture(video_path + name_file + fn_ext)
        h_im = 720
        w_im = 1280
        fps = 25
        if cap.isOpened():
            #w_im = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            #h_im = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)

        areas = [[[x * w_im, y * h_im] for x, y in areas[i]] for i in range(len(areas))]
        zones = [[[x * w_im, y * h_im] for x, y in zones[i]] for i in range(len(zones))]

        image_mask_areas = []
        for i in range(len(areas)):
            points = np.hstack(areas[i]).astype(np.int32).reshape((-1, 1, 2))
            image_mask = np.zeros((int(h_im), int(w_im), 3), dtype="uint8")
            image_mask = cv2.fillPoly(image_mask, [points], color=(1, 1, 1))
            image_mask_areas.append(image_mask)
            #cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            #cv2.imshow("mask", image_mask * 255)
            #cv2.waitKey(0)

        #kernel = np.ones((200, 200), dtype="uint8")
        #image_mask = cv2.dilate(image_mask, kernel, iterations=1)

        # Load the YOLOv8 model
        model = YOLO('yolov8n.pt')
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) % 100 == 0:
                    print(cap.get(cv2.CAP_PROP_POS_FRAMES) * 100./cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if cap.get(cv2.CAP_PROP_POS_FRAMES) % 2 == 0:
                    continue
                
                frame = cv2.resize(frame, (w_im, h_im), interpolation = cv2.INTER_LINEAR)
                
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                #results = model.track(frame, device='cpu', persist=True, classes=cls_ids, verbose=False)
                results = model.track(frame, device=0, persist=True, classes=cls_ids, verbose = False )
                
                if SHOW:
                    annotated_frame = results[0].plot()

                if results[0].boxes.id == None:
                    continue

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                classes_ids = results[0].boxes.cls.cpu().tolist()
                #boxes = results[0].boxes.xyxy.cpu()

                # Plot the tracks
                for box, track_id, class_id in zip(boxes, track_ids, classes_ids):
                    x, y, w, h = box
                    id_class[track_id].append(class_id)
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    #print(track_id, track)
                    if len(track) > 1000:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    lines_points_in = [zones[0][3], zones[0][0]]
                    lines_points_out = [zones[1][3], zones[1][0]]

                    if SHOW:
                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                        for i in range(len(areas)):
                            points = np.hstack(areas[i]).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [points], isClosed=True, color=(0, 230, 230), thickness=5)

                        for i in range(len(zones)):
                            points = np.hstack(zones[i]).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [points], isClosed=True, color=(230, 0, 230), thickness=5)

                        # отобразим линии пересечения въезда машины и выезда машины
                        cv2.line(annotated_frame, [int(lines_points_in[0][0]), int(lines_points_in[0][1])],
                                [int(lines_points_in[1][0]), int(lines_points_in[1][1])], color=(255, 255, 0), thickness=5)
                        cv2.line(annotated_frame, [int(lines_points_out[0][0]), int(lines_points_out[0][1])],
                                [int(lines_points_out[1][0]), int(lines_points_out[1][1])], color=(255, 255, 0), thickness=5)

                    if len(track) > 1:
                        # посмотрим попадает ли в зону задетектированная машина
                        left_top = [int(x - w/2), int(y - h/2)]
                        left_top = crop(left_top)
                        right_bottom = [int(x + w/2), int(y + h/2)]
                        right_bottom = crop(right_bottom)

                        left_bottom = [int(x-w/2), int(y+h/2)]
                        left_bottom = crop(left_bottom)
                        right_top = [int(x+w/2), int(y-h/2)]
                        right_top = crop(right_top)

                        num_iter_area = 0
                        for mask in image_mask_areas:
                            in_area = ((np.sum(mask[left_top[1], left_top[0]] + mask[right_bottom[1], right_bottom[0]])) >= 3) or \
                                        ((np.sum(mask[left_bottom[1], left_bottom[0]] + mask[right_top[1], right_top[0]])) >= 3)
                            if in_area:
                                id_in_area[track_id].append(num_iter_area)
                            num_iter_area += 1

                            if in_area == False and SHOW:
                                cv2.line(annotated_frame, left_bottom, right_top, color=(255, 255, 0), thickness=5)
                                cv2.line(annotated_frame, left_top, right_bottom, color=(255, 255, 0), thickness=5)

                        is_in = cross_line(lines_points_in, track[-1], track[-2])
                        is_out = cross_line(lines_points_out, track[-1], track[-2])

                        if is_in:
                            id_frame_in[track_id].append(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            if SHOW:
                                print("въехал track_id:", track_id, "frame: ", id_frame_in[track_id], ", classes_ids:", id_class[track_id][0])
                                font = cv2.FONT_HERSHEY_COMPLEX
                                cv2.putText(annotated_frame,'in',(int(x), int(y)), font, 2, (0,0,255), 3)

                        if is_out:
                            id_frame_out[track_id].append(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            if SHOW:
                                print("выехал track_id:", track_id, "frame: ", id_frame_out[track_id])
                                print( [int(lines_points_out[0][0]), int(lines_points_out[0][1])], [int(lines_points_out[1][0]), int(lines_points_out[1][1])])
                                font = cv2.FONT_HERSHEY_COMPLEX
                                cv2.putText(annotated_frame, 'out', (int(x), int(y)), font, 2, (0, 0, 255), 3)

                if SHOW:
                    # Display the annotated frame
                    cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)
                    #cv2.waitKey(0)
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    if cv2.waitKey(1) & 0xFF == ord("ы"):
                        SHOW = False
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

        mean_vels = [0.] * len(cls_ids)
        mean_vels_sign = [0.] * len(cls_ids)
        mean_vels_pos = [0.] * len(cls_ids)
        mean_vels_neg = [0.] * len(cls_ids)
        count_cls = [0] * len(cls_ids)
        count_cls_pos = [0] * len(cls_ids)
        count_cls_neg = [0] * len(cls_ids)
        count_cls_total = [0] * len(cls_ids)

        for track_id, v in id_frame_out.items():
            frame_in = id_frame_out.get(track_id, 0)
            frame_out = id_frame_in.get(track_id, 0)
            if frame_in == 0 or frame_out == 0:
                if frame_in != 0:
                    for i in range(len(cls_ids)):
                        if id_class[track_id][0] == cls_ids[i]:
                            count_cls_total[i] += 1
                continue

            for i in range(len(cls_ids)):
                # максимальное появление класса
                cur_cls = statistics.median(id_class[track_id])
                id_class[track_id][0] = cur_cls
                if cur_cls == cls_ids[i]:
                    count_cls_total[i] += 1

            if len(frame_in) > 1 or len(frame_out) > 1:
                print("track_id: ", track_id, "frame_in: ", frame_in, "frame_out: ", frame_out)

            print("areas: ", len(id_in_area[track_id]))
            if len(id_in_area[track_id]) > 2:
                diff_frames = (frame_out[0] - frame_in[0])
                vel = 20. / (diff_frames / float(fps)) * 3.6
                if vel > 90:
                    continue

                for i in range(len(cls_ids)):
                    if id_class[track_id][0] == cls_ids[i]:
                        mean_vels[i] += abs(vel)
                        mean_vels_sign[i] += vel
                        count_cls[i] += 1
                        if vel < 0:
                            mean_vels_neg[i] += vel
                            count_cls_neg[i] += 1
                        if vel > 0:
                            mean_vels_pos[i] += vel
                            count_cls_pos[i] += 1

        for i in range(len(cls_ids)):
            if count_cls[i] == 0:
                mean_vels[i] = 0
            else:
                mean_vels[i] = mean_vels[i]/count_cls[i]
                mean_vels_sign[i] = mean_vels_sign[i]/count_cls[i]

            if count_cls_neg[i] == 0:
                mean_vels_neg[i] = 0
            else:
                mean_vels_neg[i] = mean_vels_neg[i]/count_cls_neg[i]

            if count_cls_pos[i] == 0:
                mean_vels_pos[i] = 0
            else:
                mean_vels_pos[i] = mean_vels_pos[i] / count_cls_pos[i]
            print("abs class_id: ", cls_ids[i], ", count: ",  count_cls[i], ", total_count: ", count_cls_total[i], ", mean_vel:", mean_vels[i])
            print("sig class_id: ", cls_ids[i], ", count: ", count_cls[i], ", total_count: ", count_cls_total[i], ", mean_vel_sign:", mean_vels_sign[i])
            print("neg class_id: ", cls_ids[i], ", count: ", count_cls_neg[i], ", total_count: ", count_cls_total[i], ", mean_vel_neg:", mean_vels_neg[i])
            print("pos class_id: ", cls_ids[i], ", count: ", count_cls_pos[i], ", total_count: ", count_cls_total[i], ", mean_vel_pos:", mean_vels_pos[i])
        
        f = open("out_res_Denis.txt", "a")
 
        #str = file_name,car,quantity_car,average_speed_car,van,quantity_van,average_speed_van,bus,quantity_bus,average_speed_bus
        str = "{0},car,{1},{2},van,{3},{4},bus,{5},{6}\n".format(name_file,count_cls_total[0],round(mean_vels[0], 2),count_cls[1],round(mean_vels[1], 2),count_cls[2],round(mean_vels[2], 2))
        f.write(str)
        f.close()
