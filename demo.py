#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

import csv
import random

warnings.filterwarnings('ignore')

def main(yolo):

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 2.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=150, n_init=3)

    writeVideo_flag = True
    asyncVideo_flag = False

    file_path = 'input/orig_video.mp4'
    if asyncVideo_flag :
        video_capture = VideoCaptureAsync(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/output_yolov3.avi', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    track_count = 0
    unique_track_ids_set = set()
    track_to_display_mapping_dict = dict()

    with open('output/db.csv', mode='w') as db_csv:
        csv_writer = csv.writer(db_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Frame #', 'GPS(Lat)', 'GPS(Lon)', 'People Count (frame)', 'People Count (video)'])

        while True:
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if ret != True or frame_index == 10:
                 break

            t1 = time.time()

            image = Image.fromarray(frame[...,::-1])  # bgr to rgb
            detected_image = yolo.detect_image(image)
            boxs = detected_image[0]
            confidence = detected_image[1]

            features = encoder(frame,boxs)

            detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(boxs, confidence, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            frame_track_ids_set = set()

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                unique_track_ids_set.add(track.track_id)
                frame_track_ids_set.add(track.track_id)
                track_count = len(unique_track_ids_set)
                if str(track.track_id) in track_to_display_mapping_dict:
                    display_id, red, green, blue = track_to_display_mapping_dict[str(track.track_id)]
                else:
                    display_id = track_count
                    red = random.randint(0,255)
                    green = random.randint(0,255)
                    blue = random.randint(0,255)
                    track_to_display_mapping_dict[str(track.track_id)] = (display_id, red, green, blue)

                bbox = track.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(red,green,blue), 2)
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[1]) - 30),(red,green,blue), -1)
                cv2.putText(frame, 'p' + str(display_id), (int(bbox[0]), int(bbox[1]) - 10), 0, 5e-3 * 130, (255,255,255), 2)

            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2)

            cv2.putText(frame, str(track_count),(50, 50),0, 5e-3 * 200, (255,255,255),2)

            if writeVideo_flag: # and not asyncVideo_flag:
                # save a frame
                out.write(frame)
                frame_index = frame_index + 1

            fps_imutils.update()

            fps = (fps + (1./(time.time()-t1))) / 2
            print("FPS = %f"%(fps))

            # Write data to db
            csv_writer.writerow([str(frame_index), str(0), str(0), str(len(frame_track_ids_set)), len(unique_track_ids_set)])

            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

if __name__ == '__main__':
    main(YOLO())
