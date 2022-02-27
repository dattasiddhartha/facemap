# import os
from flask import Flask, request, flash, session, send_from_directory, Response
from flask_bootstrap import Bootstrap
from flask_socketio import SocketIO
from flask.templating import render_template
# import json, argparse
from werkzeug.utils import redirect
# import time
# import numpy as np

import scipy.io
import os
import random
import cv2
import numpy as np

import torch, torchvision
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import (
    Visualizer,
    _PanopticPrediction,
)

secret_key = u'f71b10b68b1bc00019cfc50d6ee817e75d5441bd5db0bd83453b398225cede69'

app = Flask(__name__, static_url_path='')
app.secret_key = secret_key
socketio = SocketIO(app, async_mode='threading')

# configs
fname = '../GPS_Long_Lat_Compass.mat'
commands = ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT'] # Assume they correspond to NSEW
images_used = ['all', [4, 3]][1]
image_folder = '../streets'
face_image = '../test.jpg'
resolution_reduction = 0.5

mat = scipy.io.loadmat(fname)

# Map coordinates to image landmark indices
landmark_indices_fnames = os.listdir(image_folder)
landmark_indices = [int(z[:6]) for z in os.listdir(image_folder)]
landmark_coordinates = [mat['GPS_Compass'][l-1] for l in landmark_indices]

# initialize agent
# init_coordinates = landmark_coordinates[int(random.uniform(0, 1)*len(landmark_coordinates))]

@app.route("/")
def index():
    # init_coordinates = landmark_coordinates[int(random.uniform(0, 1)*len(landmark_coordinates))]
    # np.save('../init_coordinates.npy', init_coordinates)
    # session['init_coordinates'] = 
    session['init_coordinates_idx'] = int(random.uniform(0, 1)*len(landmark_coordinates))
    return render_template("index.html")


# @app.route('/<path:path>')
# def send_js(path):
#     return send_from_directory(path)

def GetImage():
    global img
    (flag, encodedImage) = cv2.imencode(".jpg", img)
    while True:
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
# 
@app.route("/stream")
# @app.route('/stream/', methods=["POST"])
def stream():
    return Response(GetImage(), mimetype = "multipart/x-mixed-replace; boundary=frame")



@socketio.on('device_input')
def device_input(message):
    # Data received
    print(message)
    key = message['keyup']

    if int(key) in [38, 87]: command = commands[0]
    if int(key) in [40, 83]: command = commands[1]
    if int(key) in [39, 68]: command = commands[2]
    if int(key) in [37, 65]: command = commands[3]
    # Test navigation of WASD --- run this each time a new key command is received
    # init_coordinates = np.load('../init_coordinates.npy')
    # print(init_coordinates)
    # print(init_coordinates.keys())
    init_coordinates = landmark_coordinates[session['init_coordinates_idx']]
    dist_coordinates = landmark_coordinates - init_coordinates
    dist_coordinates = [(i, j) for (i, j) in enumerate(dist_coordinates)]

    if command in commands[0:2]: 
        dist_coordinates = sorted(dist_coordinates, 
                        key=lambda x: abs(x[1][1]))
    if command in commands[2:]: 
        dist_coordinates = sorted(dist_coordinates, 
                        key=lambda x: abs(x[1][0]))

    for idx_next, coordinates_next in dist_coordinates:
        if (coordinates_next[0] != 0 and coordinates_next[1] != 0):
            if command == commands[0]:
            # lat increaese, long constant
                if coordinates_next[0] > 0: 
                    session['init_coordinates'] = landmark_coordinates[idx_next]
                    break
            if command == commands[1]:
            # lat decreaese, long constant
                if coordinates_next[0] < 0: 
                    session['init_coordinates'] = landmark_coordinates[idx_next]
                    break
            if command == commands[2]:
            # lat constant, long increase
                if coordinates_next[1] > 0: 
                    session['init_coordinates'] = landmark_coordinates[idx_next]
                    break
            if command == commands[3]:
            # lat constant, long decrease
                if coordinates_next[1] < 0: 
                    session['init_coordinates'] = landmark_coordinates[idx_next]
                    break

    image_next = landmark_indices[idx_next]
    # required images
    if images_used == 'all': image_fnames_next = [''.join([str(0)]*(6-len(str(image_next)))+[str(image_next)])+f"_{x}.jpg" for x in range(6)]
    if images_used != 'all': image_fnames_next = [''.join([str(0)]*(6-len(str(image_next)))+[str(image_next)])+f"_{x}.jpg" for x in images_used]
    # print(image_fnames_next)

    # Test style transfer
    frame_0, frame_1 = cv2.imread(os.path.join(image_folder, image_fnames_next[0])), cv2.imread(os.path.join(image_folder, image_fnames_next[1]))
    frame = np.concatenate((frame_0, frame_1), axis=1)
    webcam = cv2.imread(face_image)
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier('../assets/models/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(webcam, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags = cv2.cv.CV_HAAR_SCALE_IMAGE,
        flags = cv2.CASCADE_SCALE_IMAGE,
    )
    # get coordinates of biggest face
    area = 0; face_coords = []
    for (x, y, w, h) in faces:
        if w*h > area: area = w*h; face_coords = [x, y, x+w, y+h]
    print(face_coords)
    crop_face = webcam[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]

    """
    Style transfer operations:
    1. Perform semantic segmentation to identify buildings and sky and cars and road -- extract their coordinates -- wrap a face around these objects with some small random perturbations
    2. Place cropped or cutmix-ed face on image first, then place warped segmentations on top

    """
    # frame = cv2.resize(frame, (int(frame.shape[1]*resolution_reduction), int(frame.shape[0]*resolution_reduction)), Image.ANTIALIAS)
    # Inference with a panoptic segmentation model
    config_model_name = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_model_name)
    predictor = DefaultPredictor(cfg)
    panoptic_seg, segments_info = predictor(frame)["panoptic_seg"]
    """Debug segmentation
    v = Visualizer(frame, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    im_out = out.get_image()
    cv2.imwrite('check2.jpg', im_out)
    """

    def general(frame):
        hh, ww = frame.shape[:2]
        ht, wd = crop_face.shape[:2]
        maxwh = max(hh, ww)
        minwh = min(ht,wd)
        scale = maxwh/minwh
        pattern_enlarge = cv2.resize(crop_face, dsize=(0,0), fx=scale, fy=scale)

        pattern_enlarge = pattern_enlarge[0:hh, 0:ww]

        # left_x = int(hh/2-random.uniform(0,1)*(hh/2))
        # right_x = int(left_x+hh)
        # left_y = int(ww/2-random.uniform(0,1)*(ww/2))
        # right_y = int(left_y+ww)
        # pattern_enlarge = pattern_enlarge[max(0, left_x):min(hh, right_x), max(0, left_y):min(ww, right_y)]
        # pattern_enlarge = pattern_enlarge[min(0, int(hh/2-hh/4)):max(hh, int(hh/2+hh/4)), min(0, int(ww/2-ww/4)):max(ww, int(ww/2+ww/4))]
        # print(pattern_enlarge.shape)

        # frame[mask,:] = pattern_enlarge[mask,:]
        frame = frame*0.8 + pattern_enlarge*0.2

        # frame = cv2.add(frame, pattern_enlarge)
        # frame = cv2.addWeighted(frame, 1., pattern_enlarge, 0., 0)
        return frame

    def building(frame):
        if label == 'building':
            hh, ww = frame.shape[:2]
            ht, wd = crop_face.shape[:2]
            maxwh = max(hh, ww)
            minwh = min(ht,wd)
            scale = maxwh/minwh
            pattern_enlarge = cv2.resize(crop_face, dsize=(0,0), fx=scale, fy=scale)
            pattern_enlarge = pattern_enlarge[min(0, int(hh/2-hh/4)):max(hh, int(hh/2+hh/4)), min(0, int(ww/2-ww/4)):max(ww, int(ww/2+ww/4))]
            # print(pattern_enlarge.shape)

            pattern_enlarge[~mask,:] = [0,0,0]
            # frame[mask,:] = pattern_enlarge[mask,:]
            frame[mask,:] = frame[mask,:]*0.5 + pattern_enlarge[mask,:]*0.5

            # frame = cv2.add(frame, pattern_enlarge)
            # frame = cv2.addWeighted(frame, 1., pattern_enlarge, 0., 0)

    def sky(frame):
        if label == 'sky':
            hh, ww = frame.shape[:2]
            ht, wd = crop_face.shape[:2]
            maxwh = max(hh, ww)
            minwh = min(ht,wd)
            scale = maxwh/minwh
            pattern_enlarge = cv2.resize(crop_face, dsize=(0,0), fx=scale, fy=scale)
            pattern_enlarge = pattern_enlarge[min(0, int(hh/2-hh/4)):max(hh, int(hh/2+hh/4)), min(0, int(ww/2-ww/4)):max(ww, int(ww/2+ww/4))]
            # print(pattern_enlarge.shape)

            pattern_enlarge[~mask,:] = [0,0,0]
            # frame[mask,:] = pattern_enlarge[mask,:]
            frame[mask,:] = frame[mask,:]*0.6 + pattern_enlarge[mask,:]*0.4


            # frame = cv2.add(frame, pattern_enlarge)
            # frame = cv2.addWeighted(frame, 1., pattern_enlarge, 0., 0)

    def road(frame):
        if label == 'road':
            hh, ww = frame.shape[:2]
            ht, wd = crop_face.shape[:2]
            maxwh = max(hh, ww)
            minwh = min(ht,wd)
            scale = maxwh/minwh
            pattern_enlarge = cv2.resize(crop_face, dsize=(0,0), fx=scale, fy=scale)
            pattern_enlarge = pattern_enlarge[min(0, int(hh/2-hh/4)):max(hh, int(hh/2+hh/4)), min(0, int(ww/2-ww/4)):max(ww, int(ww/2+ww/4))]
            # print(pattern_enlarge.shape)
            # pattern_enlarge = pattern_enlarge[0:hh, 0:ww]
            

            pattern_enlarge[~mask,:] = [0,0,0]
            # frame[mask,:] = pattern_enlarge[mask,:]
            frame[mask,:] = frame[mask,:]*0.2 + pattern_enlarge[mask,:]*0.8


            # frame = cv2.add(frame, pattern_enlarge)
            # frame = cv2.addWeighted(frame, 1., pattern_enlarge, 0., 0)

    def vehicle(frame):
        if label in ['car', 'truck']:
            hh, ww = frame.shape[:2]
            ht, wd = crop_face.shape[:2]
            maxwh = max(hh, ww)
            minwh = min(ht,wd)
            scale = maxwh/minwh
            pattern_enlarge = cv2.resize(crop_face, dsize=(0,0), fx=scale, fy=scale)
            pattern_enlarge = pattern_enlarge[min(0, int(hh/2-hh/4)):max(hh, int(hh/2+hh/4)), min(0, int(ww/2-ww/4)):max(ww, int(ww/2+ww/4))]
            # print(pattern_enlarge.shape)

            pattern_enlarge[~mask,:] = [0,0,0]
            # frame[mask,:] = pattern_enlarge[mask,:]
            frame[mask,:] = frame[mask,:]*0.5 + pattern_enlarge[mask,:]*0.5

            # frame = cv2.add(frame, pattern_enlarge)
            # frame = cv2.addWeighted(frame, 1., pattern_enlarge, 0., 0)

    frame = general(frame)

    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes
    pred = _PanopticPrediction(panoptic_seg, segments_info)
    for mask, sinfo in pred.semantic_masks():
        category_idx = sinfo["category_id"]
        label = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_idx]
        print(category_idx, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[category_idx], mask)

        # building(frame)
        # sky(frame)
        # road(frame)
        # vehicle(frame)

        def attempt():
            coinflip = random.uniform(0, 1)
            # coinflip = 0

            if coinflip > 0.4:
                hh, ww = frame.shape[:2]
                ht, wd = crop_face.shape[:2]
                maxwh = max(hh, ww)
                minwh = min(ht,wd)
                scale = maxwh/minwh
                magnitude = min(1, random.uniform(0,1)*5)
                pattern_enlarge = cv2.resize(crop_face, dsize=(0,0), fx=scale*magnitude, fy=scale*magnitude)
                # pattern_enlarge = pattern_enlarge[min(0, int(hh/2-hh/4)):max(hh, int(hh/2+hh/4)), min(0, int(ww/2-ww/4)):max(ww, int(ww/2+ww/4))]
                # print(pattern_enlarge.shape)
                left_y = int(random.uniform(0,1) * (pattern_enlarge.shape[0]-hh))
                left_x = int(random.uniform(0,1) * (pattern_enlarge.shape[1]-ww))
                print(pattern_enlarge.shape, hh, ww)
                pattern_enlarge = pattern_enlarge[left_y:left_y+hh, left_x:left_x+ww]
                print(pattern_enlarge.shape, hh, ww)
                pattern_enlarge[~mask,:] = [0,0,0]
                frame[mask,:] = frame[mask,:]*0.2 + pattern_enlarge[mask,:]*0.8
            
            if coinflip <= 0.4:
                pattern_enlarge = crop_face.copy()
                for _ in range(5):
                    pattern_enlarge = np.concatenate((pattern_enlarge, pattern_enlarge), axis=1)
                    # print(_)
                for _ in range(5):
                    pattern_enlarge = np.concatenate((pattern_enlarge, pattern_enlarge), axis=0)
                    # print(_)
                hh, ww = frame.shape[:2]
                ht, wd = pattern_enlarge.shape[:2]
                maxwh = max(hh, ww)
                minwh = min(ht,wd)
                scale = maxwh/minwh
                magnitude = min(1, random.uniform(0,1)*5)
                pattern_enlarge = cv2.resize(pattern_enlarge, dsize=(0,0), fx=scale*magnitude, fy=scale*magnitude)
                # pattern_enlarge = pattern_enlarge[min(0, int(hh/2-hh/4)):max(hh, int(hh/2+hh/4)), min(0, int(ww/2-ww/4)):max(ww, int(ww/2+ww/4))]
                # print(pattern_enlarge.shape)
                left_y = int(random.uniform(0,1) * (pattern_enlarge.shape[0]-hh))
                left_x = int(random.uniform(0,1) * (pattern_enlarge.shape[1]-ww))
                pattern_enlarge = pattern_enlarge[left_y:left_y+hh, left_x:left_x+ww]
                print(pattern_enlarge.shape)
                pattern_enlarge[~mask,:] = [0,0,0]
                frame[mask,:] = frame[mask,:]*0.8 + pattern_enlarge[mask,:]*0.2
        try:
            attempt()
        except:
            try:
                attempt()
            except: 
                continue

    for mask, sinfo in pred.instance_masks():
        category_idx = sinfo["category_id"]
        label = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[category_idx]
        print(category_idx, label, mask)

        # building(frame)
        # sky(frame)
        # road(frame)
        # vehicle(frame)

        def attempt():
            coinflip = random.uniform(0, 1)
            # coinflip = 0

            if coinflip > 0.4:
                hh, ww = frame.shape[:2]
                ht, wd = crop_face.shape[:2]
                maxwh = max(hh, ww)
                minwh = min(ht,wd)
                scale = maxwh/minwh
                magnitude = min(1, random.uniform(0,1)*5)
                pattern_enlarge = cv2.resize(crop_face, dsize=(0,0), fx=scale*magnitude, fy=scale*magnitude)
                # pattern_enlarge = pattern_enlarge[min(0, int(hh/2-hh/4)):max(hh, int(hh/2+hh/4)), min(0, int(ww/2-ww/4)):max(ww, int(ww/2+ww/4))]
                # print(pattern_enlarge.shape)
                left_y = int(random.uniform(0,1) * (pattern_enlarge.shape[0]-hh))
                left_x = int(random.uniform(0,1) * (pattern_enlarge.shape[1]-ww))
                print(pattern_enlarge.shape, hh, ww)
                pattern_enlarge = pattern_enlarge[left_y:left_y+hh, left_x:left_x+ww]
                print(pattern_enlarge.shape, hh, ww)
                pattern_enlarge[~mask,:] = [0,0,0]
                frame[mask,:] = frame[mask,:]*0.5 + pattern_enlarge[mask,:]*0.5
            
            if coinflip <= 0.4:
                pattern_enlarge = crop_face.copy()
                for _ in range(5):
                    pattern_enlarge = np.concatenate((pattern_enlarge, pattern_enlarge), axis=1)
                    # print(_)
                for _ in range(5):
                    pattern_enlarge = np.concatenate((pattern_enlarge, pattern_enlarge), axis=0)
                    # print(_)
                hh, ww = frame.shape[:2]
                ht, wd = pattern_enlarge.shape[:2]
                maxwh = max(hh, ww)
                minwh = min(ht,wd)
                scale = maxwh/minwh
                magnitude = min(1, random.uniform(0,1)*5)
                pattern_enlarge = cv2.resize(pattern_enlarge, dsize=(0,0), fx=scale*magnitude, fy=scale*magnitude)
                # pattern_enlarge = pattern_enlarge[min(0, int(hh/2-hh/4)):max(hh, int(hh/2+hh/4)), min(0, int(ww/2-ww/4)):max(ww, int(ww/2+ww/4))]
                # print(pattern_enlarge.shape)
                left_y = int(random.uniform(0,1) * (pattern_enlarge.shape[0]-hh))
                left_x = int(random.uniform(0,1) * (pattern_enlarge.shape[1]-ww))
                pattern_enlarge = pattern_enlarge[left_y:left_y+hh, left_x:left_x+ww]
                print(pattern_enlarge.shape)
                pattern_enlarge[~mask,:] = [0,0,0]
                frame[mask,:] = frame[mask,:]*0.8 + pattern_enlarge[mask,:]*0.2
        try:
            attempt()
        except:
            try:
                attempt()
            except: 
                continue

    # cv2.imwrite('../host_image.jpg', frame)
    cv2.imwrite('static/img/host_image.jpg', frame)






if __name__ == '__main__':
    img = cv2.imread('static/img/host_image.jpg', 0)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) # threaded=True is the default for the Flask development web server. No need to set anything. 
    # app.run()