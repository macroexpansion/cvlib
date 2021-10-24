'''
/*
 * Created on Thu Dec 17 2020
 *
 * Author Nguyen Van Nam: 0985038799
 * Author Nguyen Tuan Anh: 0961455828
 * Author Vu Minh Quan: 0354092495
 * Author Nguyen Hoang Thuyen: 0386927744
 * Author Tran Manh Tung: 0393370077
 * Author Pham Thi Quynh: 0974627360
 * Author Nguyen Viet Manh: 0975253099
 * Author Nguyen Tien Dat: 0829040166
 * Copyright (c) 2020 Viettel Cyber Space
 */
'''
import numpy as np 
import os
import cv2
import time
import shutil
import json
import queue
import zmq
import sys
import threading
from alpr_unconstrained.Alpr_Unconstrained_Detector import alpr_unconstrained_detector
from alpr_unconstrained.src.utils import im2single
from config import cam_name_list, plate_address, ocr_service_address

import argparse
parser = argparse.ArgumentParser("Detect plate")
parser.add_argument('--port', type=int, default=False, required=True)
parser.add_argument('--core_number', type=str, default=False, required=True)

args = parser.parse_args()



plate_queue = queue.Queue()
def gather_batch_worker():
    plate_in_address = "tcp://localhost:{}".format(args.port)
    print("Connect to address: ", plate_in_address)
    plate_in_socket = zmq.Context().socket(zmq.SUB)
    plate_in_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
    # plate_in_socket.setsockopt(zmq.RCVBUF, 20*1024*1024)
    # plate_in_socket.setsockopt(zmq.RCVHWM, 20)
    plate_in_socket.connect(plate_in_address)
        
    readed_cam_names = []; index_plate_per_cam = []
    recognize_vehicle_image_list = []; recognize_label_list = []; recognize_track_id_list = []
    start = time.time()
    n_images = 0
    while True:  
        try:
            message = plate_in_socket.recv_json()
        except Exception as e: 
            print(e)
            continue
        try:
            image_data_raw = plate_in_socket.recv()
            
            image_data_buf = memoryview(image_data_raw)
            image_data = np.frombuffer(image_data_buf, dtype='uint8')
            vehicle_image = image_data.reshape(tuple(message['shape']))
        except Exception as e: 
            print(e)
            continue
        
        label = message['label']
        track_id = message['track_id']
        cam_name = message['cam_name']
    
        recognize_vehicle_image_list.append(vehicle_image)
        recognize_label_list.append(label)
        recognize_track_id_list.append(track_id)
        index_plate_per_cam.append(cam_name)
        if cam_name not in readed_cam_names:
            readed_cam_names.append(cam_name)
        n_images += 1
        end = time.time()
        dis = end - start
        batch_size = 64

        if dis > 0.1 or n_images >= batch_size:  
            if len(readed_cam_names) > 0:
                image_size = (288, 288)
                image_resized_numpy = []
                index = 0
                for img, label in zip(recognize_vehicle_image_list, recognize_label_list):
                    if label == 'bike' or label == 'car':
                        img = img[img.shape[0]//2:,:,:]
                    else:
                        img = img[img.shape[0]//4:,:,:]
                    #cv2.imwrite("test.jpg", img)
                    i_resized = cv2.resize(img, image_size)
                    i_normalized = i_resized/255.0
                    image_resized_numpy.append(i_normalized)
                    index += 1
                if plate_queue.qsize() >= 6:
                    plate_queue.get()
                plate_queue.put((recognize_vehicle_image_list, recognize_label_list, recognize_track_id_list, index_plate_per_cam, image_resized_numpy, readed_cam_names))
            start = time.time()
            index_plate_per_cam = []; recognize_vehicle_image_list = []; recognize_label_list = []; recognize_track_id_list = []; readed_cam_names = []; n_images = 0

    
def run_gather_batch_worker():
    process = threading.Thread(target=gather_batch_worker, args=())
    process.start()

def get_batch():
    recognize_vehicle_image_list, recognize_label_list, recognize_track_id_list, index_plate_per_cam, image_resized_numpy, readed_cam_names = plate_queue.get()
    return recognize_vehicle_image_list, recognize_label_list, recognize_track_id_list, index_plate_per_cam, image_resized_numpy, readed_cam_names

transfer_queue = queue.Queue()
def transfer_worker():
    socket = zmq.Context().socket(zmq.PUSH)
    # socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
#     socket.setsockopt(zmq.RCVBUF, 20*1024*1024)
#     socket.setsockopt(zmq.RCVHWM, 40)
    socket.connect(ocr_service_address)
    print("Connect to ocr server: ", ocr_service_address)
    while True:
        recognize_vehicle_image_list, recognize_label_list, recognize_track_id_list, index_plate_per_cam, results = transfer_queue.get()
            
        i = 0
        for vehicle_image, label, track_id, cam_name in zip(recognize_vehicle_image_list, recognize_label_list, recognize_track_id_list, index_plate_per_cam):
            result = results[i]
            i += 1
            message_1 = {'dtype': str(vehicle_image.dtype), 'shape': vehicle_image.shape, 'track_id': track_id, 'label': label, 'cam_name': cam_name}
            socket.send_json(message_1)
            socket.send(vehicle_image)
            message_2 = {'dtype': str(result.dtype), 'shape': result.shape}
            socket.send_json(message_2)
            socket.send(result)

def run_transfer_worker():
    process = threading.Thread(target=transfer_worker, args=())
    process.start() 

class DetectPlateService: 
    def __init__(self):
        self.thread = None
        self.count_input = 0
        self.count_result = 0
        self.current_count_input = 0
        self.plate_detector = alpr_unconstrained_detector()
        
    def run(self):
        flag_count = 0
        run_gather_batch_worker()
        run_transfer_worker()

        while True:
            recognize_vehicle_image_list, recognize_label_list, recognize_track_id_list, index_plate_per_cam, image_resized_numpy, readed_cam_names = get_batch()
            s1 = time.time()
            # print(readed_cam_names, len(image_resized_numpy))
            #s2 = time.time()

#             if Debug.log_detect_plate in app_config.debug_mode:
#                 detect_plate_logger.debug("---------------------------")
            results = self.detect_plate(image_resized_numpy)
#             s3 = time.time()
            if transfer_queue.qsize() >= 6:
                transfer_queue.get()
            transfer_queue.put((recognize_vehicle_image_list, recognize_label_list, recognize_track_id_list, index_plate_per_cam, results))
            # print(time.time() - s1)
#             if Debug.log_detect_plate in app_config.debug_mode:
#                 detect_plate_logger.debug("Get batch time: {}s".format(s2 - s1))
#                 detect_plate_logger.debug("Detect plate time: {}s".format(s3 - s2))
#                 detect_plate_logger.debug("Total time: {}s".format(time.time() - s1))
        

    def detect_plate(self, image_resized_numpy): 
        # print(image_resized_numpy[0].shape)
        result = self.plate_detector.predict_on_batch(image_resized_numpy) 
        return result

if __name__ == "__main__":
    os.system("taskset -p -c {} {}".format(args.core_number, os.getpid()))
    detect_plate_service = DetectPlateService()
    detect_plate_service.run()