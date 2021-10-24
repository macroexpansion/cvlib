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
import os
import numpy as np 
import cv2
import time
import shutil
import json
import queue
import zmq
import sys
import threading
import multiprocessing
from attention_ocr.plate_ocr import PlateOCR 
from attention_ocr.preprocess import preprocess
from alpr_unconstrained.src.keras_utils import decode_predict, reconstruct
from config import cam_name_list, plate_address, ocr_service_address
import copy

            
def save_plate(recognize_number_list):
    date_time = time.localtime(time.time())
    hour = int(date_time.tm_hour)

#         car_vehicle_image_folder_path = 'test/vehicle_image/nighttime/car'
#         os.makedirs(car_vehicle_image_folder_path, exist_ok=True)
#         moto_vehicle_image_folder_path = 'test/vehicle_image/nighttime/moto'
#         os.makedirs(moto_vehicle_image_folder_path, exist_ok=True)
        
    for img_dict in recognize_number_list:
        vehicle_image = img_dict['vehicle_image']
        plate_shape = img_dict['plate_shape']
        plate_image = img_dict['plate_image']
        plate_number = img_dict['plate_number']
        vehicle_label = img_dict['vehicle_label']
        track_id = img_dict['track_id']
        cam_name = img_dict['cam_name']
#         if plate_number[:2] == '26':
# #             if 'B' in plate_number or 'D' in plate_number or 'H' in plate_number or 'K' in plate_number or 'R' in plate_number or '#' in plate_number or 'R' in plate_number or 'E' in plate_number or 'F' in plate_number or 'G' in plate_number or 'L' in plate_number or 'M' in plate_number or 'N' in plate_number or 'P' in plate_number or 'Q' in plate_number or 'S' in plate_number or 'T' in plate_number or 'U' in plate_number or 'V' in plate_number or 'X' in plate_number or 'Y' in plate_number or 'Z' in plate_number:
# #                 if plate_shape == 2:
# #                     plate_shape_character = '@'
# #                 elif plate_shape == 1:
# #                     plate_shape_character = ''
# #                 if len(plate_number) > 0:
# #                     cv2.imwrite(plate_folder_path + "/{}-{}.png".format(plate_number + plate_shape_character, "frame"), vehicle_image)
# #                     cv2.imwrite(plate_folder_path + "/{}.png".format(plate_number + plate_shape_character), plate_image)
#             pass
#         else:
        if 6 <= hour and hour <= 18:
            car_plate_folder_path = '../plate_data_2511/' + cam_name + '/daytime/car'
            os.makedirs(car_plate_folder_path, exist_ok=True)
            moto_plate_folder_path = '../plate_data_2511/' + cam_name + '/daytime/moto'
            os.makedirs(moto_plate_folder_path, exist_ok=True)
        else:
            car_plate_folder_path = '../plate_data_2511/' + cam_name + '/nighttime/car'
            os.makedirs(car_plate_folder_path, exist_ok=True)
            moto_plate_folder_path = '../plate_data_2511/' + cam_name + '/nighttime/moto'
            os.makedirs(moto_plate_folder_path, exist_ok=True)
        if plate_shape == 2:
            plate_shape_character = '@'
        elif plate_shape == 1:
            plate_shape_character = ''
#         if vehicle_label == 'moto':
#             vehicle_image_folder_path = moto_vehicle_image_folder_path
#         else:
#             vehicle_image_folder_path = car_vehicle_image_folder_path
        
        if len(plate_number) > 0:                     
            if vehicle_label == 'moto':
                plate_folder_path = moto_plate_folder_path
            else:
                plate_folder_path = car_plate_folder_path
            cv2.imwrite(plate_folder_path + "/{}.png".format(plate_number + plate_shape_character), plate_image)
            cv2.imwrite(plate_folder_path + "/{}_frame.png".format(plate_number + plate_shape_character), vehicle_image)


fm_id = 0
lp_id = 0
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_32F).var()

def filter_plate(plate_img, plate_shape):
    global fm_id
    h, w, _ = plate_img.shape
        
    if plate_shape == 2:
        # check min plate size
        if h < 19 or w < 19:
#             if Debug.save_plate in app_config.debug_mode:
#                 cv2.imwrite(os.path.join('blur_plate', 'filter_' + str(fm_id) + '.png'), plate_img)
            fm_id += 1
            return None
    elif plate_shape == 1:
        # check min plate size
        if h < 15 or w < 15:
#             if Debug.save_plate in app_config.debug_mode:
#                 cv2.imwrite(os.path.join('blur_plate', 'filter_' + str(fm_id) + '.png'), plate_img)
            fm_id += 1
            return None
        
    if w < h:
#         if Debug.save_plate in app_config.debug_mode:
#             cv2.imwrite(os.path.join('blur_plate', 'filter_' + str(fm_id) + '.png'), plate_img)
        fm_id += 1
        return None
    
    # fm = variance_of_laplacian(plate_img)

    # if fm < 400:
    #     #cv2.imwrite(os.path.join('blur_plate', 'blur_' + str(fm_id) + '.png'), plate_img)
    #     fm_id += 1
    #     return None
    return plate_img

plate_queue = queue.Queue()
def receive_worker(plate_queue):
    global track_id_have_lp
    global lp_id
#     socket = zmq.Context().socket(zmq.SUB)
#     socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
# #     socket.setsockopt(zmq.RCVBUF, 20*1024*1024)
# #     socket.setsockopt(zmq.RCVHWM, 40)
#     socket.connect(ocr_service_address)
    socket = zmq.Context().socket(zmq.PULL)
#     socket.setsockopt(zmq.SNDBUF, 20*1024*1024)
#     socket.setsockopt(zmq.SNDHWM, 20)
    socket.bind(ocr_service_address)
    print("Init server: ", ocr_service_address)
    image_size = (288, 288)
    threshold = 0.1
    recognize_vehicle_image_list = []
    results = []
    recognize_label_list, recognize_track_id_list, index_plate_per_cam = [], [], []
    n_images = 0
    start = time.time()
    while True:
        try:
            message_1 = socket.recv_json()
            label = message_1['label']
            track_id = message_1['track_id']
            cam_name = message_1['cam_name']
        except Exception as e: 
            continue
        try:
            image_data_raw = socket.recv()
            image_data_buf = memoryview(image_data_raw)
            image_data = np.frombuffer(image_data_buf, dtype=message_1['dtype'])
            vehicle_image = image_data.reshape(message_1['shape'])
        except Exception as e: 
            continue
            
        try:
            message_2 = socket.recv_json()
        except Exception as e: 
            continue
        try:
            result_data_raw = socket.recv()
            result_data_buf = memoryview(result_data_raw)
            result_data = np.frombuffer(result_data_buf, dtype=message_2['dtype'])
            result = result_data.reshape(message_2['shape']).copy()
        except Exception as e: 
            continue
        

        recognize_vehicle_image_list.append(vehicle_image)
        recognize_label_list.append(label)
        recognize_track_id_list.append(track_id)
        index_plate_per_cam.append(cam_name)
        results.append(result)
        n_images += 1
        end = time.time()
        dis = end - start
        batch_size = 64
        if dis > 0.1 or n_images >= batch_size:  
            recognize_number_list = []
            n_plate = 0
            for track_id in recognize_track_id_list:
                n_plate += 1
            i = 0; img_numpy = np.zeros((n_plate, 128,256,3), dtype=np.float32)
            k = 0
            for vehicle_image, vehicle_label, track_id, cam_name in zip(recognize_vehicle_image_list, recognize_label_list, recognize_track_id_list, index_plate_per_cam):
                if vehicle_label == 'bike' or vehicle_label == 'car':
                    new_vehicle_image = vehicle_image[vehicle_image.shape[0]//2:,:,:]
                else:
                    new_vehicle_image = vehicle_image[vehicle_image.shape[0]//4:,:,:]
                h_vehicle = new_vehicle_image.shape[0]     
                final_labels = decode_predict(results[i], (image_size[0], image_size[1], 3), threshold)
                if len(final_labels) > 0: 
                    Ilp, point_plate_label = reconstruct(new_vehicle_image, final_labels)
                    # print("Track id: ", track_id, " have license plate.")
                    # cv2.imwrite(os.path.join('images', str(track_id) + "-" + str(lp_id) + '.png'), Ilp)
                    # lp_id += 1
                    point_plate = point_plate_label.pts
                    is_filter = False
                    if min(point_plate[0]) <= 0 or max(point_plate[0]) >= 1:
                        is_filter = True
                    if min(point_plate[1]) <= 0 or max(point_plate[1]) >= 1:
                        is_filter = True

                    if (h_vehicle - max(point_plate[1])*h_vehicle) > 5 and is_filter == False:
                        #cv2.imwrite("plate_tmp.png", Ilp)
                        #Ilp = cv2.imread("plate_tmp.png")
                        if vehicle_label == 'moto':
                            plate_shape = 2
                        else:
                            h, w, _ = Ilp.shape
                            if h*2.5 <= w:
                                plate_shape = 1
                            else:
                                plate_shape = 2
                        Ilp = filter_plate(Ilp, plate_shape)
                        if Ilp is not None:
                            recognize_number_list.append({'plate_number': '', 'id': i, 'vehicle_image': vehicle_image, 'track_id': track_id, 'cam_name': cam_name, 'plate_image': Ilp, 'plate_shape': plate_shape, 'vehicle_label': vehicle_label, 
                                'point_plate': point_plate, 'vehicle_label': vehicle_label})
                            preprocess_img = preprocess(Ilp, plate_shape)
                            img_numpy[k] = preprocess_img
                            k += 1
                i += 1
                #img_numpy = img_numpy/255.0
                #img_numpy = np.moveaxis(img_numpy, 3, 1)
                if len(recognize_number_list) > 0:
                    if plate_queue.qsize() >= 12:
                         plate_queue.get()
                    plate_queue.put((img_numpy, recognize_number_list))
            start = time.time()
            recognize_vehicle_image_list = []
            recognize_label_list, recognize_track_id_list, index_plate_per_cam = [], [], []
            results = []
            n_images = 0

def run_gather_batch_worker():
    process = threading.Thread(target=receive_worker, args=(plate_queue,))
    process.start()

def get_batch():
    img_numpy, recognize_number_list = plate_queue.get()
    return img_numpy, recognize_number_list   

transfer_queue = queue.Queue()

def transfer_worker():
    plate_out_socket = zmq.Context().socket(zmq.PUB)

    plate_out_socket.bind("tcp://*:3001")
    while True:
        recognize_number_list = transfer_queue.get()
        # print("Number plate: ", len(recognize_number_list))
        for i in range(len(recognize_number_list)):
            if(len(recognize_number_list[i]['plate_number']) > 0):
                cam_name = recognize_number_list[i]['cam_name']
                vehicle_image = recognize_number_list[i]['vehicle_image']
                point_plate = recognize_number_list[i]['point_plate']
                vehicle_label = recognize_number_list[i]['vehicle_label']
                plate_image = recognize_number_list[i]['plate_image']
                plate_h, plate_w, _ = plate_image.shape
                vehicle_h, vehicle_w, _ = vehicle_image.shape
                
                message = {'plate_number': recognize_number_list[i]['plate_number'], 'track_id': recognize_number_list[i]['track_id'],
                    'point_plate': [list(point_plate[0]), list(point_plate[1])], 'vehicle_label': vehicle_label, 'plate_w': plate_w, 
                    'plate_h': plate_h, 'vehicle_h': vehicle_h, 'vehicle_w': vehicle_w, 'cam_name': cam_name}
                # print(message)
                plate_out_socket.send_json(message)
                # message_json = json.dumps(message)
                # plate_out_socket.send(b'Hello')
                # plate_out_socket.send(b'World')
                # plate_out_socket.send(b'Quan')

                plate_out_socket.send(plate_image)
                plate_out_socket.send(vehicle_image)
        # print(" ")
def run_transfer_worker():
    process = threading.Thread(target=transfer_worker, args=())
    process.start() 

class PlateOcrService: 
    def __init__(self):
        self.thread = None
        self.count_input = 0
        self.count_result = 0
        self.current_count_input = 0
        self.plate_ocr = PlateOCR() 
        
        self.time_live = 2000
        
    def run(self):
        flag_count = 0
        run_gather_batch_worker()
        run_transfer_worker()

        while True:
            s1 = time.time()
            img_numpy, recognize_number_list = get_batch()
            # print("Number recognize: ", len(recognize_number_list))
            
            s2 = time.time()
            recognize_number_list = self.plate_ocr.predict_on_batch(img_numpy, recognize_number_list)
            s3 = time.time()

            if False:
                save_plate(recognize_number_list)
            if transfer_queue.qsize() >= 32:
                transfer_queue.get()
            transfer_queue.put(recognize_number_list)
            # print("OCR time: {}s".format(s3 - s2))

            

if __name__ == "__main__":
    os.system("taskset -p -c 23 %d" % os.getpid())
    plate_ocr_service = PlateOcrService()
    plate_ocr_service.run() 