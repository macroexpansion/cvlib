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

#include "frame_producer.h"
#include "safe_queue.h"
#include "track_rule.h"
#include "rtsp_stream.h"
#include "violations/violation_manager.h"
#include "detect_vehicle.h"
#include <cstring>

#include<zmq.h>
//#include <json/json.h>
#include <ctime>

#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <fstream>

#include <sched.h>
#include <sys/sysinfo.h>

using namespace std;
using namespace cv;

Json::Value cam_config = utils_app::read_config("./config.json");
vector<char*> cam_name_list;
const int n_cam = cam_config["n_cam"].asInt();
int batch_size = n_cam;
vector<config::Config*> app_config_list;
vector<string> result_port;
extern char* g_classes[G_CLASSES_LEN];

SharedQueue<Mat*> gather_detect_vehicle_queue[MAX_CAM];
SharedQueue<DetectBatch*> gather_batch_detect_vehicle_queue;
SharedQueue<EmbedderBatch*> gather_batch_vehicle_embedder_queue;
SharedQueue<track_rule::TrackingBatch*> gather_tracking_queue[MAX_CAM];
SharedQueue<data_type::ResultBatch*> gather_result_queue[MAX_CAM];

VideoCapture* connect_to_server(char* rtsp_address){
    VideoCapture* cap = nullptr;

    while (1){
        cap = new VideoCapture(rtsp_address);
        
        Mat frame; 
        *cap >> frame;

        if (!frame.empty()){
            return cap;
        }
        
        delete cap;
        
        usleep(1000000);
    }
    return nullptr;
}
         
void frame_worker(int cam_id) {
    char* cam_name = cam_name_list[cam_id];
    //string cam_name(cam_name_list[cam_id]);
    cout << "Cam_name: " << cam_name << endl;
    string* rtsp_address = new string(cam_config["rtsp_addresses_list"][cam_name].asString());
    //char* rtsp_address = rtsp_addresses_list[cam_name];
    cout << "Connect to " << *rtsp_address  << endl;

    VideoCapture *cap = connect_to_server((char*)rtsp_address->c_str());
    int count = 0;
    double fps = 10;
    double cum_sum = 0;
    auto start = chrono::steady_clock::now();
    while (cap->isOpened()) {
        if(TESTING == 1) {
            if((long)(cum_sum*1000) < (long)(count/fps*1000000)) {
                usleep((long)(count/fps*1000000) - (long)(cum_sum*1000));
            }
        }
//         clock_t start, end; 
//         start = clock();
        
        Mat *frame = new Mat; 
        // count += 1;
        // cout << "count: " << count << endl;
        *cap >> *frame;         

        if (frame->empty()) {
            cout << "Reconnect " << rtsp_address  << endl;
            
            delete cap;
            delete frame;
            batch_size--;
            cap = connect_to_server((char*)rtsp_address->c_str());
            batch_size++;
            // continue;
            break;
        } 
        if (gather_detect_vehicle_queue[cam_id].size() <= 7) {
            gather_detect_vehicle_queue[cam_id].push_back(frame);
        } else {
            delete frame;
        }
        // usleep(70000);
        auto end = chrono::steady_clock::now();
        // double time_taken = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        
        cum_sum = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        count += 1;
//         end = clock();
//         //Calculating total time taken by the program. 
//         double time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
//         cout << "Time taken by program is : " << fixed  
//             << time_taken << setprecision(5); 
//         cout << " sec " << endl; 
    }
    delete rtsp_address;       
    delete cap;
}

cv::Mat* static_resize(cv::Mat *img, float r) {
    // r = std::min(r, 1.0f);
    // *img = cv::imread("test.jpg");
    // r = 1;
    int unpad_w = r * img->cols;
    int unpad_h = r * img->rows;
    cv::Mat re(unpad_h, unpad_w, CV_32FC3);
    cv::resize(*img, re, re.size());
    cv::Mat* out = new cv::Mat(detect_vehicle::INPUT_W, detect_vehicle::INPUT_H, CV_32FC3, cv::Scalar(114, 114, 114));;
    re.copyTo((*out)(cv::Rect(0, 0, re.cols, re.rows)));
    // out->convertTo((*out), CV_32FC3);
    
    return out;
}

void gather_frame_worker() {
    Mat** frame_batch = new Mat*[n_cam];
    int* readed_cam_id = new int[n_cam];
    int batch_id = 0;
    while(1) {
        
        for(int cam_id = 0; cam_id < n_cam; cam_id++) {
            char* cam_name = cam_name_list[cam_id];
            
            if (gather_detect_vehicle_queue[cam_id].size() > 0) {
                Mat* frame = gather_detect_vehicle_queue[cam_id].pop_front();
                //gather_detect_vehicle_queue[cam_id].pop_front();
                frame_batch[batch_id] = frame;
                readed_cam_id[batch_id] = cam_id;
                batch_id += 1;
                if (batch_id >= n_cam) {
                    break;
                }
            } else {
                usleep(10000);
            }
        }
        if (batch_id >= batch_size) {
            
            if (gather_batch_detect_vehicle_queue.size() <= 3) {
                DetectBatch* detect_batch = new DetectBatch;
                detect_batch->frame_list = frame_batch;
                detect_batch->readed_cam_id = readed_cam_id;
                detect_batch->n = batch_id;
                detect_batch->sized_list = nullptr;

                cv::Mat** sized_list = new cv::Mat*[batch_id];
                float *scales = new float[batch_id];
                int *img_w_list = new int[batch_id];
                int *img_h_list = new int[batch_id];
                for(int i = 0; i < batch_id; i++) {
                    float r = std::min(detect_vehicle::INPUT_W / (frame_batch[i]->cols*1.0), detect_vehicle::INPUT_H / (frame_batch[i]->rows*1.0));
                    cv::Mat resized;
                    cv::Mat* resized_pointer = static_resize(frame_batch[i], r);
                    sized_list[i] = resized_pointer;
                    scales[i] = r;
                    img_w_list[i] = frame_batch[i]->size().width;
                    img_h_list[i] = frame_batch[i]->size().height;
                }
                detect_batch->sized_list = sized_list;
                detect_batch->scales = scales;
                detect_batch->img_w_list = img_w_list;
                detect_batch->img_h_list = img_h_list;
                gather_batch_detect_vehicle_queue.push_back(detect_batch);
            } else {
                for(int i = 0; i < batch_id; i++) {
                    delete frame_batch[i];
                }
                delete[] frame_batch;
                delete[] readed_cam_id;
            }    
            frame_batch = new Mat*[n_cam];
            readed_cam_id = new int[n_cam];
            batch_id = 0;
        }
        
    }
}
// int g_id = 0;
void detect_vehicle_worker(int n_cam) {
    cout << "Run detect vehicle..." << endl;
    const int num_classes = 6;
    detect_vehicle::DetectVehicle detect_model("model/yolox_utvm_2.engine");
   
    vector<config::SideWalk> side_walks_list[MAX_CAM];
    for(int cam_id = 0; cam_id < n_cam; cam_id++) {
        side_walks_list[cam_id] = app_config_list[cam_id]->get_side_walks();
    }
    
    while(1) {
        DetectBatch* vehicle_batch = gather_batch_detect_vehicle_queue.pop_front();
        auto start_infer = chrono::steady_clock::now();
        cv::Mat** sized_list = vehicle_batch->sized_list;
        Mat** frame_batch = vehicle_batch->frame_list;
        int* readed_cam_id = vehicle_batch->readed_cam_id;
        int n = vehicle_batch->n;
        float *scales = vehicle_batch->scales;
        int *img_w_list = vehicle_batch->img_w_list;
        int *img_h_list = vehicle_batch->img_h_list;
        data_type::BoundingBox** total_bboxes_list = new data_type::BoundingBox*[n];
        int *n_percam_list = new int[n];
        int** labels_list = new int*[n];
        // cout << "N: " << n << endl;
        vector<vector<detect_vehicle::Object>> objects = detect_model.predict(sized_list, n, scales, img_w_list, img_h_list);
        
        vector<data_type::BoundingBox*> bboxes_list[n];
        vector<pair<float, int>> scores[n];
        vector<int> label_id_list[n];
        for (int i = 0; i < n; ++i) {
            int cam_id = readed_cam_id[i];
            int frame_width = frame_batch[i]->size().width;
            int frame_height = frame_batch[i]->size().height;
            
            int n_percam = 0;
            vector<data_type::BoundingBox> bboxes_list_tmp;
            vector<int> labels_list_tmp;
            // cout << "N_PERCAM: " << objects[i].size() << endl;
            for(int j = 0; j < objects[i].size(); j++) {
                detect_vehicle::Object object = objects[i][j];
                int w = frame_width*object.rect.width;
                int h = frame_height*object.rect.height;
                float ratio = (float)h/(float)w;
                // cout << h << " " << w << endl;
                // cout << g_classes[label_id] << ": " << ratio << endl;

                if(w >20 && h > 20 && object.rect.width < 0.6 && ratio <= 8.0) {
                    float x_min = object.rect.x;
                    float x_max = object.rect.x + object.rect.width;
                    float y_min = object.rect.y;
                    float y_max = object.rect.y + object.rect.height;
                    float x_center = object.rect.x + object.rect.width/2.0;
                    int label_id = object.label;
                    if(label_id == data_type::TRUCK) {
                        if(x_center < 0.5) {
                            x_center = x_center - object.rect.width/4.0;
                        } else if(x_center > 0.5) {
                            x_center = x_center + object.rect.width/4.0;
                        }
                    }
                    float y_center = object.rect.y + object.rect.height/2.0;

                    K::Point_2 center_point(x_center , y_max);
                    int is_in_side_walk = 0; 
                    for(int p = 0; p < side_walks_list[cam_id].size(); p++) {
                        int n_point = side_walks_list[cam_id][p].points.size();
                        K::Point_2 side_walk_points[n_point];
                        for(int q = 0; q < n_point; q++) {
                            side_walk_points[q] = K::Point_2(side_walks_list[cam_id][p].points[q].x, side_walks_list[cam_id][p].points[q].y);
                        }
                        
                        if(utils_app::check_point_inside_poly(center_point, side_walk_points, side_walk_points + n_point, K())) {
                            is_in_side_walk = 1;
                            break;
                        }
                    }
                    if(is_in_side_walk == 0) {
                        n_percam++;
                        data_type::BoundingBox bbox;
                        bbox.x_min = x_min;
                        bbox.x_max = x_max;
                        bbox.y_min = y_min;
                        bbox.y_max = y_max;

                        bboxes_list_tmp.emplace_back(bbox);
                        labels_list_tmp.emplace_back(label_id);
                    }
                }
            }
            
            total_bboxes_list[i] = new data_type::BoundingBox[n_percam];
            n_percam_list[i] = n_percam;
            labels_list[i] = new int[n_percam];
            
            for(int x = 0; x < n_percam; x++) {
                total_bboxes_list[i][x] = bboxes_list_tmp[x];
                labels_list[i][x] = labels_list_tmp[x];
            }

            // 
        }
        delete[] vehicle_batch->scales;
        delete[] vehicle_batch->img_w_list;
        delete[] vehicle_batch->img_h_list;
        auto end_infer = chrono::steady_clock::now();
        double time_infer = chrono::duration_cast<chrono::milliseconds>(end_infer - start_infer).count();
        // cout << "Total inference " << n <<  " time: " << time_infer << endl; 

        EmbedderBatch* embedder_batch = new EmbedderBatch;
        embedder_batch->frame_list = frame_batch;
        embedder_batch->sized_list = sized_list;
        embedder_batch->n = n;
        embedder_batch->readed_cam_id = readed_cam_id;
        embedder_batch->bboxes_list = total_bboxes_list;
        embedder_batch->labels_list = labels_list;
        embedder_batch->n_percam_list = n_percam_list;
        if(gather_batch_vehicle_embedder_queue.size() <= 3) {
//                 cout << "--Push back: " << embedder_batch->frame_list << endl;
            gather_batch_vehicle_embedder_queue.push_back(embedder_batch);
        } else {
            for(int i = 0; i < embedder_batch->n; i++) {
                delete embedder_batch->frame_list[i];
                delete embedder_batch->sized_list[i];
                delete[] embedder_batch->bboxes_list[i];
                delete[] embedder_batch->labels_list[i];
            }
            
            delete[] embedder_batch->bboxes_list;
            delete[] embedder_batch->n_percam_list;
            delete[] embedder_batch->labels_list;
            delete[] embedder_batch->frame_list;
            delete[] embedder_batch->sized_list;
            delete[] embedder_batch->readed_cam_id;
            delete embedder_batch;
        }

        delete vehicle_batch;

//         auto end = chrono::steady_clock::now();
//         cout << "Elapsed time in milliseconds : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    }
}


cv::Mat* crop_resize_image(cv::Mat* image, std::vector<config::Point> &rec, int new_w, int new_h) {
    float xmin = rec[0].x;
    float ymin = rec[0].y;
    float xmax = rec[1].x;
    float ymax = rec[1].y;
    int frame_width = image->size().width;
    int frame_height = image->size().height;
    
    // cout << "Vehicle embedder: " << xmin << " " << ymin << " " << xmax << " " << ymax << " " << (int)(xmin*frame_width) << " " <<  (int)(ymin*frame_height) << " " << (int)(h*frame_height) << " " << (int)(w*frame_width) << endl;
    float tmp = -1;
    
    
    if (xmin > xmax) {
        tmp = xmin;
        xmin = xmax;
        xmax = tmp;
    }

    if (ymin > ymax) {
        tmp = ymin;
        ymin = ymax;
        ymax = tmp;
    }
    float w = xmax - xmin;
    float h = ymax - ymin;
    
    cv::Rect roi((int)(xmin*frame_width)-1, (int)(ymin*frame_height) - 1, (int)(w*frame_width) - 1, (int)(h*frame_height)-1);

// Crop the full image to that image contained by the rectangle myROI
// Note that this doesn't copy the data
    cv::Mat *cropped_image = new Mat;
    
    cv::resize((*image)(roi), *cropped_image, cv::Size(new_w, new_h));
    return cropped_image;
}
int g_id = 0;
void vehicle_embedder_worker() {
    cout << "Run vehicle embedder..." << endl;
    detect_color::DetectColor color_detecter("model/lightcolor_vit.trt");
    vehicle_embedder::VehicleEmbedder embedder("model/vehicle_embedder.trt");
    
    while(1) {
        int queue_size = gather_batch_vehicle_embedder_queue.size();
        EmbedderBatch* embedder_batch = gather_batch_vehicle_embedder_queue.pop_front();


        int n = embedder_batch->n;
        int total_box = 0;
        
        for(int i = 0; i < n; i++) {
            int n_box = embedder_batch->n_percam_list[i];
            total_box += n_box;           
        }
        auto start = chrono::steady_clock::now();
        cv::Mat gpu_frame_list[total_box];
        int k = 0;
        
        for(int i = 0; i < n; i++) {
            int n_box = embedder_batch->n_percam_list[i];
            int frame_width = embedder_batch->frame_list[i]->size().width;
            int frame_height = embedder_batch->frame_list[i]->size().height;
            for(int j = 0; j < n_box; j++) {
                data_type::BoundingBox bbox = embedder_batch->bboxes_list[i][j];
                float xmin = bbox.x_min;
                float ymin = bbox.y_min;
                float xmax = bbox.x_max;
                float ymax = bbox.y_max;
                float w = xmax - xmin;
                float h = ymax - ymin;
                //cout << xmin << " " << ymin << " " << xmax << " " << ymax << endl;
                cv::Rect roi((int)(xmin*frame_width), (int)(ymin*frame_height), (int)(w*frame_width), (int)(h*frame_height));
                
                cv::Mat cropped_image = (*embedder_batch->frame_list[i])(roi);
                cv::Mat resized;
                //gpu_frame_list[k].upload(cropped_image);
                cv::resize(cropped_image, resized, cv::Size(64, 128));
                resized.convertTo(gpu_frame_list[k], CV_32FC3);
                // string file_name = string("image/vehicle_") + std::to_string(g_id) + string("_") + std::to_string(embedder_batch->readed_cam_id[i])+ string(".jpg");
                // cv::imwrite((char*)file_name.c_str(), gpu_frame_list[k]);
                // g_id++;
                k++;
            }          
        }
        auto end_s = chrono::steady_clock::now();
        float** result_embedder = embedder.predict(gpu_frame_list, total_box);
        auto end = chrono::steady_clock::now();
        //cout << "Embedder infer time in milliseconds : " << chrono::duration_cast<chrono::milliseconds>(end - end_s).count() << " ms" << endl;
        //cout << "Embedder total time in milliseconds : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
        auto start_light_color = chrono::steady_clock::now();
        char** light_colors = new char*[n];
        vector<int> map_batch;
        vector<Mat*> light_colors_batch; 
        for(int i = 0; i < n; i++) {
            int cam_id = embedder_batch->readed_cam_id[i];
            light_colors[i] = color_detecter.classes[1];
            std::vector<config::Point> red = app_config_list[cam_id]->get_red();
            std::vector<config::Point> yellow = app_config_list[cam_id]->get_yellow();
            std::vector<config::Point> green = app_config_list[cam_id]->get_green();
            
            if(red.size() > 0) {            
                cv::Mat *cropped_red_image = crop_resize_image(embedder_batch->frame_list[i], red, 24, 24);
                cv::Mat *cropped_yellow_image = crop_resize_image(embedder_batch->frame_list[i], yellow, 24, 24);
                cv::Mat *cropped_green_image = crop_resize_image(embedder_batch->frame_list[i], green, 24, 24);
                cv::Mat *concatenate_image = new Mat;
                cv::vconcat(*cropped_red_image, *cropped_yellow_image, *concatenate_image);
                cv::vconcat(*concatenate_image, *cropped_green_image, *concatenate_image);
                concatenate_image->convertTo(*concatenate_image, CV_32FC3);
                
                light_colors_batch.emplace_back(concatenate_image);
                delete cropped_red_image;
                delete cropped_yellow_image;
                delete cropped_green_image;
                map_batch.emplace_back(i);
            }
        }
        
        char** result_color = color_detecter.predict(light_colors_batch, light_colors_batch.size());
        auto end_light_color = chrono::steady_clock::now();
        //cout << "Detect light color time in milliseconds : " << chrono::duration_cast<chrono::milliseconds>(end_light_color - start_light_color).count() << " ms" << endl;
        for(int i = 0; i < light_colors_batch.size(); i++) {
            light_colors[map_batch[i]] = result_color[i];
            delete light_colors_batch[i];
        }
        
        int pre_nbox = 0;
        for(int i = 0; i < n; i++) {
            track_rule::TrackingBatch* tracking_batch = new track_rule::TrackingBatch;
            int cam_id = embedder_batch->readed_cam_id[i];
            tracking_batch->cam_id = cam_id;
            tracking_batch->n_box = embedder_batch->n_percam_list[i];
            tracking_batch->frame = embedder_batch->frame_list[i];
            tracking_batch->sized_frame = embedder_batch->sized_list[i];
            tracking_batch->bboxes = embedder_batch->bboxes_list[i];
            tracking_batch->labels = embedder_batch->labels_list[i];
            tracking_batch->light_colors = light_colors[i];
            tracking_batch->vehicle_features = new float*[tracking_batch->n_box];
            
            for(int j = 0; j < tracking_batch->n_box; j++) {
                tracking_batch->vehicle_features[j] = result_embedder[pre_nbox + j];
            }
            pre_nbox += tracking_batch->n_box;
            
            if(gather_tracking_queue[cam_id].size() <= 3) {
                gather_tracking_queue[cam_id].push_back(tracking_batch);
            } else {
                for(int j = 0; j < tracking_batch->n_box; j++) {
                    delete[] tracking_batch->vehicle_features[j];
                }
                delete tracking_batch->frame;
                delete tracking_batch->sized_frame;
                delete[] tracking_batch->bboxes;
                delete[] tracking_batch->labels;
//                     delete tracking_batch->light_colors;
//                     cout << "10-";
                delete[] tracking_batch->vehicle_features;
//                     cout << "11" << endl;
                delete tracking_batch;
            }
        } 
        delete[] result_embedder;
        delete[] result_color;
        delete[] light_colors;
        delete[] embedder_batch->sized_list;
        delete[] embedder_batch->bboxes_list;
        delete[] embedder_batch->n_percam_list;
        delete[] embedder_batch->labels_list;
            

//         cout << "Embedder delete: " << embedder_batch->frame_list << endl;
        delete[] embedder_batch->frame_list;
        delete[] embedder_batch->readed_cam_id;
        delete embedder_batch;
        
//         auto end = chrono::steady_clock::now();
//         cout << "Elapsed time in milliseconds : " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    }
}

void tracking_worker(int cam_id) {
    void *context = zmq_ctx_new();
    void *publisher = zmq_socket(context, ZMQ_PUB);
    string address = "tcp://*:" + result_port[cam_id];
    int bind = zmq_bind(publisher, address.c_str());

    cout << "Run tracking worker... " << cam_id << endl;
    cout << "Bind: " << address << endl;
    
    auto start_fps = chrono::steady_clock::now();
    auto end_fps = chrono::steady_clock::now();
    float time_interval;
    float fps =0;
    float count_fps = -1;
    track_rule::Tracking tracker(cam_id, string(cam_name_list[cam_id]));
    while(1) {
        if(gather_tracking_queue[cam_id].size() > 0) {
            int queue_size = gather_tracking_queue[cam_id].size();
            
            track_rule::TrackingBatch* tracking_batch = gather_tracking_queue[cam_id].pop_front();
            if(count_fps == -1) {
                start_fps = chrono::steady_clock::now();
                count_fps = 0;
                if(TESTING == 1) {
                    time_t init_timestamp = time(0);
                    std::tm* now = std::localtime(&init_timestamp);
                    std::ostringstream str_hour;
                    str_hour << std::setw(2) << std::setfill('0') << now->tm_hour;
                    string hour = str_hour.str();
                    std::ostringstream str_min;
                    str_min << std::setw(2) << std::setfill('0') << now->tm_min;
                    string minute = str_min.str();
                    std::ostringstream str_sec;
                    str_sec << std::setw(2) << std::setfill('0') << now->tm_sec;
                    string second = str_sec.str();
                    string time_sequence = hour + ":" + minute + ":" + second;
                    ofstream myfile ("time.txt");
                    if (myfile.is_open())
                    {
                        myfile << time_sequence;
                        myfile.close();
                    }
                }
            }
            // tracking_batch->light_colors = "no_light";

            int n_box = tracking_batch->n_box;
            data_type::TrackRuleResult* track_result = tracker.update(tracking_batch, fps);
            data_type::ResultBatch* result = new data_type::ResultBatch;
            result->n_box = n_box;
            result->cam_id = cam_id;
            result->fps = fps;
            result->labels = tracking_batch->labels;
            result->bboxes = tracking_batch->bboxes;
            result->track_rule_result = track_result;
            

            if(gather_result_queue[cam_id].size() <= 3) {
                gather_result_queue[cam_id].push_back(result);
            } else {
                delete[] result->labels;
                delete[] result->bboxes;
                //delete result->frame;
                delete result->track_rule_result;
                delete result;
            }
            for(int j = 0; j < tracking_batch->n_box; j++) {
                delete[] tracking_batch->vehicle_features[j];
            }
            //delete tracking_batch->light_colors;
            delete[] tracking_batch->vehicle_features;
            count_fps++;
            end_fps = chrono::steady_clock::now();
            time_interval = chrono::duration_cast<chrono::milliseconds>(end_fps - start_fps).count(); 
            //cout << cam_name_list[cam_id] << ": " << count_fps << " " << time_interval << endl;
            if(time_interval >= 60*1000) {
                if(time_interval == 0) {
                    fps = 0;
                } else {
                    fps = count_fps/time_interval*1000;
                }
                // if(NDEBUG == 0) {
                //     cout << endl;
                //     cout << cam_name_list[cam_id] << ": fps=" << fps << endl;
                // }

                start_fps = chrono::steady_clock::now();
                count_fps = 0;
            }

            //gather_tracking_queue[cam_id].pop_front();
            delete tracking_batch->frame;
            delete tracking_batch;
            // cout << "10" << endl;
        } else {
            usleep(10000);
        }
    }
}

thread** run_frame_producer() {
    int n = cam_name_list.size();
    
    thread** thread_list = new thread*[n];

    for(int i = 0; i < n; i++) {
        //char* cam_name = cam_name_list[i];
        thread_list[i] = new thread(frame_worker, i); 
    }
    
    return thread_list;
}

thread* run_gather_frame() {
    thread* thread_gather_frame = new thread(gather_frame_worker);
    return thread_gather_frame;
}

thread* run_detect_vehicle(int n_cam) {
    thread* thread_detect = new thread(detect_vehicle_worker, n_cam);
    return thread_detect;
}

thread* run_vehicle_embedder() {
    thread* thread_embedder = new thread(vehicle_embedder_worker);
    return thread_embedder;
}

thread** run_tracking() {
    int n = cam_name_list.size();
    
    thread** thread_list = new thread*[n];

    for(int i = 0; i < n; i++) {
        thread_list[i] = new thread(tracking_worker, i); 
    }
    
    return thread_list;
}

int main(int argc, char** argv) {
    cpu_set_t  mask;
    CPU_ZERO(&mask);
   
    CPU_SET(0, &mask);
    CPU_SET(1, &mask);
    CPU_SET(2, &mask);
    CPU_SET(3, &mask);

    CPU_SET(4, &mask);
    CPU_SET(5, &mask);
    CPU_SET(6, &mask);
    CPU_SET(7, &mask);

    CPU_SET(8, &mask);
    CPU_SET(9, &mask);
    CPU_SET(10, &mask);
    CPU_SET(11, &mask);

    CPU_SET(12, &mask);
    CPU_SET(13, &mask);
    CPU_SET(14, &mask);
    CPU_SET(15, &mask);

    CPU_SET(16, &mask);
    CPU_SET(17, &mask);
    CPU_SET(18, &mask);
    CPU_SET(19, &mask);
    CPU_SET(20, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
    cout << n_cam << endl;
    
    for(int i=0; i < n_cam; i++) {
        string* cam_name = new string(cam_config["cam_name_list"][i].asString());
        cam_name_list.emplace_back((char*)cam_name->c_str());
        string* port = new string(cam_config["result_port"][(char*)cam_name->c_str()].asString());
        result_port.emplace_back((char*)port->c_str());
        
        app_config_list.emplace_back(new config::Config(*cam_name));
    }
    int n = cam_name_list.size();
    thread** rtsp_thread_list = rtsp_stream::run_rtsp_server_worker(cam_name_list, n, result_port);
    usleep(3000000);
    
    thread* thread_garbage_collection_routine = track_rule::run_garbage_collection_routine_worker();
    
    
    thread* thread_gather_frame = run_gather_frame();
    thread* thread_detect = run_detect_vehicle(n);
    thread* thread_embedder = run_vehicle_embedder();
    
    thread* thread_send_plate = track_rule::run_send_plate_worker(n_cam);
    thread* thread_recv_plate = track_rule::run_recv_plate_worker(n_cam);
    thread* thread_save_vehicle = track_rule::run_save_vehicle_count_worker(app_config_list);
    thread** thread_save_violation_list = violation_manager::run_save_violation_worker(n_cam);
    thread** tracking_thread_list = run_tracking();
    usleep(3000000);
    thread** producer_thread_list = run_frame_producer();
    
    for(int i = 0; i < n_cam; i++) {
        producer_thread_list[i]->join();
    }
    
    thread_garbage_collection_routine->join();
    thread_gather_frame->join();
    thread_detect->join();
    thread_embedder->join();
    thread_send_plate->join();
    thread_recv_plate->join();
    thread_save_vehicle->join();
    
    for(int i = 0; i < n_cam; i++) {
        tracking_thread_list[i]->join();
        rtsp_thread_list[i]->join();
        thread_save_violation_list[i]->join();
    }

    delete thread_garbage_collection_routine;
    for(int i = 0; i < n_cam; i++) {
        delete producer_thread_list[i];
        delete tracking_thread_list[i];
        delete rtsp_thread_list[i];
        delete thread_save_violation_list[i];
    }
    delete[] producer_thread_list;
    delete thread_gather_frame;
    delete thread_detect;
    delete thread_embedder;
    delete thread_send_plate;
    delete thread_recv_plate;
    delete thread_save_vehicle;
    delete[] thread_save_violation_list;
    delete[] tracking_thread_list;
    delete[] rtsp_thread_list;
    for(int i = 0; i < app_config_list.size(); i++) {
        delete app_config_list[i];
    }
    return 0;
} 
