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
#ifndef DATA_TYPE_H
#define DATA_TYPE_H 1

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <ctime>

#define M_PI 3.14159265358979323846
#define NDEBUG 0
#define TESTING 1
namespace data_type 
{
    struct ResultBatch;
    struct BoundingBox;
    struct TrackRuleResult;
    struct TrackInfo;
    struct SeekInfo;
    struct FrameInfo;
    struct Plate;
    struct Node;
    enum VEHICLE_ID{
        BIKE, TRUCK, BICYCLE, CAR, PERSON, VAN
    };
}


struct data_type::Plate {
    std::string number;
    cv::Mat* plate_image;
    cv::Mat* vehicle_image;
};

struct data_type::BoundingBox {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
};

struct data_type::TrackInfo {
    int track_id;
    data_type::BoundingBox bbox;
    int label_id;
};

struct data_type::Node {
    cv::Mat *frame;
    std::unordered_map<int, data_type::TrackInfo> track_map;
    time_t time_stamp;
    int number_use = 0;
    int adj_del_flag = 0;
    data_type::Node* next;
    data_type::Node* prev;
};

struct data_type::SeekInfo {
    float x;
    float y;
    float w;
    float h;
    int label_id;
    char* light_color;
    time_t time;
    int is_send_plate = 0;
};

struct data_type::TrackRuleResult {
    data_type::Node* cur_pointer = nullptr;
    std::vector<data_type::TrackInfo> track_list;
    std::unordered_map<int, std::vector<data_type::SeekInfo>> seek_list;
    int count_vie[6] = {0};
    std::unordered_map<int, std::string> license_plate_set;
};

struct data_type::ResultBatch {
    int cam_id = -1;
    int n_box = 0;
    float fps = 0;
    int* labels=nullptr;
    data_type::BoundingBox* bboxes = nullptr;
    data_type::TrackRuleResult* track_rule_result;    
};



#endif 