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

#ifndef DETECT_VEHICLE_H
#define DETECT_VEHICLE_H 1

#include "utils_tensorrt.h"
#include <vector>

namespace detect_vehicle
{
    class DetectVehicle;
    struct Object;
    const int INPUT_H = 416;
    const int INPUT_W = 416;
    const int NC = 6;
    const int NO = NC + 5;
    const float NMS_THRESH = 0.4;
    const float BBOX_CONF_THRESH = 0.4;
    const int FEATURE_SIZE = 3549;
    const int OUTPUT_SIZE = FEATURE_SIZE * NO;
}

struct detect_vehicle::Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};


class detect_vehicle::DetectVehicle {
    private:
        const static int MAX_BATCH_SIZE = 20;
        const static int INPUT_C = 3;
        
        
        float *output = nullptr;
        int input_index = -1;
        int	output_index = -1;
        const char* INPUT_BLOB_NAME = "input";
        const char* OUTPUT_BLOB_NAME = "output";
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IExecutionContext *context;
        Logger logger;
        void* buffers[2];
    public:
        DetectVehicle(char* model_path);
        std::vector<std::vector<detect_vehicle::Object>> predict(cv::Mat** image_src, int n, float scales[], int img_w_list[], int img_h_list[]);
        ~DetectVehicle();
};

#endif 

