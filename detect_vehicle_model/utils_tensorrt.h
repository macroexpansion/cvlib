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
#ifndef UTILS_TENSORRT_H
#define UTILS_TENSORRT_H 1

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char * msg) override
    {
      std::cout << msg << std::endl;
    }
};


#endif 
