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
#include "detect_vehicle.h"
#include "data_type.h"

using namespace std;

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static int generate_grids_and_stride(const int target_size, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid = target_size / stride;
        for (int g1 = 0; g1 < num_grid; g1++)
        {
            for (int g0 = 0; g0 < num_grid; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

static inline float intersection_area(const detect_vehicle::Object& a, const detect_vehicle::Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<detect_vehicle::Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<detect_vehicle::Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<detect_vehicle::Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const detect_vehicle::Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const detect_vehicle::Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            // cout << "Overlap: " << inter_area / union_area << endl;
            // if(b.label == data_type::TRUCK) {
            //     cout << "Overlap: " << inter_area / union_area << " " << nms_threshold << endl;
            // }
            if (inter_area / union_area > nms_threshold) {
                keep = 0;
                break;
            }  
        }

        if (keep)
            picked.push_back(i);
    }
}

// int temp_count = 0;

static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<detect_vehicle::Object>& objects)
{
    const int num_class = 6;

    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * detect_vehicle::NO;

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            
            float box_prob = box_objectness * box_cls_score;
            if (class_idx == data_type::PERSON) {
                prob_threshold = 0.7;
            } 
            if (class_idx == data_type::TRUCK) {
                prob_threshold = 0.9;
            }
            if (box_prob > prob_threshold)
            {
                // if (class_idx == data_type::PERSON) {
                //     cout << "PERSON PROB: " << box_objectness << " " << box_cls_score << endl;
                // } 
                detect_vehicle::Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;
                
                // if (temp_count < 10) {
                //     cout << box_prob << "===";
                //     temp_count++;
                // }
                
                objects.push_back(obj);
            }
            
        } // class loop

    } // point anchor loop
}

static void decode_outputs(float* prob, std::vector<std::vector<detect_vehicle::Object>>& objects, float scales[], int img_w_list[], int img_h_list[], int batch_size) {
    
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(detect_vehicle::INPUT_W, strides, grid_strides);
    for(int cam_i = 0; cam_i < batch_size; cam_i++) {
        std::vector<detect_vehicle::Object> proposals;
        int img_w = img_w_list[cam_i];
        int img_h = img_h_list[cam_i];
        generate_yolox_proposals(grid_strides, prob + cam_i*detect_vehicle::OUTPUT_SIZE,  detect_vehicle::BBOX_CONF_THRESH, proposals);
        
        // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, detect_vehicle::NMS_THRESH);


        int count = picked.size();

        // std::cout << "num of boxes: " << count << std::endl;

        for (int i = 0; i < count; i++)
        {
            detect_vehicle::Object object = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (object.rect.x) / scales[cam_i];
            float y0 = (object.rect.y) / scales[cam_i];
            float x1 = (object.rect.x + object.rect.width) / scales[cam_i];
            float y1 = (object.rect.y + object.rect.height) / scales[cam_i];

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            object.rect.x = x0/img_w;
            object.rect.y = y0/img_h;
            object.rect.width = (x1 - x0)/img_w;
            object.rect.height = (y1 - y0)/img_h;
            objects[cam_i].emplace_back(object);

        }
    } 
}

detect_vehicle::DetectVehicle::DetectVehicle(char* model_path) {
    std::stringstream gie_model_stream;
    gie_model_stream.seekg(0, gie_model_stream.beg);
    std::ifstream cache(model_path);
    gie_model_stream << cache.rdbuf();
    cache.close();
    gie_model_stream.seekg(0, std::ios::end);
    const int model_size = gie_model_stream.tellg();
    gie_model_stream.seekg(0, std::ios::beg);
    void* model_mem = malloc(model_size);
    gie_model_stream.read((char*)model_mem, model_size);
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(this->logger);
    this->engine = runtime->deserializeCudaEngine(model_mem, model_size, nullptr);
    this->context = this->engine->createExecutionContext();
    void* buffers[2];
    this->input_index = this->engine->getBindingIndex(this->INPUT_BLOB_NAME);
	this->output_index = this->engine->getBindingIndex(this->OUTPUT_BLOB_NAME);


    // cudaMalloc(&buffers[inputIndex], BATCH_SIZE * INPUT_H * INPUT_W * INPUT_C * sizeof(float));
    cudaMalloc(&(this->buffers[this->input_index]), this->MAX_BATCH_SIZE * detect_vehicle::INPUT_H * detect_vehicle::INPUT_W * this->INPUT_C * sizeof(float));
    cudaMalloc(&(this->buffers[this->output_index]), this->MAX_BATCH_SIZE * detect_vehicle::OUTPUT_SIZE * sizeof(float));
    this->output = new float[detect_vehicle::OUTPUT_SIZE * this->MAX_BATCH_SIZE]; 
}

vector<vector<detect_vehicle::Object>> detect_vehicle::DetectVehicle::predict(cv::Mat** image_src, int n, float scales[], int img_w_list[], int img_h_list[]) {
    for(int i = 0; i < n; i++) {
        // cv::imwrite("test.jpg", *image_src[i]);
        cudaMemcpy2D((float*)this->buffers[this->input_index] + i*detect_vehicle::INPUT_H * detect_vehicle::INPUT_W * this->INPUT_C, \
            image_src[i]->cols*image_src[i]->elemSize(), image_src[i]->data, image_src[i]->step, \
            image_src[i]->cols*image_src[i]->elemSize(), image_src[i]->rows, cudaMemcpyHostToDevice);
    }
    nvinfer1::Dims4 input_dims = nvinfer1::Dims4(n, detect_vehicle::INPUT_H, detect_vehicle::INPUT_W, this->INPUT_C);
    this->context->setBindingDimensions(0, input_dims);
    this->context->executeV2(this->buffers);

    cudaMemcpy(this->output, this->buffers[this->output_index], n * detect_vehicle::OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // cout << "this->output: " << this->output[0] << " " << this->output[1] << " " << this->output[2] << " " << this->output[3] << " " << this->output[4] << " " << this->output[5]  << " " << this->output[6] << " " << this->output[7] << " " << this->output[8] << " " << this->output[9] << " " << this->output[103]<< endl;
    vector<vector<detect_vehicle::Object>> objects(n);
    decode_outputs(this->output, objects, scales, img_w_list, img_h_list, n);

    return objects;
}       

detect_vehicle::DetectVehicle::~DetectVehicle() {
	cudaFree(this->buffers[this->input_index]);
	cudaFree(this->buffers[this->output_index]);    
    delete[] this->output;
} 
