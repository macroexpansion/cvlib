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
os.environ["CUDA_VISIBLE_DEVICES"]='1'

from alpr_unconstrained.src.keras_utils import detect_lp_on_batch
import time
import tensorrt as trt
import numpy as np
import sys
# from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger()

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)
        
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, batch_size, image_size, bindings, inputs, outputs, stream):
    (IN_IMAGE_H, IN_IMAGE_W) = image_size
    context.set_binding_shape(0, (batch_size, IN_IMAGE_H, IN_IMAGE_W, 3))
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
  
    context.execute_v2(bindings=bindings)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def detect(context, buffers, image_list, image_size, batch_size):
    IN_IMAGE_H, IN_IMAGE_W = image_size

    ta = time.time()
    inputs, outputs, bindings, stream = buffers
#     print('Length of inputs: ', len(inputs))

    inputs[0].host = np.asarray(image_list).astype(np.float32)

    trt_outputs = do_inference(context, batch_size, image_size, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
#     print(trt_outputs)
#     print('Len of outputs: ', len(trt_outputs[0]))

    tb = time.time()

#     print('-----------------------------------')
#     print('    TRT inference time: %f' % (tb - ta))
#     print('-----------------------------------')
    trt_outputs = trt_outputs[0][0:18*18*8*batch_size].reshape((batch_size, 18, 18, 8))
    return trt_outputs


class alpr_unconstrained_detector():
    def __init__(self):
        self.engine_path = '../model/detect_plate.trt'
        self.image_size = (288, 288)
        self.engine = self.get_engine()
        self.context = self.engine.create_execution_context()
        self.max_batch_size = 256
        self.buffers = allocate_buffers(self.engine, self.max_batch_size)

    def get_engine(self):
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def main(self, image_list, image_size):
        batch_size = len(image_list)
        #print("Len batch: ", batch)
        
        
        return detect(self.context, self.buffers, image_list, image_size, batch_size)

    def predict_on_batch(self, image_src):
        start 	= time.time()
        n_img = len(image_src)
        n_left = n_img
        Yr = []
        while n_left > 0:
            n_samples = min(self.max_batch_size, n_left)
            start_index = n_img-n_left
            end_index = start_index + n_samples
            Yr_batch = self.main(image_src[start_index:end_index], self.image_size)
            for i in range(n_samples):
                Yr.append(Yr_batch[i])
            n_left -= n_samples

        elapsed = time.time() - start
        #print("Inference time of detect plate: ", elapsed)
        return np.array(Yr)


 
