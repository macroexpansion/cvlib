# -*- coding: utf-8 -*-
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import cv2
import numpy as np
import onnxruntime as rt

from .postprocess import postprocess
vocab = "<ABCDEFGHKLMNPQRSTUVXYZ0123456789#>"
# start symbol <
# end symbol >
token2char = {0: "PAD"}
for i, c in enumerate(vocab):
    token2char[i+1] = c
    
class PlateOCR(object):
    def __init__(self, checkpoint='../model/checkpoint_ocr.onnx'):
        self.sess = rt.InferenceSession(checkpoint)
        self.input_name = self.sess.get_inputs()[0].name

    def post_preprocess(self, prediction, probs):
        result = ''
        for c in prediction:
            if not c.isalpha() and not c.isdigit() and c != '#':
                break
            result += c
        #print(result)
        #print(probs.shape, probs)
        text = postprocess(result, probs)
        #text = text.replace('#', 'D')
        return text
    
    def predict_on_batch(self, img_numpy, img_list):
        batch_size = 256
        #img_numpy = np.array(img_numpy)
        n_img = len(img_list)
        n_left = n_img
        idx_img_list = 0
        while n_left > 0:
            n_samples = min(batch_size, n_left)
            start_index = n_img-n_left
            end_index = start_index + n_samples
            imgs = img_numpy[start_index:end_index]

            # (bs,3,128 ,256)
            ys, probs = self.sess.run(None, {self.input_name:imgs})

            ret = ys
            for i in range(n_samples):
                out = []
                for j in range(len(ret[i])):
                    c = token2char[ret[i][j]]
                    if c == '>':
                        break
                    out.append(c)
                text = "".join(out[1:])
                prob_text = probs[i][1:len(text)+1]

                print(img_list[idx_img_list]['track_id'], text)
                text = self.post_preprocess(text, prob_text)
                text = text.replace("#", "Đ")
                img_list[idx_img_list]['plate_number'] = text
                img_list[idx_img_list]['prob'] = probs[i]
                idx_img_list += 1
            n_left -= n_samples
          
        return img_list