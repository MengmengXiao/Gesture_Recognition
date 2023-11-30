import os
import sys
import onnx
import onnxruntime

import cv2
import numpy as np
import torch
import torch.nn as nn

class Gesture:

    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.sess = onnxruntime.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
        self.input = np.zeros((1,16,21,2))
        #self.input = np.zeros((1,8,21,2))
        print("Please set these vars in ./network/gesture.py")
        self.ori_width = 1280 # frame width
        self.ori_height = 720 # frame height
        self.neg_thresh = 0.9

    def preprocess(self, queue):
        self.input[0] = np.array(queue)
        self.input = self.input.astype(np.float32)
        
        # normalization
        self.input[:,:,:,0] = self.input[:,:,:,0] / self.ori_width
        self.input[:,:,:,1] = self.input[:,:,:,1] / self.ori_height
        self.input[0] -= self.input[0][0][1]
        #self.input[0] -= self.input[0][0][0]


    def infer(self, queue):
        self.preprocess(queue)
        output = self.sess.run(['out'], {'input': self.input})[0]
        output = self.postprocess(output)
        return output


    def postprocess(self, output):
        # softmax
        e_x = np.exp(output[0] - np.max(output[0]))
        output = e_x / e_x.sum(axis=0)

        # debug
        o_list = output.tolist()
        o_list = [str(round(n,3)) for n in o_list]
        str_line = ' -- '.join(o_list)

        # logic
        max_index = np.argmax(output)
        if output[max_index] < self.neg_thresh:
            max_index = 0        

        return max_index, str_line

