import os
import sys
import onnx
import onnxruntime

import cv2
import numpy as np
import torch
import torch.nn as nn

class Skeleton:

    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.sess = onnxruntime.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
        self.ori_size = None
        self.offset_x = None
        self.offset_y = None
        self.hand_bbox = None

    def preprocess(self, frame, hand_bbox):
        self.ori_size = frame.shape
        self.offset_x = int(hand_bbox[0])
        self.offset_y = int(hand_bbox[1])
        self.hand_bbox = hand_bbox

        # crop bounding box area and resize image
        frame = self.crop_resize(frame)
        frame = frame.transpose(2,0,1).reshape(1,3,256,256).astype('float32')

        # mean and std
        frame = (frame - 128.) / 256.
        return frame

    def crop_resize(self, frame):
        frame_crop_box = frame[int(self.hand_bbox[1]) : int(self.hand_bbox[3]), int(self.hand_bbox[0]) : int(self.hand_bbox[2])]
        frame_final = cv2.resize(frame_crop_box,(256, 256))
        return frame_final

    def infer(self, frame, hand_bbox):
        input = self.preprocess(frame, hand_bbox)
        output = self.sess.run(['output'], {'input': input})[0].reshape(21,2)
        output = self.postprocess(output)
        return output

    def postprocess(self, output):
        # transfer the coordinates into origianl image
        output[:,0] = output[:,0] * (self.hand_bbox[2].item() - self.hand_bbox[0].item()) + self.offset_x 
        output[:,1] = output[:,1] * (self.hand_bbox[3].item() - self.hand_bbox[1].item()) + self.offset_y

        return output

        
