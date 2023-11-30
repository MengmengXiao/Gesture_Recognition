import os
import sys
import onnx
import onnxruntime

import cv2
import numpy as np
import torch
import torch.nn as nn


class Det:

    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.sess = onnxruntime.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])

        self.input_size = 512
        self.ori_height = None
        self.ori_width = None
        self.offset = None

        self.conf_thres=0.3
        self.nms_thres=0.1

    def preprocess(self, frame):
        # crop and resize image
        frame = self.crop_resize(frame)
        # 1 * 3 * 512 * 512
        frame = frame[:,:,::-1].transpose(2,0,1).reshape(1,3,self.input_size, self.input_size)
        # mean and std
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        frame /= 255.0
        return frame

    def crop_resize(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]
        length = int(1.0 * height)

        self.ori_height = height
        self.ori_width = width
        self.offset = (width - height) / 2

        x = int(width/2)
        y = int(height/2)
        frame_crop = frame[y-int(length/2):y+int(length/2), x-int(length/2):x+int(length/2)]
        # note: we use frame_crop as original input
        frame_final = cv2.resize(frame_crop,(self.input_size, self.input_size))
        return frame_final

    def infer(self, image):
        feature = self.preprocess(image)
        output = self.sess.run(['output'], {'input': feature})[0]
        output = self.postprocess(output)

        if output == [None]:
            return None

        output = self.trans_ori(output)

        return output

    def trans_ori(self, output):
        boxes = output[0]
        boxes[:,0] = boxes[:,0] / 512.0 * self.ori_height + self.offset
        boxes[:,1] = boxes[:,1] / 512.0 * self.ori_height
        boxes[:,2] = boxes[:,2] / 512.0 * self.ori_height + self.offset
        boxes[:,3] = boxes[:,3] / 512.0 * self.ori_height

        boxes[:,0] = torch.min(boxes[:,0], torch.tensor(self.ori_width))
        boxes[:,0] = torch.max(boxes[:,0], torch.tensor(0.0))
        boxes[:,2] = torch.min(boxes[:,2], torch.tensor(self.ori_width))
        boxes[:,2] = torch.max(boxes[:,2], torch.tensor(0.0))
        boxes[:,1] = torch.min(boxes[:,1], torch.tensor(self.ori_height))
        boxes[:,1] = torch.max(boxes[:,1], torch.tensor(0.0))
        boxes[:,3] = torch.min(boxes[:,3], torch.tensor(self.ori_height))
        boxes[:,3] = torch.max(boxes[:,3], torch.tensor(0.0))
        
        return boxes

    def postprocess(self, prediction):
        """
        Removes detections with lower object confidence score than 'conf_thres'
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class)
        """

        prediction = torch.tensor(prediction)
        min_wh = 5  # (pixels) minimum box width and height


        output = [None] * len(prediction)
        for image_i, pred in enumerate(prediction):
            # Filter out confidence scores below threshold
            class_conf, class_pred = pred[:, 4:5].max(1)  # max class_conf, index
            i = (pred[:, 4] > self.conf_thres) & (pred[:, 2] > min_wh) & (pred[:, 3] > min_wh)
            pred2 = pred[i]

            # If none are remaining => process next image
            if len(pred2) == 0:
                continue

            # Select predicted classes
            class_conf = class_conf[i]
            class_pred = class_pred[i].unsqueeze(1).float()

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            # Make bbox a square & expand box
            pred2[:, 2] = torch.max(pred2[:, 2], pred2[:, 3]) * 1.4
            pred2[:, 3] = pred2[:,2] 

            pred2[:, :4] = xywh2xyxy(pred2[:, :4])

            # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
            pred2 = torch.cat((pred2[:, :5], class_conf.unsqueeze(1), class_pred), 1)

            # Get detections sorted by decreasing confidence scores
            pred2 = pred2[(-pred2[:, 4]).argsort()]

            det_max = []
            for c in pred2[:, -1].unique():
                dc = pred2[pred2[:, -1] == c]  # select class c
                dc = dc[:min(len(dc), 100)]  # limit to first 100 boxes
                while len(dc):
                    i = bbox_iou(dc[0], dc) > self.nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            if len(det_max):
                det_max = torch.cat(det_max)  # concatenate
                output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

        return output


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou


