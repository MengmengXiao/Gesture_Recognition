import os
import sys
import numpy as np
import math


def infer_detection_skeleton(frame, args):
    
    # Get hand detection result
    det_bbox = args["det"].infer(frame)

    # Only focus on the first hand
    if det_bbox!=None:
        det_bbox = bbox_suppression(det_bbox, args)
        args["detector_patch"]["box_cache"] = det_bbox
        args["detector_patch"]["frame_num"] = 0
    else:
        # keep the box_pre for 5 frames to avoid missing
        if args["detector_patch"]["frame_num"] < 10:
            if args["detector_patch"]["box_cache"] != None:
                det_bbox = args["detector_patch"]["box_cache"]
                args["detector_patch"]["frame_num"] += 1
        else:
            args["detector_patch"]["box_cache"] = None
            args["detector_patch"]["frame_num"] = 0

    # Get hand landmark result
    if det_bbox!=None:
        lm_res = args["sk"].infer(frame, det_bbox)
    else:
        lm_res = np.zeros((21,2))

    return det_bbox, lm_res


def bbox_suppression(boxes, args):
    box_pre = args["box_pre"]

    if box_pre == None:
        # select box with biggest confidence
        final_box = boxes[0] 
    else:
        # select nearest box
        box_pre_center = ( int((box_pre[2] + box_pre[0])/2), int((box_pre[3] + box_pre[1])/2) )
        
        selected_index = -1
        min_dis = 10000.0

        for index in range(boxes.shape[0]):
            box = boxes[index]
            box_center = ( int((box[2] + box[0])/2), int((box[3] + box[1])/2) ) 
            dis = math.sqrt( (box_center[0] - box_pre_center[0])**2 + (box_center[1] - box_pre_center[1])**2 )
            if dis < min_dis:
                selected_index = index
                min_dis = dis
        final_box = boxes[selected_index]

    args["box_pre"] = final_box

    return final_box

