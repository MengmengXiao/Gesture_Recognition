import os
import sys
import argparse
import cv2
from collections import deque
from network import *
from utils import *


def process_main(frame, queue, args):

    box_res, lm_res = infer_detection_skeleton(frame, args)

    str_line = ""
    if len(queue) == args["time_len"]:
        queue.popleft()
        queue.append(lm_res)
        gesture_result, str_line = args["ges"].infer(queue)
    else:
        queue.append(lm_res)
        gesture_result = 7

        return


    if args["show_box"] and box_res!=None:
        frame = plot_one_box(frame, box_res)

    if args["show_landmarks"] and box_res!=None:
        frame = draw_bd_handpose(frame, lm_res)

    str_line = ""
    res_string = mapping[gesture_result]
    res_string += "  "
    res_string += str_line
    frame = cv2.putText(frame, res_string, (50,50), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255,0,0), 3, cv2.LINE_AA) 

    return frame


if __name__ == "__main__":

    #-------------------
    #  Global variable
    #-------------------
    
    mapping = ["",
               "Throw Up",
               "Throw Down",
               "Throw Right",
               "Throw Left",
               "Zoom Out",
               "Zoom In",
               "Cold Starting...",
                ]

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_camera", type=bool, default=False,
                        help="Use camera captured information as input.")
    parser.add_argument("--video_path", type=str, default="./data/test.mp4",
                        help="Use offline video as input.")
    parser.add_argument("--show_box", type=bool, default=True,
                        help="Draw hand bbox on display.")
    parser.add_argument("--show_landmarks", type=bool, default=True,
                        help="Draw hand landmarks on display.")
    parser.add_argument("--time_len", type=int, default=16,
                        help="Frame number in a classification sequence.")
    parser.add_argument("--save_lm", type=bool, default=False,
                        help="Save landmark results.")
    args = parser.parse_args()

    #-------------------
    #  Main process
    #-------------------

    landmarks_queue = deque()

    if args.with_camera == True:
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(args.video_path)

    model_det = Det("./onnx_model/hand_detection.onnx")
    model_sk = Skeleton("./onnx_model/landmark.onnx")
    model_ges = Gesture("./onnx_model/gesture.onnx")

    args = vars(args)
    args["det"] = model_det
    args["sk"] = model_sk
    args["ges"] = model_ges
    args["box_pre"] = None

    # detector patch
    args["detector_patch"] = {"frame_num":0,
                             "box_cache":None}

    cnt = 0
    while True:
        cnt += 1
        print(cnt)
        ret, frame = capture.read()
        if not ret:
            break
        final_im = process_main(frame, landmarks_queue, args)

        cv2.imshow("camera", final_im)
        if cv2.waitKey(5) == 27:    # push ESC to quit
            break

    print("Done")
