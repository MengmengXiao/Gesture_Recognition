import os
import sys
import cv2


def plot_one_box(img_, box, color=(125,125,0), label="hand", line_thickness=None):

    # Plots one bounding box on image img_
    tl = line_thickness or round(0.002 * max(img_.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img_, c1, c2, color, thickness=tl)

    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img_, c1, c2, color, -1)  # filled
    cv2.putText(img_, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img_


def draw_bd_handpose(img_, landmarks):
    thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55),(255,255,0)]

    cv2.line(img_, (int(landmarks[0][0]), int(landmarks[0][1])),(int(landmarks[1][0]), int(landmarks[1][1])), colors[0], thick)
    cv2.line(img_, (int(landmarks[1][0]), int(landmarks[1][1])),(int(landmarks[2][0]), int(landmarks[2][1])), colors[0], thick)
    cv2.line(img_, (int(landmarks[2][0]), int(landmarks[2][1])),(int(landmarks[3][0]), int(landmarks[3][1])), colors[0], thick)
    cv2.line(img_, (int(landmarks[3][0]), int(landmarks[3][1])),(int(landmarks[4][0]), int(landmarks[4][1])), colors[0], thick)

    cv2.line(img_, (int(landmarks[0][0]), int(landmarks[0][1])),(int(landmarks[5][0]), int(landmarks[5][1])), colors[1], thick)
    cv2.line(img_, (int(landmarks[5][0]), int(landmarks[5][1])),(int(landmarks[6][0]), int(landmarks[6][1])), colors[1], thick)
    cv2.line(img_, (int(landmarks[6][0]), int(landmarks[6][1])),(int(landmarks[7][0]), int(landmarks[7][1])), colors[1], thick)
    cv2.line(img_, (int(landmarks[7][0]), int(landmarks[7][1])),(int(landmarks[8][0]), int(landmarks[8][1])), colors[1], thick)

    cv2.line(img_, (int(landmarks[0][0]), int(landmarks[0][1])),(int(landmarks[9][0]), int(landmarks[9][1])), colors[2], thick)
    cv2.line(img_, (int(landmarks[9][0]), int(landmarks[9][1])),(int(landmarks[10][0]), int(landmarks[10][1])), colors[2], thick)
    cv2.line(img_, (int(landmarks[10][0]), int(landmarks[10][1])),(int(landmarks[11][0]), int(landmarks[11][1])), colors[2], thick)
    cv2.line(img_, (int(landmarks[11][0]), int(landmarks[11][1])),(int(landmarks[12][0]), int(landmarks[12][1])), colors[2], thick)

    cv2.line(img_, (int(landmarks[0][0]), int(landmarks[0][1])),(int(landmarks[13][0]), int(landmarks[13][1])), colors[3], thick)
    cv2.line(img_, (int(landmarks[13][0]), int(landmarks[13][1])),(int(landmarks[14][0]), int(landmarks[14][1])), colors[3], thick)
    cv2.line(img_, (int(landmarks[14][0]), int(landmarks[14][1])),(int(landmarks[15][0]), int(landmarks[15][1])), colors[3], thick)
    cv2.line(img_, (int(landmarks[15][0]), int(landmarks[15][1])),(int(landmarks[16][0]), int(landmarks[16][1])), colors[3], thick)

    cv2.line(img_, (int(landmarks[0][0]), int(landmarks[0][1])),(int(landmarks[17][0]), int(landmarks[17][1])), colors[4], thick)
    cv2.line(img_, (int(landmarks[17][0]), int(landmarks[17][1])),(int(landmarks[18][0]), int(landmarks[18][1])), colors[4], thick)
    cv2.line(img_, (int(landmarks[18][0]), int(landmarks[18][1])),(int(landmarks[19][0]), int(landmarks[19][1])), colors[4], thick)
    cv2.line(img_, (int(landmarks[19][0]), int(landmarks[19][1])),(int(landmarks[20][0]), int(landmarks[20][1])), colors[4], thick)

    for point in landmarks:
        cv2.circle(img_, (int(point[0]), int(point[1])), 3, colors[5], thick)

    return img_

