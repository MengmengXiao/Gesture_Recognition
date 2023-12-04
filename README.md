# Gesture_Recognition
Gesture Recognition with deep learning models. There are 6 class: throw_up, throw_down, throw_left, throw_right, zoom_in and zoom_out.

<img src='data/show.gif'  width="50%"> 

### Structure
``` bash
.
├── data
│   └── test.mp4
├── main.py
├── network
│   ├── detector.py
│   ├── gesture.py
│   ├── infer.py
│   ├── __init__.py
│   └── skeleton.py
├── onnx_model
│   ├── gesture.onnx
│   ├── hand_detection.onnx
│   └── landmark.onnx
├── README.md
└── utils
    ├── draw.py
    └── __init__.py

```
### Run
``` bash
python main.py
```

### TODO:
- [] model compression
- [] more gesture classes

___
## Reference
+ https://gitcode.net/EricLee/classification
+ https://gitcode.net/EricLee/handpose_x?from_codechina=yes
+ https://github.com/yuxiaochen1103/DG-STA

