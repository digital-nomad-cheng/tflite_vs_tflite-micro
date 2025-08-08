# tflite_vs_tflite-micro

## Install
If you are using this repo, all scripts are already provided. You don't need to do anything but install the environment.

1. create a virtual env: `uv venv .env`
2. install packages `uv pip install -r requirements`
3. download python scripts from [ultralytics](https://github.com/ultralytics/ultralytics/) library: `wget -O yolov8_tflite.py https://raw.githubusercontent.com/ultralytics/ultralytics/refs/heads/main/examples/YOLOv8-TFLite-Python/main.py`
4. provide tflite-micro script which is subclass of `YOLOv8TFLite` in tflite script.

## Run 
1. export tflite model: `yolo export model=yolov8n.pt format=tflite int8=True`
2. get tflite prediction result:`python yolov8_tflite.py`
    ```
    Inference time: 0.03912806510925293
    person: 0.89
    person: 0.89
    person: 0.89
    bus: 0.87
    person: 0.34
    ```
3. get tflite-micro prediction result: `python yolov8_tflite_micro.py`
    ```
    Inference time: 76.6522068977356
    person: 0.89
    person: 0.89
    person: 0.89
    bus: 0.87
    person: 0.39
    ```