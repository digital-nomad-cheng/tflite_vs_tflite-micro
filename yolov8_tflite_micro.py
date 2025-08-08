import time
from yolov8_tflite import YOLOv8TFLite
import cv2
import numpy as np
import yaml
from tflite_micro.python.tflite_micro import runtime
from typing import Union
from ultralytics.utils import ASSETS
import argparse

class YOLOv8TFLM(YOLOv8TFLite):
    def __init__(self, model: str, conf: float = 0.25, iou: float = 0.45, metadata: Union[str, None] = None):
        self.conf = conf
        self.iou = iou
        if metadata is None:
            self.classes = {i: i for i in range(1000)}
        else:
            with open(metadata) as f:
                self.classes = yaml.safe_load(f)["names"]
        np.random.seed(42)  # Set seed for reproducible colors
        self.color_palette = np.random.uniform(128, 255, size=(len(self.classes), 3))

        # Initialize the TFLite interpreter
        self.model = runtime.Interpreter.from_file(model_path=model)

        # Get input details
        input_details = self.model.get_input_details(0)
        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_scale, self.in_zero_point = input_details["quantization_parameters"]["scales"][0], input_details["quantization_parameters"]["zero_points"][0]
        self.int8 = input_details["dtype"] == np.int8

        # Get output details
        output_details = self.model.get_output_details(0)
        self.out_scale, self.out_zero_point = output_details["quantization_parameters"]["scales"][0], output_details["quantization_parameters"]["zero_points"][0]
    
    def detect(self, img_path: str) -> np.ndarray:
        """
        Perform object detection on an input image.

        Args:
            img_path (str): Path to the input image file.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        # Load and preprocess image
        img = cv2.imread(img_path)
        x, pad = self.preprocess(img)

        # Apply quantization if model is int8
        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        # Set input tensor and run inference
        self.model.set_input(x, 0)
        start = time.time()
        self.model.invoke()
        end = time.time()
        print("Inference time:", end - start)

        # Get output and dequantize if necessary
        y = self.model.get_output(0)
        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale

        # Process detections and return result
        return self.postprocess(img, y, pad)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n_saved_model/yolov8n_full_integer_quant.tflite",
        help="Path to TFLite model.",
    )
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--metadata", type=str, default="yolov8n_saved_model/metadata.yaml", help="Metadata yaml")
    args = parser.parse_args()

    detector = YOLOv8TFLM(args.model, args.conf, args.iou, args.metadata)
    result = detector.detect(args.img)

    cv2.imshow("Output", result)
    cv2.waitKey(0)
