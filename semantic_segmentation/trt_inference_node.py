#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import os
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

COLOR_PALETTE = [
    [127, 127, 127],  # background = gray
    [44, 160, 44],    # stable = green
    [255, 127, 14],   # granular = brown
    [140, 86, 75],    # poor foothold = orange
    [214, 39, 40],    # high resistance = red
    [31, 119, 180]    # obstacle = blue
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def input_transform(image):
    """Transform input image for the model"""
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= MEAN
    image /= STD
    return image

class TRTInferenceNode(Node):
    def __init__(self):
        super().__init__('trt_inference_node')
        self.bridge = CvBridge()

        self.declare_parameter('inference_fps', 20.0)
        self.declare_parameter('model_path', '')
        self.declare_parameter('verbosity', 0)

        self.declare_parameter('input_width', 1024)
        self.declare_parameter('input_height', 544)

        self.desired_fps = self.get_parameter('inference_fps').value
        self.model_path = self.get_parameter('model_path').value
        self.verbosity = self.get_parameter('verbosity').value
        self.input_width = self.get_parameter('input_width').value
        self.input_height = self.get_parameter('input_height').value

        self.input_width = (self.input_width // 8) * 8
        self.input_height = (self.input_height // 8) * 8

        self.get_logger().info(f"Running with fixed input dimensions: {self.input_width}x{self.input_height}")
        self.get_logger().info(f"FPS limit: {self.desired_fps if self.desired_fps > 0 else 'Unlimited'}")

        self.load_trt_engine()

        self.subscription = self.create_subscription(
            CompressedImage,
            '/zed/zed_node/left_raw/image_raw_color/compressed',
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/left/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.camera_info_msg = None
        
        self.latest_msg = None
        self.processing = False
        self.last_process_time = self.get_clock().now()
        
        if self.desired_fps > 0:
            timer_period = 1.0 / self.desired_fps
            self.timer = self.create_timer(timer_period, self.timer_callback)

        self.mask_pub = self.create_publisher(Image, '/segmentation_result/image_raw', 10)
        self.mask_info_pub = self.create_publisher(CameraInfo, '/segmentation_result/camera_info', 10)

        self.recent_inference_times = []
        self.recent_total_times = []
        self.window_size = 10
        self.frame_count = 0
        self.last_fps_print_time = self.get_clock().now()
        self.fps_print_interval = 3.0
        
        self.get_logger().info("TensorRT inference node initialized")

    def load_trt_engine(self):
        """Load TensorRT engine from file"""
        try:
            if not os.path.exists(self.model_path):
                self.get_logger().error(f"TensorRT engine not found at {self.model_path}")
                return

            self.get_logger().info(f"Loading TensorRT engine: {self.model_path}")
            with open(self.model_path, "rb") as f:
                self.serialized_engine = f.read()

            self.runtime = trt.Runtime(TRT_LOGGER)

            temp_engine = self.runtime.deserialize_cuda_engine(self.serialized_engine)

            self.input_name = temp_engine.get_tensor_name(0)
            self.output_name = temp_engine.get_tensor_name(1)
            self.expert_name = temp_engine.get_tensor_name(2) if temp_engine.num_io_tensors > 2 else None

            self.raw_height = self.input_height // 8
            self.raw_width = self.input_width // 8

            del temp_engine
            
            self.get_logger().info("TensorRT engine loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load TensorRT engine: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera info message"""
        self.camera_info_msg = msg

    def image_callback(self, msg: CompressedImage):
        """Handle incoming compressed images"""
        if self.desired_fps > 0:
            self.latest_msg = msg
        else:
            if not self.processing:
                self.processing = True
                self.process_image(msg)
                self.processing = False

    def timer_callback(self):
        """Process images at the desired rate"""
        if self.latest_msg is not None and not self.processing:
            now = self.get_clock().now()
            elapsed = (now - self.last_process_time).nanoseconds / 1e9

            if elapsed >= (1.0 / self.desired_fps):
                self.processing = True
                self.process_image(self.latest_msg)
                self.latest_msg = None
                self.last_process_time = now
                self.processing = False

    def process_image(self, msg: CompressedImage):
        """Process an image with the TensorRT engine"""
        try:
            total_start_time = time.time()

            np_arr = np.frombuffer(msg.data, np.uint8)
            original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            original_height, original_width = original_image.shape[:2]

            resized_image = cv2.resize(original_image, (self.input_width, self.input_height))

            img_t = input_transform(resized_image)
            img_t = img_t.transpose((2, 0, 1)).copy()
            img_t = np.expand_dims(img_t, axis=0)

            engine = self.runtime.deserialize_cuda_engine(self.serialized_engine)
            context = engine.create_execution_context()

            context.set_input_shape(self.input_name, (1, 3, self.input_height, self.input_width))

            raw_output_shape = (1, 6, self.raw_height, self.raw_width)
            raw_output_size = int(np.prod(raw_output_shape) * np.dtype(np.float32).itemsize)

            d_input = cuda.mem_alloc(img_t.nbytes)
            d_raw_output = cuda.mem_alloc(raw_output_size)
            d_expert = cuda.mem_alloc(8) if self.expert_name else None

            h_raw_output = cuda.pagelocked_empty(int(np.prod(raw_output_shape)), dtype=np.float32)
            h_expert = cuda.pagelocked_empty(1, dtype=np.int64) if self.expert_name else None

            stream = cuda.Stream()

            cuda.memcpy_htod_async(d_input, img_t, stream)

            context.set_tensor_address(self.input_name, int(d_input))
            context.set_tensor_address(self.output_name, int(d_raw_output))
            if self.expert_name and d_expert:
                context.set_tensor_address(self.expert_name, int(d_expert))

            cuda.Context.synchronize()
            start_time = time.time()
            
            context.execute_async_v3(stream_handle=stream.handle)

            cuda.memcpy_dtoh_async(h_raw_output, d_raw_output, stream)
            if self.expert_name and h_expert is not None and d_expert:
                cuda.memcpy_dtoh_async(h_expert, d_expert, stream)

            stream.synchronize()
            inference_time = (time.time() - start_time) * 1000  # ms

            selected_expert = None
            if self.expert_name and h_expert is not None:
                selected_expert = int(h_expert[0])
                if self.verbosity >= 2:
                    self.get_logger().debug(f"Selected expert: {selected_expert}")

            self.recent_inference_times.append(inference_time)
            if len(self.recent_inference_times) > self.window_size:
                self.recent_inference_times.pop(0)

            raw_pred = h_raw_output.reshape(raw_output_shape)

            if np.all(raw_pred == 0) and self.frame_count == 0:
                self.get_logger().warn("Model output is all zeros! Check model compatibility.")

            raw_tensor = torch.from_numpy(raw_pred)
            if torch.cuda.is_available():
                raw_tensor = raw_tensor.cuda()

            pred = F.interpolate(
                raw_tensor,
                size=(self.input_height, self.input_width),
                mode='bilinear',
                align_corners=True
            )

            pred = torch.argmax(pred, dim=1).squeeze(0)
            if pred.is_cuda:
                pred = pred.cpu()
            pred = pred.numpy()

            colored_mask = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
            for class_id, color in enumerate(COLOR_PALETTE):
                mask = (pred == class_id)
                colored_mask[mask] = color

            if original_height != self.input_height or original_width != self.input_width:
                colored_mask = cv2.resize(colored_mask, (original_width, original_height), 
                                         interpolation=cv2.INTER_NEAREST)

            out_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding="rgb8")

            if self.camera_info_msg is not None:
                out_msg.header = self.camera_info_msg.header
            else:
                out_msg.header.stamp = self.get_clock().now().to_msg()
                out_msg.header.frame_id = "camera_link"

            self.mask_pub.publish(out_msg)

            if self.camera_info_msg is not None:
                self.mask_info_pub.publish(self.camera_info_msg)

            d_input.free()
            d_raw_output.free()
            if d_expert:
                d_expert.free()
            del context
            del engine

            total_time = (time.time() - total_start_time) * 1000
            self.recent_total_times.append(total_time)
            if len(self.recent_total_times) > self.window_size:
                self.recent_total_times.pop(0)

            self.frame_count += 1

            now = self.get_clock().now()
            if (now - self.last_fps_print_time).nanoseconds / 1e9 >= self.fps_print_interval:
                avg_inference_time = sum(self.recent_inference_times) / len(self.recent_inference_times)
                avg_total_time = sum(self.recent_total_times) / len(self.recent_total_times)
                
                inference_fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0.0
                total_fps = 1000.0 / avg_total_time if avg_total_time > 0 else 0.0
                
                if self.verbosity >= 1:
                    expert_str = f", Expert: {selected_expert}" if selected_expert is not None else ""
                    self.get_logger().info(f"Inference: {avg_inference_time:.2f} ms{expert_str}, Pure Inference FPS: {inference_fps:.1f}, Actual FPS: {total_fps:.1f}")
                else:
                    self.get_logger().info(f"FPS: {total_fps:.1f}")
                
                self.last_fps_print_time = now
                
        except Exception as e:
            self.get_logger().error(f"Error in process_image: {str(e)}")
            if self.verbosity >= 2:
                import traceback
                self.get_logger().error(traceback.format_exc())
            
            try:
                if 'd_input' in locals(): d_input.free()
                if 'd_raw_output' in locals(): d_raw_output.free()
                if 'd_expert' in locals(): d_expert.free()
                if 'context' in locals(): del context
                if 'engine' in locals(): del engine
            except:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = TRTInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()