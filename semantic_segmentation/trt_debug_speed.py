#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import os
import time
import threading
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

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

class TRTInferenceNode(Node):
    def __init__(self):
        super().__init__('trt_debug_speed')
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

        self.cuda_initialized = False

        self.raw_height = self.input_height // 8
        self.raw_width = self.input_width // 8
        self.raw_output_shape = (1, 6, self.raw_height, self.raw_width)

        self.load_trt_engine()

        self.input_shape = (1, 3, self.input_height, self.input_width)
        self.input_size = int(np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
        self.output_size = int(np.prod(self.raw_output_shape) * np.dtype(np.float32).itemsize)

        self.cuda_setup()

        self.resized_image = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
        self.transposed_input = np.zeros((3, self.input_height, self.input_width), dtype=np.float32)
        self.input_tensor = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)

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
        self.cuda_lock = threading.Lock()
        
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

        self.timing_stats = {
            'decompress': [],
            'preprocess': [],
            'cuda_setup': [],
            'transfer_to_gpu': [],
            'inference': [],
            'transfer_from_gpu': [],
            'postprocess': [],
            'ros_publish': [],
            'cleanup': []
        }
        
        self.get_logger().info("TensorRT inference node initialized")

    def load_trt_engine(self):
        """Load TensorRT engine from file (done once at startup)"""
        try:
            if not os.path.exists(self.model_path):
                self.get_logger().error(f"TensorRT engine not found at {self.model_path}")
                return

            self.get_logger().info(f"Loading TensorRT engine: {self.model_path}")
            with open(self.model_path, "rb") as f:
                self.serialized_engine = f.read()

            self.runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(self.serialized_engine)

            self.input_name = self.engine.get_tensor_name(0)
            self.output_name = self.engine.get_tensor_name(1)
            self.expert_name = self.engine.get_tensor_name(2) if self.engine.num_io_tensors > 2 else None
            
            self.get_logger().info("TensorRT engine loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load TensorRT engine: {str(e)}")
            if self.verbosity >= 3:
                import traceback
                self.get_logger().error(traceback.format_exc())

    def cuda_setup(self):
        """Set up CUDA resources (done once at startup)"""
        try:
            self.exec_context = self.engine.create_execution_context()

            self.stream = cuda.Stream()

            self.d_input = cuda.mem_alloc(int(self.input_size))
            self.d_output = cuda.mem_alloc(int(self.output_size))
            self.d_expert = cuda.mem_alloc(8) if self.expert_name else None

            self.h_input = cuda.pagelocked_empty((1, 3, self.input_height, self.input_width), dtype=np.float32)
            self.h_output = cuda.pagelocked_empty(int(np.prod(self.raw_output_shape)), dtype=np.float32)
            self.h_expert = cuda.pagelocked_empty(1, dtype=np.int64) if self.expert_name else None

            self.exec_context.set_input_shape(self.input_name, (1, 3, self.input_height, self.input_width))
            self.exec_context.set_tensor_address(self.input_name, int(self.d_input))
            self.exec_context.set_tensor_address(self.output_name, int(self.d_output))
            if self.expert_name and self.d_expert:
                self.exec_context.set_tensor_address(self.expert_name, int(self.d_expert))

            self.cuda_initialized = True
            
        except Exception as e:
            self.get_logger().error(f"Failed to set up CUDA resources: {str(e)}")
            if self.verbosity >= 3:
                import traceback
                self.get_logger().error(traceback.format_exc())
            self.cuda_initialized = False

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

    def fast_preprocess(self, img_bgr):
        """Optimized preprocessing using pre-allocated buffers"""
        resized = cv2.resize(img_bgr, (self.input_width, self.input_height))

        img_rgb = resized[..., ::-1].astype(np.float32) / 255.0
        img_rgb = (img_rgb - MEAN) / STD

        img_chw = img_rgb.transpose(2, 0, 1).copy()

        return np.expand_dims(img_chw, axis=0)

    def update_timing(self, stage, time_ms):
        """Update timing statistics for a processing stage"""
        self.timing_stats[stage].append(time_ms)
        if len(self.timing_stats[stage]) > self.window_size:
            self.timing_stats[stage].pop(0)

    def process_image(self, msg: CompressedImage):
        """Process an image with the TensorRT engine"""
        try:
            if not self.cuda_initialized:
                self.get_logger().warn("CUDA resources not properly initialized, falling back to per-frame initialization")

            total_start_time = time.time()
            
            #---------- STAGE 1: Image Decompression ----------
            t_start = time.time()

            np_arr = np.frombuffer(msg.data, np.uint8)
            original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            original_height, original_width = original_image.shape[:2]
            
            t_decompress = (time.time() - t_start) * 1000
            self.update_timing('decompress', t_decompress)
            
            #---------- STAGE 2: Preprocessing ----------
            t_start = time.time()

            input_data = self.fast_preprocess(original_image)

            if self.cuda_initialized:
                np.copyto(self.h_input, input_data)
            
            t_preprocess = (time.time() - t_start) * 1000
            self.update_timing('preprocess', t_preprocess)
            
            if self.cuda_initialized:
                #---------- STAGE 3: CUDA Setup (already done) ----------
                self.update_timing('cuda_setup', 0.0)
                
                #---------- STAGE 4: Transfer to GPU ----------
                t_start = time.time()

                with self.cuda_lock:
                    cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
                    self.stream.synchronize()
                    
                t_to_gpu = (time.time() - t_start) * 1000
                self.update_timing('transfer_to_gpu', t_to_gpu)
                
                #---------- STAGE 5: Inference ----------
                t_start = time.time()

                with self.cuda_lock:
                    cuda.Context.synchronize()
                    self.exec_context.execute_async_v3(stream_handle=self.stream.handle)
                    self.stream.synchronize()
                    cuda.Context.synchronize()
                    
                inference_time = (time.time() - t_start) * 1000
                self.update_timing('inference', inference_time)
                
                #---------- STAGE 6: Transfer from GPU ----------
                t_start = time.time()

                with self.cuda_lock:
                    cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
                    if self.expert_name and self.h_expert is not None:
                        cuda.memcpy_dtoh_async(self.h_expert, self.d_expert, self.stream)
                    self.stream.synchronize()
                    
                t_from_gpu = (time.time() - t_start) * 1000
                self.update_timing('transfer_from_gpu', t_from_gpu)

                selected_expert = None
                if self.expert_name and self.h_expert is not None:
                    selected_expert = int(self.h_expert[0])

                raw_pred = self.h_output.reshape(self.raw_output_shape)
            else:
                self.get_logger().error("Using fallback path - CUDA not initialized properly")
                self.update_timing('cuda_setup', 0.0)
                self.update_timing('transfer_to_gpu', 0.0)
                self.update_timing('inference', 0.0)
                self.update_timing('transfer_from_gpu', 0.0)

                raw_pred = np.zeros(self.raw_output_shape, dtype=np.float32)
                selected_expert = 0
                inference_time = 0.0
            
            #---------- STAGE 7: Post-processing ----------
            t_start = time.time()

            self.recent_inference_times.append(inference_time)
            if len(self.recent_inference_times) > self.window_size:
                self.recent_inference_times.pop(0)

            raw_classes = np.argmax(raw_pred, axis=1).squeeze(0).astype(np.uint8)

            upscaled = cv2.resize(raw_classes, 
                                 (self.input_width, self.input_height), 
                                 interpolation=cv2.INTER_NEAREST)

            colored_mask = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
            for class_id, color in enumerate(COLOR_PALETTE):
                mask = (upscaled == class_id)
                colored_mask[mask] = color

            if original_height != self.input_height or original_width != self.input_width:
                colored_mask = cv2.resize(colored_mask, (original_width, original_height), 
                                 interpolation=cv2.INTER_NEAREST)
            
            t_postprocess = (time.time() - t_start) * 1000
            self.update_timing('postprocess', t_postprocess)
            
            #---------- STAGE 8: ROS Publishing ----------
            t_start = time.time()

            out_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding="rgb8")

            if self.camera_info_msg is not None:
                out_msg.header = self.camera_info_msg.header
            else:
                out_msg.header.stamp = self.get_clock().now().to_msg()
                out_msg.header.frame_id = "camera_link"

            self.mask_pub.publish(out_msg)

            if self.camera_info_msg is not None:
                self.mask_info_pub.publish(self.camera_info_msg)
            
            t_ros_publish = (time.time() - t_start) * 1000
            self.update_timing('ros_publish', t_ros_publish)
            
            #---------- STAGE 9: Cleanup ----------

            self.update_timing('cleanup', 0.0)

            total_time = (time.time() - total_start_time) * 1000  # ms
            self.recent_total_times.append(total_time)
            if len(self.recent_total_times) > self.window_size:
                self.recent_total_times.pop(0)

            self.frame_count += 1
            
            now = self.get_clock().now()
            if (now - self.last_fps_print_time).nanoseconds / 1e9 >= self.fps_print_interval:
                avg_inference_time = sum(self.recent_inference_times) / len(self.recent_inference_times) if self.recent_inference_times else 0
                avg_total_time = sum(self.recent_total_times) / len(self.recent_total_times) if self.recent_total_times else 0
                
                inference_fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0.0
                total_fps = 1000.0 / avg_total_time if avg_total_time > 0 else 0.0
                
                if self.verbosity == 0:
                    self.get_logger().info(f"FPS: {total_fps:.1f}")
                elif self.verbosity == 1:
                    expert_str = f", Expert: {selected_expert}" if selected_expert is not None else ""
                    self.get_logger().info(f"Inference: {avg_inference_time:.2f} ms{expert_str}, Pure Inference FPS: {inference_fps:.1f}, Actual FPS: {total_fps:.1f}")
                elif self.verbosity >= 2:
                    self.get_logger().info(f"Performance Breakdown (ms):")
                    for stage in self.timing_stats:
                        if self.timing_stats[stage]:
                            avg = sum(self.timing_stats[stage])/len(self.timing_stats[stage])
                            self.get_logger().info(f"  {stage.capitalize()}: {avg:.2f}")
                    self.get_logger().info(f"  Total: {avg_total_time:.2f}, FPS: {total_fps:.1f}")
                
                self.last_fps_print_time = now
                
        except Exception as e:
            self.get_logger().error(f"Error in process_image: {str(e)}")
            if self.verbosity >= 3:
                import traceback
                self.get_logger().error(traceback.format_exc())

    def __del__(self):
        """Clean up CUDA resources on shutdown"""
        try:
            if hasattr(self, 'd_input') and hasattr(self, 'cuda_initialized') and self.cuda_initialized:
                self.d_input.free()
            if hasattr(self, 'd_output') and hasattr(self, 'cuda_initialized') and self.cuda_initialized:
                self.d_output.free()
            if hasattr(self, 'd_expert') and self.d_expert and hasattr(self, 'cuda_initialized') and self.cuda_initialized:
                self.d_expert.free()
            if hasattr(self, 'exec_context'):
                del self.exec_context
            if hasattr(self, 'engine'):
                del self.engine
        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = TRTInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()