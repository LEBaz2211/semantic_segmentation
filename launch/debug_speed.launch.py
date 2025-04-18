from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('semantic_segmentation')

    default_model_path = os.path.join(pkg_dir, 'models', 'pidnet_large_c6_fp16_v1.trt')

    model_path = LaunchConfiguration('model_path', default=default_model_path)
    inference_fps = LaunchConfiguration('inference_fps', default='0.0')
    input_width = LaunchConfiguration('input_width', default='1024')
    input_height = LaunchConfiguration('input_height', default='544')
    verbosity = LaunchConfiguration('verbosity', default='0')

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model_path,
        description='Path to the TensorRT model file'
    )
    
    inference_fps_arg = DeclareLaunchArgument(
        'inference_fps',
        default_value='0.0',
        description='Target inference FPS (0 for no rate limiting)'
    )
    
    input_width_arg = DeclareLaunchArgument(
        'input_width',
        default_value='1024',
        description='Input width for the model (must be divisible by 8)'
    )
    
    input_height_arg = DeclareLaunchArgument(
        'input_height',
        default_value='544',
        description='Input height for the model (must be divisible by 8)'
    )
    
    verbosity_arg = DeclareLaunchArgument(
        'verbosity',
        default_value='0',
        description='Verbosity level (0=minimal, 1=normal, 2=detailed timing, 3=debug)'
    )

    inference_node = Node(
        package='semantic_segmentation',
        executable='trt_debug_speed',
        name='trt_debug_speed',
        parameters=[{
            'model_path': model_path,
            'inference_fps': inference_fps,
            'input_width': input_width,
            'input_height': input_height,
            'verbosity': verbosity
        }],
        output='screen'
    )
    
    return LaunchDescription([
        model_path_arg,
        inference_fps_arg,
        input_width_arg,
        input_height_arg,
        verbosity_arg,
        inference_node
    ])