from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'semantic_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='basil',
    maintainer_email='basil@mannaerts.dev',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trt_inference_node = semantic_segmentation.trt_inference_node:main',
            'trt_debug_speed = semantic_segmentation.trt_debug_speed:main',
        ],
    },
)