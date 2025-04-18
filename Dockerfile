FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  gnupg \
  wget \
  software-properties-common \
  python3-pip \
  python3-opencv \
  ros-humble-rmw-cyclonedds-cpp \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  rm cuda-keyring_1.1-1_all.deb && \
  apt-get update && apt-get install -y --no-install-recommends \
  cuda-toolkit-12-8 \
  && rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/cuda-12.8/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}

RUN pip3 install --no-cache-dir \
  torch \
  torchvision \
  numpy \
  pycuda \
  tensorrt

RUN apt-get update && apt-get install -y \
  ros-humble-cv-bridge \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws/src
COPY . /ros2_ws/src/semantic_segmentation/

WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && \
  colcon build --symlink-install

COPY ./ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]