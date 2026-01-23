# Base image with CUDA + cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG SESSION_ID=fallback

LABEL gpu_box_benchmark_session=$SESSION_ID

# Install Python, pip, git
RUN apt-get update && apt-get install -y python3-pip git
RUN pip install --upgrade pip

# Install PyTorch + torchvision (CUDA 11.8)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Clone NVIDIA DeepLearningExamples
RUN git clone https://github.com/NVIDIA/DeepLearningExamples.git /workspace/DeepLearningExamples

# Install Python dependencies for NVIDIA ConvNets
RUN pip install -r /workspace/DeepLearningExamples/PyTorch/Classification/ConvNets/requirements.txt
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/ nvidia-dali-cuda110

WORKDIR /workspace/DeepLearningExamples/PyTorch/Classification/ConvNets

# Copy the bash script into the container
COPY ./run_resnet_benchmark.sh /workspace/DeepLearningExamples/PyTorch/Classification/ConvNets/run_resnet_benchmark.sh
RUN chmod +x /workspace/DeepLearningExamples/PyTorch/Classification/ConvNets/run_resnet_benchmark.sh

# Set ENTRYPOINT to the script
ENTRYPOINT ["/workspace/DeepLearningExamples/PyTorch/Classification/ConvNets/run_resnet_benchmark.sh"]