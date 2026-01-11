# Base image with CUDA + cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

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

# Default environment variables
ENV BATCH_SIZE=64
ENV NGPUS=1
ENV PRECISION_FLAG=""

# Entry point runs training
ENTRYPOINT ["torchrun", "--nproc_per_node=1", "main.py", "--data-backend", "synthetic", "--arch", "resnet50", "--training-only", "--no-checkpoints", "--amp", "-b", "32", "--run-epochs", "1", "--prof", "300", "--raport-file", "/results/output.json", "/workspace/dataset"]



