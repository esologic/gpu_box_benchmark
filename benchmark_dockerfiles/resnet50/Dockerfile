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

# ---------------------------
# Benchmark parameters
# ---------------------------
ENV BATCH_SIZE=32
ENV NGPUS=1
ENV AMP=1
ENV NUM_WORKERS=8

# ---------------------------
# Fixed benchmark entrypoint
# ---------------------------
ENTRYPOINT ["/bin/bash", "-lc", "\
torchrun \
  --nproc_per_node=${NGPUS} \
  main.py \
    --arch resnet50 \
    --data-backend synthetic \
    --training-only \
    --no-checkpoints \
    --run-epochs 1 \
    --prof 300 \
    --workers ${NUM_WORKERS} \
    -b ${BATCH_SIZE} \
    $( [ \"${AMP}\" = \"1\" ] && echo \"--amp\" ) \
    --raport-file /results/output.txt \
    /workspace/dataset \
"]
