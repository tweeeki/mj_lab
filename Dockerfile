# Training-only container for the mjlab (unitree_rl_mjlab) stack.
# Build context = this folder (mj_lab/mjlab). Nothing else in mj_lab is needed.
#
#   docker build -t mjlab-train .
#   docker run --gpus all --rm -v "$(pwd)/logs:/workspace/mjlab/logs" mjlab-train \
#       scripts/train.py --task Unitree-G1-Reach --num-envs 4096 --max-iterations 5001
#
# Requires an NVIDIA GPU on the host + nvidia-container-toolkit (Brev has both).

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # mjlab renders offscreen via EGL through the NVIDIA driver; "graphics" cap
    # is required or EGL context creation fails.
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl

# Python 3.10 (mjlab requires >=3.10,<3.14) + EGL/GL runtime libs for MuJoCo.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip \
      libegl1 libgl1 libglib2.0-0 libgomp1 \
      git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install the CUDA 12.8 torch build first so the mjlab dep resolver is satisfied
# by it (mjlab needs torch>=2.7.0) instead of pulling a CPU wheel.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip \
    && python3.10 -m pip install --no-cache-dir \
       torch --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace/mjlab
COPY . /workspace/mjlab

# Editable install of unitree_rl_mjlab: pulls mjlab==1.2.0 + mujoco-warp==3.5.0
# and makes the local `src` task package importable (train.py does `import src.tasks`).
RUN python3.10 -m pip install --no-cache-dir -e .

# Training writes checkpoints + policy.onnx + deploy.yaml here; mount a host
# volume over it to keep the artifacts after the container exits.
VOLUME /workspace/mjlab/logs

# ENTRYPOINT is the interpreter, so `docker run ... <script> <args>` just works.
ENTRYPOINT ["python3.10"]
CMD ["scripts/train.py", "--task", "Unitree-G1-Reach", "--num-envs", "4096", "--max-iterations", "5001"]
