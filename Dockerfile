FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# --- Variables ---
ENV DEBIAN_FRONTEND=noninteractive
ENV BLENDER_VERSION=4.4.0
ENV BLENDER_URL=https://download.blender.org/release/Blender4.4/blender-${BLENDER_VERSION}-linux-x64.tar.xz
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:/root/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
#     SYSTEM DEPENDENCIES
# -----------------------------
RUN apt-get update && apt-get install -y \
    wget tar git curl ffmpeg \
    python3 python3-pip \
    libgl1 libx11-6 libxi6 libxcursor1 libxrandr2 libxinerama1 \
    libxxf86vm1 libegl1 libdbus-1-3 libnss3 libxss1 libasound2 \
    libxcomposite1 libxdamage1 libxext6 libxfixes3 \
    libxkbcommon0 libsm6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
#            BLENDER
# -----------------------------
WORKDIR /opt
RUN wget $BLENDER_URL && \
    tar -xf blender-${BLENDER_VERSION}-linux-x64.tar.xz && \
    ln -s /opt/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender && \
    rm blender-${BLENDER_VERSION}-linux-x64.tar.xz

# -----------------------------
#         PYTHON PACKAGES
# -----------------------------
RUN pip3 install --upgrade pip && \
    pip3 install \
        numpy Pillow rasterio geoclip pyproj \
        google-genai \
        pycolmap==3.12.6 \
        trimesh open3d \
        enlighten pygeodesy \
        ollama \
        setuptools && \
    pip3 install torch torchvision

ENV OLLAMA_HOST=http://ollama:11434

# -----------------------------
#   DEEP IMAGE MATCHING SETUP
# -----------------------------
RUN git clone https://github.com/3DOM-FBK/deep-image-matching.git /workspace/dim
WORKDIR /workspace/dim
RUN git checkout dev

# attiva parse_cli nel demo.py
RUN sed -i 's|# from deep_image_matching.parser import parse_cli|from deep_image_matching.parser import parse_cli|' demo.py && \
    sed -i 's|# args = parse_cli()|args = parse_cli()|' demo.py

# install DIM in editable mode
RUN pip3 install -e .

# -----------------------------
#        WORK DIRECTORY
# -----------------------------
WORKDIR /app

# -----------------------------
#        COPY FILES
# -----------------------------
COPY pipeline/ /app/pipeline/
COPY hdri/ /app/hdri/
COPY main.py /app/main.py