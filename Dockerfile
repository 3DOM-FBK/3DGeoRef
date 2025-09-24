FROM ubuntu:22.04

# --- Variables ---
ENV DEBIAN_FRONTEND=noninteractive
ENV BLENDER_VERSION=4.4.0
ENV BLENDER_URL=https://download.blender.org/release/Blender4.4/blender-${BLENDER_VERSION}-linux-x64.tar.xz

RUN apt-get update && apt-get install -y \
    wget \
    tar \
    python3 \
    python3-pip \
    libgl1 \
    libx11-6 \
    libxi6 \
    libxcursor1 \
    libxrandr2 \
    libxinerama1 \
    libxxf86vm1 \
    libegl1 \
    libdbus-1-3 \
    libnss3 \
    libxss1 \
    libasound2 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    && rm -rf /var/lib/apt/lists/*

# Install Blender 4.4 (Release or Alpha)
WORKDIR /opt
RUN wget https://download.blender.org/release/Blender4.4/blender-${BLENDER_VERSION}-linux-x64.tar.xz && \
    tar -xf blender-${BLENDER_VERSION}-linux-x64.tar.xz && \
    ln -s /opt/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender && \
    rm blender-${BLENDER_VERSION}-linux-x64.tar.xz

# --- Python packages ---
RUN pip3 install --upgrade pip
RUN pip3 install numpy Pillow rasterio

# --- GeoCLIP installation ---
RUN pip3 install geoclip pyproj

RUN apt update && apt install -y \
    **libxkbcommon0** \
    **libsm6**

# --- Deep Image Matching ---
RUN pip3 install pycolmap

RUN apt update && apt install -y \
    libglib2.0-0

WORKDIR /opt/
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    ffmpeg

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone repo - dev branch
RUN git clone https://github.com/3DOM-FBK/deep-image-matching.git /workspace/dim
WORKDIR /workspace/dim

# Checkout the specified branch
RUN git checkout dev

RUN sed -i '30,31s/^# *//' /workspace/dim/demo.py

# RUN sed -i 's/torch.load(\(.*\)map_location=self._device)/torch.load(\1map_location=self._device, weights_only=False)/' /workspace/dim/src/deep_image_matching/matchers/se2loftr.py
# RUN sed -i "s|fr'{input_dir}\\\\features.h5'|os.path.join(input_dir, 'features.h5')|g" /workspace/dim/src/deep_image_matching/utils/loftr_roma_to_multiview.py
# RUN sed -i "s|fr'{input_dir}\\\\matches.h5'|os.path.join(input_dir, 'matches.h5')|g" /workspace/dim/src/deep_image_matching/utils/loftr_roma_to_multiview.py
# RUN sed -i "s|fr'{input_dir}\\\\matches_loftr.h5'|os.path.join(input_dir, 'matches_loftr.h5')|g" /workspace/dim/src/deep_image_matching/utils/loftr_roma_to_multiview.py
# RUN sed -i "s|fr'{output_dir}\\\\keypoints.h5'|fr'{output_dir}/keypoints.h5'|g" /workspace/dim/src/deep_image_matching/utils/loftr_roma_to_multiview.py
# RUN sed -i "s|fr'{output_dir}\\\\matches.h5'|fr'{output_dir}/matches.h5'|g" /workspace/dim/src/deep_image_matching/utils/loftr_roma_to_multiview.py

# Install deep-image-matching
RUN python3 -m pip install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install torch torchvision
RUN pip3 install -e .

RUN rm -rf /var/lib/apt/lists/*

# install trimesh
RUN pip3 install trimesh

RUN pip3 install enlighten

# --- Working directory ---
WORKDIR /app