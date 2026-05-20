FROM ubuntu:22.04 

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# Install python, system dependencies, utils
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    python${PYTHON_VERSION}-dev \
    build-essential \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    patchelf \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create alias for python3 <=> python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# We copy ONLY the requirements file from your laptop into the image so that the pip install can be run using this cached data unless the file is changed
COPY setup/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Remove cv2/typing directory that shadows stdlib typing in spawn workers
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/typing

# this is a dummy argument that we can use to force the cache to bust and the image to be rebuilt when we want to update the code in the repo, since the code is cloned in the same layer as this argument, any change to this argument will cause the code to be re-cloned and thus updated in the image
ARG CACHEBUST=1

# Clone & setup repo - fixed version per image is standard practice, so try this
RUN git clone https://github.com/Lavansurendra/berkeley_humanoid.git && \
    cd berkeley_humanoid && \
    git checkout temp-CHTC && \
    git pull origin temp-CHTC
RUN mv berkeley_humanoid workspace

RUN date > /image_build_date.txt

WORKDIR workspace


CMD ["/bin/bash"]