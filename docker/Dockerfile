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

## Clone & setup repo - fixed version per image is standard practice, but ig we'll do this instead
#RUN git clone https://github.com/Lavansurendra/berkeley_humanoid.git
#RUN mv berkeley_humanoid workspace

#WORKDIR workspace
COPY ../setup/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]