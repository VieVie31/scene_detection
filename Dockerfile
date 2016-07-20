FROM kavolorn/opencv


RUN echo "deb http://httpredir.debian.org/debian jessie-backports main" > \
        /etc/apt/sources.list.d/backports.list && \
        apt-get clean && \
        apt-get update && \
        apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libavformat-dev \
        libpq-dev \
        libfreetype6-dev \
        ffmpeg \
        libopencv-dev \
        python3-matplotlib \
        python3-numpy \
        python3-scipy \
        python3-pip \
        && \
        rm -rf /var/cache/apt/*


RUN pip3 install --upgrade \
    flake8 \
    pep8 \
    tqdm

WORKDIR /src/
CMD bash main.sh
